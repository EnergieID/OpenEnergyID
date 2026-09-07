"""Pre-flight checks for contract-based cost calculation.

``energy_cost`` resolves price indexes by name from a process-global registry, and does
so deep inside the calculation. An unregistered name therefore surfaces as a bare
``KeyError`` after the meters are already built. These helpers front-load that failure
and surface the conditions that make a cost result misleading rather than wrong.
"""

import datetime as dt
import logging
from typing import Any

import isodate
import pandas as pd
from energy_cost import Contract, ContractHistory
from energy_cost.formula import IndexAdder
from energy_cost.index import Index
from energy_cost.resolution import Resolution, to_pandas_freq
from pydantic import BaseModel, RootModel

from .models import CostWarning, CostWarningCode

logger = logging.getLogger(__name__)


class UnknownIndexError(LookupError):
    """A contract references a price index that is not registered."""


class CostCalculationError(ValueError):
    """Cost could not be calculated for the given frame and contract."""


def iter_index_names(obj: Any, _seen: set[int] | None = None) -> set[str]:
    """Collect every price index name referenced anywhere in a contract.

    This walks the Pydantic model tree generically rather than switching on formula
    kinds. ``Formula`` is a union of eight kinds that nest arbitrarily -- the shipped
    Fluvius capacity node is ``maximum -> minimum -> by_meter_type`` -- so a generic
    walk is both shorter and immune to new formula kinds being added upstream.
    """
    if _seen is None:
        _seen = set()

    if isinstance(obj, IndexAdder):
        return {obj.index}

    # Guard against shared sub-models being walked repeatedly (or cycles).
    if isinstance(obj, (BaseModel, list, dict)):
        marker = id(obj)
        if marker in _seen:
            return set()
        _seen.add(marker)

    names: set[str] = set()

    if isinstance(obj, RootModel):
        names |= iter_index_names(obj.root, _seen)
    elif isinstance(obj, BaseModel):
        for field_name in type(obj).model_fields:
            names |= iter_index_names(getattr(obj, field_name, None), _seen)
    elif isinstance(obj, dict):
        for value in obj.values():
            names |= iter_index_names(value, _seen)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            names |= iter_index_names(value, _seen)

    return names


def check_indexes(contract: Contract | ContractHistory, end: dt.datetime) -> list[CostWarning]:
    """Resolve every index the contract references, before any calculation runs.

    Raises ``UnknownIndexError`` for a name that is not registered. Warns when an index
    exposing its backing data stops before ``end``: the YAML indexes are registered with
    ``forward_fill=True``, so a period past the last data point silently reuses the last
    known value instead of failing.
    """
    warnings: list[CostWarning] = []

    for name in sorted(iter_index_names(contract)):
        try:
            index = Index.get(name)
        except KeyError as exc:
            raise UnknownIndexError(name) from exc

        frame = getattr(index, "df", None)
        if not isinstance(frame, pd.DataFrame) or "timestamp" not in frame:
            continue
        if frame.empty:
            continue

        covered_until = pd.Timestamp(frame["timestamp"].max())
        resolution = getattr(index, "resolution", None)
        if isinstance(resolution, dt.timedelta):
            covered_until = covered_until + resolution

        end_ts = pd.Timestamp(end)
        if covered_until.tzinfo is None or end_ts.tzinfo is None:
            continue
        # Report both moments in the same zone, so the message reads coherently.
        covered_until = covered_until.tz_convert(end_ts.tz)
        if covered_until < end_ts:
            message = (
                f"Price index '{name}' has data only until "
                f"{covered_until.isoformat()}, before the requested end "
                f"{end_ts.isoformat()}. Values past that point are forward-filled from "
                "the last known value."
            )
            logger.warning(message)
            warnings.append(
                CostWarning(
                    code=CostWarningCode.INDEX_DATA_INCOMPLETE,
                    message=message,
                    context={
                        "index": name,
                        "covered_until": covered_until.isoformat(),
                        "requested_end": end_ts.isoformat(),
                    },
                )
            )

    return warnings


def _is_on_boundary(moment: dt.datetime, freq: str) -> bool:
    """Whether ``moment`` sits exactly on a ``freq`` boundary."""
    try:
        return pd.Timestamp(moment).floor(freq) == pd.Timestamp(moment)
    except ValueError:
        # Non-fixed frequencies (MS, YS) cannot be floored; compare against the offset.
        offset = pd.tseries.frequencies.to_offset(freq)
        return bool(offset.is_on_offset(pd.Timestamp(moment)))


def check_period(
    contract: Contract | ContractHistory,
    start: dt.datetime,
    end: dt.datetime,
    output_resolution: Resolution,
) -> list[CostWarning]:
    """Warn about billing windows that make the absolute cost hard to interpret."""
    warnings: list[CostWarning] = []
    freq = to_pandas_freq(output_resolution)
    span = pd.Timestamp(end) - pd.Timestamp(start)

    if not (_is_on_boundary(start, freq) and _is_on_boundary(end, freq)):
        warnings.append(
            CostWarning(
                code=CostWarningCode.PARTIAL_BILLING_PERIOD,
                message=(
                    f"The period {start.isoformat()} to {end.isoformat()} does not align "
                    f"with the {isodate.duration_isoformat(output_resolution)} billing "
                    "resolution. Fixed and capacity charges are billed for the whole "
                    "snapped period, so absolute totals are inflated; the before/after "
                    "comparison is unaffected."
                ),
                context={"start": start.isoformat(), "end": end.isoformat()},
            )
        )

    period_length = (
        pd.Timestamp(start) + pd.tseries.frequencies.to_offset(freq) - pd.Timestamp(start)
    )
    if span < period_length:
        warnings.append(
            CostWarning(
                code=CostWarningCode.PERIOD_SHORTER_THAN_BILLING_PERIOD,
                message=(
                    f"The simulated period ({span}) is shorter than one billing period "
                    f"({period_length}). Fixed charges for the full period still apply."
                ),
                context={"span_days": span.days},
            )
        )

    capacity_rule = _capacity_rule(contract)
    if capacity_rule is not None and capacity_rule.window_periods:
        billing_offset = pd.tseries.frequencies.to_offset(
            to_pandas_freq(capacity_rule.billing_period)
        )
        window_end = pd.Timestamp(start) + billing_offset * capacity_rule.window_periods
        required = window_end - pd.Timestamp(start)
        if span < required:
            warnings.append(
                CostWarning(
                    code=CostWarningCode.CAPACITY_WINDOW_INCOMPLETE,
                    message=(
                        f"The capacity tariff averages over {capacity_rule.window_periods} "
                        f"billing periods, but the simulation spans only {span}. With no "
                        "prior history the early periods are not averaged down, so the "
                        "absolute capacity cost is overstated. Supply at least "
                        f"{capacity_rule.window_periods} billing periods of data for a "
                        "steady-state figure; the before/after delta remains meaningful."
                    ),
                    context={
                        "window_periods": capacity_rule.window_periods,
                        "span_days": span.days,
                    },
                )
            )

    return warnings


def _capacity_rule(contract: Contract | ContractHistory):
    """The capacity rule of a contract, or of the first version of a history."""
    if isinstance(contract, ContractHistory):
        versions = contract.root
        return versions[0].capacity_rule if versions else None
    return contract.capacity_rule
