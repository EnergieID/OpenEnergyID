"""Measured month peaks as capacity history for the contract cost.

The Flemish capacity tariff bills the average of the last twelve monthly peaks, so a
costed year only has a full window if the eleven months *before* it are known.
``energy_cost`` accepts that history on the consumption ``Meter`` as a ``capacity``
series (MW, one value per month). By default ``openenergyid.cost`` lets it derive the
capacity from the costed frame alone; this module builds the series when the caller
supplies measured month peaks:

* the months before the first billed month are the provided history;
* the months from there on are recalculated from the frame being costed -- ex-ante data
  for the current situation, simulated data for the simulated one. Provided peaks inside
  the period are never used, which keeps the before/after comparison like-for-like.

When the history lacks any of the ``window_periods - 1`` months directly before the first
billed month there is nothing to build, and costing proceeds exactly as without peaks.
"""

import datetime as dt

import pandas as pd
from energy_cost import CapacityRule, Contract, ContractHistory, Meter, TimeseriesFrame
from energy_cost.resolution import to_pandas_freq

from .models import MonthPeaks

W_PER_MW = 1_000_000.0


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize().replace(day=1)


def first_version(contract: Contract | ContractHistory) -> Contract | None:
    """The contract itself, or the first version of a history (None when empty)."""
    if isinstance(contract, ContractHistory):
        return contract.root[0] if contract.root else None
    return contract


def first_billed_month(first_billed: dt.datetime, timezone: str) -> pd.Timestamp:
    """Start of the first billed month, in the contract timezone.

    A naive datetime (a naive ``ContractCostSettings.start``) is read in that timezone.
    """
    ts = pd.Timestamp(first_billed)
    ts = ts.tz_localize(timezone) if ts.tzinfo is None else ts.tz_convert(timezone)
    return _month_start(ts)


def build_capacity(
    month_peaks: MonthPeaks,
    consumption: Meter,
    rule: CapacityRule,
    cutoff: pd.Timestamp,
    timezone: str,
) -> pd.DataFrame | None:
    """The monthly capacity series (``timestamp``, ``value`` in MW) for one frame.

    Months before ``cutoff`` (the start of the first billed month) are the provided
    history; later months are the peaks of ``consumption``. Returns ``None`` when the
    history lacks any of the ``window_periods - 1`` months directly before ``cutoff``, or
    when the rule has no monthly rolling window to fill.
    """
    if rule.window_periods is None or to_pandas_freq(rule.billing_period) != "1MS":
        return None

    cutoff = _month_start(pd.Timestamp(cutoff).tz_convert(timezone))
    history = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(v.timestamp).tz_convert(timezone) for v in month_peaks.values
            ],
            "value": [v.value / W_PER_MW for v in month_peaks.values],
        }
    )
    history = history[history["timestamp"] < cutoff].reset_index(drop=True)

    needed = {cutoff - pd.DateOffset(months=n) for n in range(1, rule.window_periods)}
    have = {_month_start(ts) for ts in history["timestamp"]}
    if not needed <= have:
        return None

    # Same derivation as energy_cost, minus the rolling window (applied over the whole
    # series later, once the history is attached).
    peaks_rule = CapacityRule(
        measurement_period=rule.measurement_period, billing_period=rule.billing_period
    )
    own = peaks_rule.apply(consumption.align_to_timezone(cutoff.tzinfo)).capacity
    own = pd.DataFrame(own)[["timestamp", "value"]]
    own = own[own["timestamp"] >= cutoff]

    combined = pd.concat([history, own], ignore_index=True).sort_values("timestamp")
    return combined.reset_index(drop=True)


def with_capacity_history(
    consumption: Meter,
    month_peaks: MonthPeaks,
    contract: Contract | ContractHistory,
    first_billed: dt.datetime,
) -> Meter | None:
    """``consumption`` carrying the history-backed capacity series, or None if unusable."""
    version = first_version(contract)
    if version is None or version.capacity_rule is None:
        return None
    rule = version.capacity_rule
    timezone = str(version.timezone)

    series = build_capacity(
        month_peaks, consumption, rule, first_billed_month(first_billed, timezone), timezone
    )
    if series is None:
        return None
    return consumption.model_copy(
        update={"capacity": TimeseriesFrame(series, resolution=rule.billing_period)}
    )
