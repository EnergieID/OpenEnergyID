"""Mapping simulation frames onto ``energy_cost`` meters."""

import datetime as dt

import pandas as pd
from energy_cost import Meter, MeterType, PowerDirection, TimeseriesFrame

from .. import const
from .models import CostWarning, CostWarningCode

# energy_cost works in MWh and EUR/MWh; simulation frames are in kWh.
KWH_PER_MWH = 1000.0


def detect_frame_resolution(index: pd.DatetimeIndex) -> dt.timedelta:
    """Infer a single resolution from a DatetimeIndex.

    Prefers ``index.freq``, then ``pd.infer_freq``, then the modal positive difference.
    The modal fallback matters: ``infer_freq`` returns ``None`` as soon as there is a
    gap, and a tz-aware index keeps constant 15-minute deltas across a DST transition,
    so the mode survives both.
    """
    if len(index) < 2:
        raise ValueError("Cannot detect a resolution from fewer than 2 timestamps.")
    if not index.is_monotonic_increasing:
        raise ValueError("Cannot detect a resolution from a non-monotonic index.")

    if index.freq is not None:
        return pd.to_timedelta(index.freq)

    inferred = pd.infer_freq(index)
    if inferred is not None:
        offset = pd.tseries.frequencies.to_offset(inferred)
        try:
            return pd.to_timedelta(offset)
        except (ValueError, TypeError):
            # Non-fixed offsets (month/year starts) have no timedelta; fall through.
            pass

    diffs = index.to_series().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("Cannot detect a resolution: no positive timestamp differences.")
    return diffs.mode().iloc[0].to_pytimedelta()


def _to_timeseries_frame(series: pd.Series, resolution: dt.timedelta) -> TimeseriesFrame:
    """Build an ``energy_cost`` TimeseriesFrame from a kWh series.

    Timestamps are normalised to UTC. ``energy_cost`` merges meter values against
    index and bin frames it builds itself in UTC, and ``pandas.merge_asof`` refuses
    to join keys whose timezones differ, so handing it a localised index fails.
    ``Contract.apply`` converts to the contract timezone afterwards anyway.
    """
    return TimeseriesFrame(
        {
            "timestamp": pd.DatetimeIndex(series.index).tz_convert("UTC"),
            "value": series.to_numpy() / KWH_PER_MWH,
        },
        resolution=resolution,
    )


def _prepare_column(
    data: pd.DataFrame, column: str, warnings: list[CostWarning]
) -> pd.Series | None:
    """Return a NaN-free copy of ``column``, or None when it carries no data."""
    if column not in data.columns:
        return None

    series = data[column]
    if series.isna().all():
        return None

    nan_count = int(series.isna().sum())
    if nan_count:
        warnings.append(
            CostWarning(
                code=CostWarningCode.ENERGY_DATA_GAPS_FILLED,
                message=(
                    f"{nan_count} missing values in '{column}' were treated as 0 kWh. "
                    "energy_cost masks any billing period containing a NaN, so gaps "
                    "would otherwise void the whole period's cost."
                ),
                context={"column": column, "filled": nan_count},
            )
        )
        series = series.fillna(0.0)

    return series


def frame_to_meters(
    data: pd.DataFrame,
    meter_type: MeterType = MeterType.SINGLE_RATE,
    resolution: dt.timedelta | None = None,
) -> tuple[Meter, Meter | None, list[CostWarning]]:
    """Map a simulation frame to ``energy_cost`` consumption/injection meters.

    ``electricity_delivered`` becomes the consumption meter and ``electricity_exported``
    the injection meter. Values are converted from kWh to MWh.

    Injection volumes are passed through **positive**; a feed-in payment is expressed as
    a negative rate in the tariff, because ``energy_cost`` sums every cost group into the
    bill total and applies no sign logic of its own.

    No capacity frame is supplied: ``energy_cost.CapacityRule`` derives capacity from the
    measurements itself, which is both correct for sub-hourly data and the method the
    Flemish capaciteitstarief prescribes.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Cost calculation requires a DatetimeIndex.")
    if data.index.tz is None:
        raise ValueError("Cost calculation requires a timezone-aware index; got a naive one.")

    warnings: list[CostWarning] = []
    resolution = resolution if resolution is not None else detect_frame_resolution(data.index)

    delivered = _prepare_column(data, const.ELECTRICITY_DELIVERED, warnings)
    if delivered is None:
        raise ValueError(
            f"Cost calculation requires a '{const.ELECTRICITY_DELIVERED}' column with data."
        )

    consumption = Meter(
        direction=PowerDirection.CONSUMPTION,
        type=meter_type,
        measurements=_to_timeseries_frame(delivered, resolution),
    )

    exported = _prepare_column(data, const.ELECTRICITY_EXPORTED, warnings)
    injection: Meter | None = None
    if exported is None:
        warnings.append(
            CostWarning(
                code=CostWarningCode.NO_INJECTION_DATA,
                message=(
                    f"No '{const.ELECTRICITY_EXPORTED}' data; injection costs and "
                    "feed-in revenue are excluded."
                ),
                context={"column": const.ELECTRICITY_EXPORTED},
            )
        )
    else:
        injection = Meter(
            direction=PowerDirection.INJECTION,
            type=meter_type,
            measurements=_to_timeseries_frame(exported, resolution),
        )

    return consumption, injection, warnings
