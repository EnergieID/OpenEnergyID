"""
PV load factors from the Elia open data API.

Until 2026-10-25 Elia published measured PV production with a `loadfactor` per region
in ods032. From 2026-10-26 the measured data lives in ods209 and the forecasts in
ods210. Those datasets have no load factor, use other region names, and publish one
row per grid connection type. The functions here hide that switch: they take the same
`Region` values as before and return a load factor in percent either way.
"""

import datetime as dt

import aiohttp
import pandas as pd

from .api import get_dataset, parse_response
from .const import (
    PV_DATASET_CUTOVER,
    PV_HISTORICAL_DATASET,
    PV_HISTORICAL_DATASET_LEGACY,
    PV_NEAR_REALTIME_DATASET,
    PV_REGION_MEMBERS,
    Region,
)

TOTAL_GRID_CONNECTION = "gridconnectiontype='Total'"

# ods209 is published with a delay of up to a day. A range that starts this many days
# before today, or earlier, must return rows.
PUBLICATION_DELAY_DAYS = 2


def _check_response(data, dataset: str) -> None:
    if not isinstance(data, list):
        raise TypeError(f"Unexpected response from Elia for {dataset}: {data}")


def _brussels_today() -> dt.date:
    return pd.Timestamp.now(tz="Europe/Brussels").date()


def _check_members(
    data: list[dict],
    members: tuple[str, ...],
    dataset: str,
    start: dt.date,
    end: dt.date,
    require_data: bool,
) -> None:
    """
    Raise when Elia's answer no longer matches the names we query.

    A renamed region, or a dropped `Total` row, does not make the query fail: it just
    matches fewer rows. Without this check that would turn into a series with gaps, or
    an empty one, which callers read as no production.
    """
    if not data:
        if require_data:
            raise ValueError(
                f"Elia {dataset} returned no rows with {TOTAL_GRID_CONNECTION} for "
                f"regions {list(members)} between {start} and {end}. "
                "Elia may have renamed the regions or the grid connection type."
            )
        return
    returned = {row.get("region") for row in data}
    missing = [member for member in members if member not in returned]
    if missing:
        raise ValueError(
            f"Elia {dataset} returned no rows for regions {missing} between {start} and "
            f"{end}, but did for {sorted(str(r) for r in returned)}. "
            "Elia may have renamed these regions."
        )


def _to_number(values: pd.Series) -> pd.Series:
    """Coerce a column to float, also when Elia sends text with a decimal comma."""
    if not pd.api.types.is_numeric_dtype(values):
        values = values.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(values, errors="coerce")


def _load_factor_from_members(
    data: list[dict],
    production_field: str,
    member_count: int,
    timezone: str,
) -> pd.Series:
    """
    Sum production and monitored capacity over the member regions per timestamp and
    return production / capacity in percent. Timestamps where a member is missing are
    dropped, so a partial sum never passes for a full one.
    """
    columns = [production_field, "monitoredcapacity"]
    df = parse_response(data=data, index="datetime", columns=columns, timezone=timezone)
    for column in columns:
        df[column] = _to_number(df[column])
    df = df.dropna()

    grouped = df.groupby(level=0)
    totals = grouped.sum()
    totals = totals[grouped.size() == member_count]
    totals = totals[totals["monitoredcapacity"] > 0]

    load_factor = totals[production_field] / totals["monitoredcapacity"] * 100
    load_factor.index = pd.DatetimeIndex(load_factor.index)
    return load_factor.rename(None)


async def _get_member_load_factors(
    dataset: str,
    production_field: str,
    start: dt.date,
    end: dt.date,
    region: Region,
    session: aiohttp.ClientSession,
    timezone: str,
    require_data: bool,
) -> pd.Series:
    members = PV_REGION_MEMBERS[region]
    data = await get_dataset(
        dataset=dataset,
        start=start,
        end=end,
        region=region,
        regions=members,
        extra_where=TOTAL_GRID_CONNECTION,
        select={production_field, "monitoredcapacity", "region"},
        session=session,
        timezone=timezone,
    )
    _check_response(data, dataset)
    _check_members(data, members, dataset, start, end, require_data)
    return _load_factor_from_members(
        data=data,
        production_field=production_field,
        member_count=len(members),
        timezone=timezone,
    )


async def _get_legacy_load_factors(
    start: dt.date,
    end: dt.date,
    region: Region,
    session: aiohttp.ClientSession,
    timezone: str,
) -> pd.Series:
    data = await get_dataset(
        dataset=PV_HISTORICAL_DATASET_LEGACY,
        start=start,
        end=end,
        region=region,
        select={"loadfactor"},
        session=session,
        timezone=timezone,
    )
    _check_response(data, PV_HISTORICAL_DATASET_LEGACY)
    df = parse_response(data=data, index="datetime", columns=["loadfactor"], timezone=timezone)
    return _to_number(df["loadfactor"]).dropna().rename(None)


async def get_pv_load_factors(
    start: dt.date,
    end: dt.date,
    region: Region,
    session: aiohttp.ClientSession,
    timezone: str = "Europe/Brussels",
) -> pd.Series:
    """
    Measured PV load factors (percent of monitored capacity) per quarter hour.

    Both `start` and `end` are inclusive days. Days before 2026-10-26 come from ods032,
    later days from ods209.

    Raises ValueError when ods209 answers without one of the regions that make up
    `region`, or without any row for a range that starts more than a day ago: both mean
    Elia changed a name this module relies on.
    """
    parts: list[pd.Series] = []
    if start < PV_DATASET_CUTOVER:
        legacy_end = min(end, PV_DATASET_CUTOVER - dt.timedelta(days=1))
        parts.append(await _get_legacy_load_factors(start, legacy_end, region, session, timezone))
    if end >= PV_DATASET_CUTOVER:
        new_start = max(start, PV_DATASET_CUTOVER)
        parts.append(
            await _get_member_load_factors(
                dataset=PV_HISTORICAL_DATASET,
                production_field="measured_upscaled",
                start=new_start,
                end=end,
                region=region,
                session=session,
                timezone=timezone,
                require_data=new_start
                <= _brussels_today() - dt.timedelta(days=PUBLICATION_DELAY_DAYS),
            )
        )

    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz=timezone))

    result = pd.concat(parts).sort_index()
    return result[~result.index.duplicated(keep="last")]


async def get_pv_load_factor_forecast(
    start: dt.date,
    end: dt.date,
    region: Region,
    session: aiohttp.ClientSession,
    timezone: str = "Europe/Brussels",
) -> pd.Series:
    """
    Most recent PV forecast as a load factor (percent of monitored capacity), from ods210.

    ods210 is a rolling window of roughly the past week to six days ahead; days outside
    it return nothing. Raises ValueError when the answer lacks one of the regions that
    make up `region`.
    """
    return await _get_member_load_factors(
        dataset=PV_NEAR_REALTIME_DATASET,
        production_field="mostrecentforecast",
        start=start,
        end=end,
        region=region,
        session=session,
        timezone=timezone,
        require_data=False,
    )
