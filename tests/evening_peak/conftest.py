"""Shared builders for the evening peak avoidance tests."""

import datetime as dt
from collections.abc import Callable
from zoneinfo import ZoneInfo

import polars as pl

from openenergyid.models import TimeSeries

TIMEZONE = "Europe/Amsterdam"


def day(year: int, month: int, day_of_month: int, hour: int = 0, minute: int = 0) -> dt.datetime:
    """A naive local wall-clock moment, to be localized by :func:`local_quarters`."""
    return dt.datetime(year, month, day_of_month, hour, minute)  # noqa: DTZ001


def local_quarters(
    start_local: dt.datetime, count: int, timezone: str = TIMEZONE
) -> list[dt.datetime]:
    """Contiguous quarter-hourly local timestamps, correct across DST transitions.

    Walks in real time from the first instant and converts back, so a 25-hour local day
    yields 100 timestamps and a 23-hour day yields 92.
    """
    zone = ZoneInfo(timezone)
    start_utc = start_local.replace(tzinfo=zone).astimezone(dt.UTC)
    return [(start_utc + dt.timedelta(minutes=15 * i)).astimezone(zone) for i in range(count)]


def frame(
    index: list[dt.datetime],
    value_of: Callable[[dt.datetime], float],
    *,
    name: str = "gross_offtake",
    timezone: str = TIMEZONE,
) -> pl.LazyFrame:
    """Build a gross series frame through the production TimeSeries path."""
    series = TimeSeries(name=name, index=index, data=[value_of(t) for t in index])
    return series.to_polars(timezone=timezone)


def minute_of_day(timestamp: dt.datetime) -> int:
    """Minutes since local midnight."""
    return timestamp.hour * 60 + timestamp.minute


def in_evening(timestamp: dt.datetime) -> bool:
    """Whether a local timestamp falls in the default 16:00-21:00 window."""
    return 16 * 60 <= minute_of_day(timestamp) < 21 * 60


#: A plain weekday profile: 0.05 kWh/quarter baseload, 0.5 kWh/quarter in the evening.
def flat_evening_profile(timestamp: dt.datetime) -> float:
    """Baseload with a constant evening block."""
    return 0.5 if in_evening(timestamp) else 0.05
