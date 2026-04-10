"""
Weather data utilities for PV simulation.

Provides functions to retrieve and process weather data for PV simulations,
including timezone-aware handling and leap year adjustments.
"""

import asyncio
import datetime as dt
import threading
import time
import typing
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd
import pvlib
import requests

WeatherCacheKey = tuple[float, float, str]
_WEATHER_CACHE_MAX_SIZE = 256


@dataclass
class _InFlightWeatherRequest:
    condition: threading.Condition
    ready: bool = False
    value: pd.DataFrame | None = None
    error: Exception | None = None


class _NormalYearWeatherCache:
    """Thread-safe LRU cache with single-flight loading per weather key."""

    def __init__(self, max_size: int = _WEATHER_CACHE_MAX_SIZE) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[WeatherCacheKey, pd.DataFrame] = OrderedDict()
        self._in_flight: dict[WeatherCacheKey, _InFlightWeatherRequest] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        """Remove all completed cache entries."""
        with self._lock:
            self._cache.clear()

    def get_or_load(
        self, key: WeatherCacheKey, loader: typing.Callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        creator = False

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

            entry = self._in_flight.get(key)
            if entry is None:
                entry = _InFlightWeatherRequest(condition=threading.Condition(self._lock))
                self._in_flight[key] = entry
                creator = True
            else:
                while not entry.ready:
                    entry.condition.wait()

                if entry.error is not None:
                    raise entry.error
                if entry.value is None:
                    raise RuntimeError("Weather cache load completed without a value.")
                return entry.value

        if not creator:
            raise RuntimeError("Unreachable cache loading state.")

        try:
            value = loader()
        except Exception as exc:
            with self._lock:
                entry.error = exc
                entry.ready = True
                self._in_flight.pop(key, None)
                entry.condition.notify_all()
            raise

        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

            entry.value = value
            entry.ready = True
            self._in_flight.pop(key, None)
            entry.condition.notify_all()

        return value


_NORMAL_YEAR_WEATHER_CACHE = _NormalYearWeatherCache()


def get_utc_offset_on_1_jan(timezone: str) -> int:
    """
    Get the UTC offset in hours for the given timezone on January 1st.

    Args:
        timezone: The timezone string (e.g., "Europe/Amsterdam").

    Returns:
        The UTC offset in hours as an integer.

    Raises:
        ValueError: If the UTC offset cannot be determined.
    """
    jan_first = pd.Timestamp("2020-01-01 00:00:00", tz=timezone)
    utc_offset = jan_first.utcoffset()
    if utc_offset is None:
        raise ValueError(f"Could not determine UTC offset for timezone {timezone}")
    return int(utc_offset.total_seconds() / 3600)


def clear_weather_cache() -> None:
    """Clear the process-local PVGIS normal-year weather cache."""
    _NORMAL_YEAR_WEATHER_CACHE.clear()


def _download_normal_year_weather_sync(
    latitude: float,
    longitude: float,
    tz: str,
    timeout: int,
) -> pd.DataFrame:
    """
    Retrieve and process weather data for the specified location and date range.

    Downloads a "normal year" TMY dataset from PVGIS, aligns it to the requested
    date range and timezone, and handles leap year adjustments.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
        tz: Timezone string.
        timeout: PVGIS request timeout in seconds.

    Returns:
        A pandas DataFrame indexed by timestamp with normal-year weather data.
    """
    # Get a "normal year" from pvgis, it will be indexed in 1990
    utc_offset = get_utc_offset_on_1_jan(tz)
    weather = pvlib.iotools.get_pvgis_tmy(
        latitude=latitude,
        longitude=longitude,
        timeout=timeout,
        roll_utc_offset=utc_offset,
    )[0]
    weather = typing.cast(pd.DataFrame, weather)
    weather = weather.tz_convert(tz)
    weather.index.name = None
    return weather


def _materialize_weather_for_range_sync(
    normal_year_weather: pd.DataFrame, start: dt.date, end: dt.date, tz: str
) -> pd.DataFrame:
    """
    Align cached normal-year weather to the requested date range.

    Args:
        normal_year_weather: Cached normal-year weather for the site.
        start: Start date (inclusive).
        end: End date (exclusive).
        tz: Timezone string.

    Returns:
        A pandas DataFrame indexed by timestamp with weather data.
    """
    weather = normal_year_weather.copy()

    # Check if 29 februari is included in the weather data
    weather_index = typing.cast(pd.DatetimeIndex, weather.index)
    leap_included = "02-29" in weather_index.strftime("%m-%d").unique()

    # Construct our desired index
    new_index = pd.date_range(start, end, freq="15min", tz=tz)
    temp_df = pd.DataFrame(index=new_index)
    # Add a key that doesn't contain year
    temp_df["tkey"] = new_index.tz_convert("UTC").strftime("%m-%dT%H:%M:%S%z")
    temp_df["timestamp"] = new_index
    if not leap_included:
        # Replace all tkey's starting with "02-29" with "02-28"
        temp_df.loc[temp_df.tkey.str.startswith("02-29"), "tkey"] = temp_df.loc[
            temp_df.tkey.str.startswith("02-29"), "tkey"
        ].str.replace("02-29", "02-28")

    # Add the key to the weather frame and join
    weather["tkey"] = weather_index.tz_convert("UTC").strftime("%m-%dT%H:%M:%S%z")
    df = (
        pd.merge(temp_df, weather, on="tkey", how="left")
        .set_index("timestamp")
        .drop(columns=["tkey"])
    )

    # Interpolate and drop last value
    df = df.interpolate(method="time")
    df = df.iloc[:-1]

    return df


def _load_normal_year_weather_with_retry_sync(
    latitude: float,
    longitude: float,
    tz: str,
    timeout: int,
    retry_count: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """Download normal-year weather with retry logic for transient failures."""
    transient_exceptions = (requests.Timeout, requests.ConnectionError)

    for attempt in range(retry_count + 1):
        try:
            return _download_normal_year_weather_sync(
                latitude=latitude,
                longitude=longitude,
                tz=tz,
                timeout=timeout,
            )
        except transient_exceptions:
            if attempt == retry_count:
                raise

            delay = retry_backoff_seconds * (2**attempt)
            time.sleep(delay)

    raise RuntimeError("Unreachable retry loop state")


def _get_cached_normal_year_weather_sync(
    latitude: float,
    longitude: float,
    tz: str,
    timeout: int,
    retry_count: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """Load normal-year weather from the shared cache or fetch on cache miss."""
    key: WeatherCacheKey = (latitude, longitude, tz)
    return _NORMAL_YEAR_WEATHER_CACHE.get_or_load(
        key,
        lambda: _load_normal_year_weather_with_retry_sync(
            latitude=latitude,
            longitude=longitude,
            tz=tz,
            timeout=timeout,
            retry_count=retry_count,
            retry_backoff_seconds=retry_backoff_seconds,
        ),
    )


def _get_weather_sync(
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
    tz: str,
    timeout: int,
    retry_count: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    """Load request-specific weather using cached normal-year weather when available."""
    normal_year_weather = _get_cached_normal_year_weather_sync(
        latitude=latitude,
        longitude=longitude,
        tz=tz,
        timeout=timeout,
        retry_count=retry_count,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return _materialize_weather_for_range_sync(
        normal_year_weather=normal_year_weather,
        start=start,
        end=end,
        tz=tz,
    )


async def get_weather(
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
    tz: str,
    timeout: int = 30,
    retry_count: int = 2,
    retry_backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Retrieve weather data asynchronously using shared cached normal-year weather.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
        start: Start date (inclusive).
        end: End date (exclusive).
        tz: Timezone string.
        timeout: PVGIS request timeout in seconds per attempt.
        retry_count: Number of retries after the initial failed attempt.
        retry_backoff_seconds: Base delay in seconds for exponential retry backoff.

    Returns:
        A pandas DataFrame indexed by timestamp with weather data.
    """
    return await asyncio.to_thread(
        _get_weather_sync,
        latitude,
        longitude,
        start,
        end,
        tz,
        timeout,
        retry_count,
        retry_backoff_seconds,
    )
