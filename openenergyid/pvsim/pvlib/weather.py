"""
Weather data utilities for PV simulation.

Provides functions to retrieve and process weather data for PV simulations,
including timezone-aware handling and leap year adjustments.
"""

import datetime as dt
import typing

import pandas as pd
import pvlib


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


def get_weather(
    latitude: float, longitude: float, start: dt.date, end: dt.date, tz: str
) -> pd.DataFrame:
    """
    Retrieve and process weather data for the specified location and date range.

    Downloads a "normal year" TMY dataset from PVGIS, aligns it to the requested
    date range and timezone, and handles leap year adjustments.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
        start: Start date (inclusive).
        end: End date (exclusive).
        tz: Timezone string.

    Returns:
        A pandas DataFrame indexed by timestamp with weather data.
    """
    # Get a "normal year" from pvgis, it will be indexed in 1990
    utc_offset = get_utc_offset_on_1_jan(tz)
    weather = pvlib.iotools.get_pvgis_tmy(
        latitude=latitude, longitude=longitude, roll_utc_offset=utc_offset
    )[0]
    weather = typing.cast(pd.DataFrame, weather)
    weather = weather.tz_convert(tz)
    weather.index.name = None

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
