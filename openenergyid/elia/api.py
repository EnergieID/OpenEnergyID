"""
This module contains the functions to interact with the Elia API.
"""

import datetime as dt

import aiohttp
import pandas as pd

from .const import Region

DATE_FORMAT = "%Y-%m-%d"


async def get_dataset(
    dataset: str,
    start: dt.date,
    end: dt.date,
    region: Region,
    select: set[str],
    session: aiohttp.ClientSession,
    timezone: str = "Europe/Brussels",
) -> dict:
    """
    Fetches a dataset from the Elia open data API within a specified date range and region.

    Args:
        dataset (str): The name of the dataset to fetch.
        start (dt.date): The start date for the data range.
        end (dt.date): The end date for the data range.
        region (Region): The region for which to fetch the data.
        select (set[str]): A set of fields to select in the dataset.
        session (aiohttp.ClientSession): The aiohttp session to use for making the request.
        timezone (str, optional): The timezone to use for the data. Defaults to "Europe/Brussels".

    Returns:
        dict: The dataset fetched from the API.

    Raises:
        aiohttp.ClientError: If there is an error making the request.
    """
    url = f"https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{dataset}/exports/json"

    if "datetime" not in select:
        select.add("datetime")
    select_str = ",".join(select)

    params = {
        "where": (
            f"datetime IN [date'{start.strftime(DATE_FORMAT)}'..date'{end.strftime(DATE_FORMAT)}'] "
            f"AND region='{region.value}'"
        ),
        "timezone": timezone,
        "select": select_str,
    }

    async with session.get(url, params=params) as response:
        data = await response.json()

    return data


def parse_response(
    data: dict, index: str, columns: list[str], timezone: str = "Europe/Brussels"
) -> pd.DataFrame:
    """
    Parses a response dictionary into a pandas DataFrame.

    Args:
        data (dict): The input data where each key is a column name
            and each value is a list of column values.
        index (str): The key in the data dictionary to be used as the index for the DataFrame.
        columns (list[str]): The list of column names for the DataFrame.
        timezone (str, optional): The timezone to convert the DataFrame index to.
            Defaults to "Europe/Brussels".

    Returns:
        pd.DataFrame: A pandas DataFrame with the specified columns and index,
            converted to the specified timezone.
    """
    df = pd.DataFrame(
        data,
        index=pd.to_datetime([x[index] for x in data], utc=True),
        columns=columns,
    )
    df.index = pd.DatetimeIndex(df.index)
    df = df.tz_convert(timezone)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    return df
