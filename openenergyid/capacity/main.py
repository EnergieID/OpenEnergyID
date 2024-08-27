"""Main module for capacity analysis."""

import datetime as dt
import json
import typing
import pandas as pd
import pandera.typing as pdt
import polars as pl


def find_peaks(data: pl.LazyFrame, threshold: float = 0, amount: int = 10) -> pl.LazyFrame:
    """
    Identify the top peaks in the dataset.

    Parameters:
    -----------
    data : pl.LazyFrame
        The input data containing 'value' column.
    threshold : float, optional
        Minimum value to consider as a peak. Default is 0.
    amount : int, optional
        Number of peaks to return. Default is 10.

    Returns:
    --------
    pl.LazyFrame
        A LazyFrame containing the top peaks sorted by 'value' in descending order.
    """
    peaks = data.filter(pl.col("value") > threshold).sort(by="value", descending=True).head(amount)
    return peaks


def find_temporal_relative_peaks(
    df: pl.LazyFrame, amount: int = 10, threshold: float = 0, x_padding: int = 4
) -> list[tuple[dt.datetime, float]]:
    """
    Find temporal relative peaks in a dataset.

    This function identifies the top `amount` of peaks in the dataset that are
    separated by at least a specified time window. The peaks are determined based
    on the `value` column, and only values above a certain `threshold` are considered.

    Parameters:
    -----------
    df : pl.LazyFrame
        A Polars LazyFrame containing the data with at least 'timestamp' and 'value' columns.
    amount : int, optional
        The number of peaks to find. Default is 10.
    threshold : float, optional
        The minimum value to consider as a peak. Default is 0.
    x_padding : int, optional
        The number of 15-minute intervals to pad on either side of a peak to define
        the window size. Default is 4.

    Returns:
    --------
    typing.List[typing.Tuple[dt.datetime, float]]
        A list of tuples, each containing the timestamp and value of a peak.

    Example:
    --------
    >>> df = pl.LazyFrame({"timestamp": [...], "value": [...]})
    >>> peaks = find_temporal_relative_peaks(df, amount=5, threshold=10, x_padding=2)
    >>> print(peaks)
    [(datetime.datetime(...), ...), ...]

    Notes:
    ------
    - The function first filters the data to include only values above the threshold.
    - It then sorts the data in descending order based on the 'value' column.
    - The top `amount * 2` peaks are selected to ensure enough candidates.
    - For each peak, it checks if it is sufficiently separated from previously
      selected peaks based on the window size.
    - The function stops once the desired number of peaks (`amount`) is found.
    """
    window_size = dt.timedelta(minutes=15 * (2 * x_padding + 1))

    peaks = (
        df.filter(pl.col("value") > threshold)
        .sort("value", descending=True)
        .limit(amount * 2)
        .with_columns(
            [
                pl.col("timestamp").dt.offset_by(f"-{15 * x_padding}m").alias("start_time"),
                pl.col("timestamp").dt.offset_by(f"{15 * x_padding}m").alias("end_time"),
            ]
        )
        .collect()
    )

    result = []
    for peak in peaks.iter_rows(named=True):
        if not any(abs(peak["timestamp"] - prev_peak[0]) < window_size for prev_peak in result):
            result.append((peak["timestamp"], peak["value"]))
            if len(result) == amount:
                break

    return result


def read_in_json_to_polars(path: str) -> pl.LazyFrame:
    """
    Read JSON data from a file and convert it into a Polars LazyFrame.

    Args:
        path (str): The path to the JSON file.

    Returns:
        pl.LazyFrame: A Polars LazyFrame containing the data from the JSON file.

    Raises:
        ValueError: If there is an error reading the data into a LazyFrame.
    """
    try:
        with open(path) as f:
            data = json.load(f)
            timezone_info = data["timeZone"]
        lf = pl.LazyFrame({"timestamp": data["series"]["index"], "value": data["series"]["data"]})
        lf = lf.with_columns(
            pl.col("timestamp").cast(pl.Datetime).dt.convert_time_zone(timezone_info)
        )
        return lf
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise ValueError("Error reading data into LazyFrame.") from e


class CapacityAnalysis:
    """
    A class for performing capacity analysis on a given dataset.

    Attributes:
        data (CapacityInput): The input data for capacity analysis.
        threshold (float): The value above which a peak is considered significant.
        window (str): The window size for grouping data before finding peaks. Defaults to "MS" (month start).
        x_padding (int): The padding to apply on the x-axis for visualization purposes.

    Methods:
        find_peaks(): Identifies peaks in the data based on the specified threshold and window.
        find_peaks_with_surroundings(num_peaks=10): Finds peaks along with their surrounding data points.
    """

    def __init__(
        self,
        data: pdt.Series,
        threshold: float = 2.5,
        window: str = "MS",  # Default to month start
        x_padding: int = 4,
    ):
        """
        Constructs all the necessary attributes for the CapacityAnalysis object.

        Parameters:
            data (CapacityInput): Localized Pandas Series containing power measurements.
            threshold (float): The value above which a peak is considered significant. Defaults to 2.5.
            window (str): The window size for grouping data before finding peaks. Defaults to "MS" (month start).
            x_padding (int): The padding to apply on the x-axis for visualization purposes. Defaults to 4.
        """

        self.data = data
        self.threshold = threshold
        self.window = window
        self.x_padding = x_padding

    def find_peaks(self) -> pd.Series:
        """
        Identifies peaks in the data based on the specified threshold and window.

        Returns:
            pd.Series: A Pandas Series containing the peaks
        """
        # Group by the specified window (default is month start)
        grouped = self.data.groupby(pd.Grouper(freq=self.window))

        # Find the index (timestamp) of the maximum value in each group
        peak_indices = grouped.idxmax()

        # Get the corresponding peak values
        peaks = self.data.loc[peak_indices][self.data > self.threshold]
        return peaks

    def find_peaks_with_surroundings(
        self, num_peaks: int = 10
    ) -> list[tuple[dt.datetime, float, pd.Series]]:
        """
        Finds peaks along with their surrounding data points.

        Parameters:
            num_peaks (int): The number of peaks to find. Defaults to 10.

        Returns:
            List[tuple[dt.datetime,float,pd.Series]]: A list of tuples containing peak time, peak value, and surrounding data.
        """
        peaks = self.data.nlargest(num_peaks * 2)
        peaks = peaks[peaks > self.threshold]
        if peaks.empty:
            return []

        result = []
        window_size = dt.timedelta(minutes=15 * (2 * self.x_padding + 1))

        for peak_time, peak_value in peaks.items():
            peak_time = typing.cast(pd.Timestamp, peak_time)

            if any(abs(peak_time - prev_peak[0]) < window_size for prev_peak in result):
                continue

            start_time = peak_time - dt.timedelta(minutes=15 * self.x_padding)
            end_time = peak_time + dt.timedelta(minutes=15 * (self.x_padding + 1))
            surrounding_data = self.data[start_time:end_time]

            result.append(
                [
                    peak_time,
                    peak_value,
                    surrounding_data,
                ]
            )
            if len(result) == num_peaks:
                break
        return result
