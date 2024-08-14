"""Main module for capacity analysis."""

import datetime as dt
import typing
import pandas as pd
import pandera.typing as pdt


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
