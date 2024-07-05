from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
from openenergyid.capacity.models import CapacityInput


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
        data: CapacityInput,
        threshold: float = 2.5,
        window: str = "MS",  # Default to month start
        x_padding: int = 4,
    ):
        """
        Constructs all the necessary attributes for the CapacityAnalysis object.

        Parameters:
            data (CapacityInput): The input data for capacity analysis.
            threshold (float): The value above which a peak is considered significant. Defaults to 2.5.
            window (str): The window size for grouping data before finding peaks. Defaults to "MS" (month start).
            x_padding (int): The padding to apply on the x-axis for visualization purposes. Defaults to 4.
        """

        self.data = data
        self.threshold = threshold
        self.window = window
        self.x_padding = x_padding

    def find_peaks(self) -> List[Tuple[datetime, float]]:
        """
        Identifies peaks in the data based on the specified threshold and window.

        Returns:
            List[Tuple[datetime, float]]: A list of tuples where each tuple contains the timestamp (datetime) of the peak and its value (float).
        """
        series = self.data.get_series()
        # Group by the specified window (default is month start)
        grouped = series.groupby(pd.Grouper(freq=self.window))
        # Find the index (timestamp) of the maximum value in each group
        peak_indices = grouped.idxmax()
        # Get the corresponding peak values
        peaks = series.loc[peak_indices][series > self.threshold]
        return [(index, value) for index, value in peaks.items()]

    def find_peaks_with_surroundings(self, num_peaks: int = 10) -> List[Dict]:
        """
        Finds peaks along with their surrounding data points.

        Parameters:
            num_peaks (int): The number of peaks to find. Defaults to 10.

        Returns:
            List[Dict]: A list of dictionaries, each representing a peak and its surroundings.
        """
        series = self.data.get_series()
        peaks = []

        for i in range(len(series) - 1):
            if (
                series.iloc[i] > series.iloc[i - 1]
                and series.iloc[i] > series.iloc[i + 1]
            ):
                peaks.append((series.index[i], series.iloc[i]))

        peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = peaks[:num_peaks]

        result = []
        for peak_time, peak_value in top_peaks:
            start_time = peak_time - timedelta(minutes=15 * self.x_padding)
            end_time = peak_time + timedelta(minutes=15 * (self.x_padding + 1))
            surrounding_data = series[start_time:end_time]

            result.append(
                {
                    "peak_time": peak_time.isoformat(),
                    "peak_value": float(peak_value),
                    "surrounding_data": [
                        {"timestamp": ts.isoformat(), "value": float(val)}
                        for ts, val in surrounding_data.items()
                    ],
                }
            )

        return result

    def run_analysis(self):
        peaks = self.find_peaks()
        return {
            "peak_moments": [
                {"peak_time": peak[0].isoformat(), "peak_value": float(peak[1])}
                for peak in peaks
            ]
        }


# Usage example
if __name__ == "__main__":
    # Load data from JSON file
    with open("data/capacity/electricity_delivered_sample.json", "r") as f:
        input_data = CapacityInput.model_validate_json(f.read())

    # Create CapacityAnalysis instance
    analyzer = CapacityAnalysis(
        data=input_data,
        num_peaks=24,  # Get 24 peaks as per your requirement
        threshold=2.5,
        window="MS",  # Month Start
    )

    results = analyzer.run_analysis()

    # Print results
    print("Peak Moments:")
    for peak in results["peak_moments"]:
        print(f"Time: {peak['peak_time']}, Value: {peak['peak_value']}")
