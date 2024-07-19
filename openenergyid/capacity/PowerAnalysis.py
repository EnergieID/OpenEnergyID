import datetime

from openenergyid.capacity.models import CapacityInput


class PowerPeakAnalysis:
    """analysis
    This class is used to analyze the power peaks of a given time series.
    The analysis is based on the following parameters:
    - min_peak_value: The minimum value of a peak to be considered a peak.
    - num_peaks: The number of peaks to be returned.
    - from_date: The start date of the analysis.
    - to_date: The end date of the analysis.
    - x_padding: The number of days to be added to the start and end date to
      ensure that the peaks are not cut off.
    - capacity_input: The input data for the analysis.
    """

    def __init__(
        self,
        min_peak_value: float,
        num_peaks: int,
        from_date: datetime,
        to_date: datetime,
        x_padding: int = 2,
        capacity_input=CapacityInput,
    ):
        self.data = input.get_series()
        self.timezone = capacity_input.timezone
        self.min_peak_value = min_peak_value
        self.num_peaks = num_peaks
        self.from_date = from_date
        self.to_date = to_date
        self.x_padding = x_padding
