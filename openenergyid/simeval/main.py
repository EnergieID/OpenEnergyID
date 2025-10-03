"""Module for evaluating energy simulation data."""

import pandas as pd

from .. import const


def evaluate(
    data: pd.DataFrame, return_frequencies: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Evaluate the data and return resampled results.

    Args:
        data: A pandas DataFrame containing time series data with columns:
            - electricity_delivered
            - electricity_exported
            - electricity_produced
        return_frequencies: List of pandas offset aliases for resampling frequencies.
            Defaults to ['MS'] (Month Start) if None.

    Returns:
        A dictionary with keys as frequencies and values as resampled DataFrames.
    """
    evaluator = Evaluator(data=data, return_frequencies=return_frequencies)
    return evaluator.evaluate()


class Evaluator:
    """Evaluator for basic energy system evaluation."""

    def __init__(self, data: pd.DataFrame, return_frequencies: list[str] | None = None):
        """Initialize the evaluator with data and return frequencies."""
        self.data = data
        if return_frequencies is None:
            self.return_frequencies = ["MS"]  # Month Start
        else:
            self.return_frequencies = return_frequencies

    def evaluate(self) -> dict[str, pd.DataFrame]:
        """Evaluate the data and return resampled results."""
        # Add electricy_consumed
        self.data[const.ELECTRICITY_CONSUMED] = (
            self.data[const.ELECTRICITY_DELIVERED]
            - self.data[const.ELECTRICITY_EXPORTED]
            + self.data[const.ELECTRICITY_PRODUCED]
        )

        results = {}
        for freq in self.return_frequencies:
            resampled = (
                self.data[
                    [
                        const.ELECTRICITY_DELIVERED,
                        const.ELECTRICITY_EXPORTED,
                        const.ELECTRICITY_PRODUCED,
                        const.ELECTRICITY_CONSUMED,
                    ]
                ]
                .resample(freq)
                .sum()
            )
            results[freq] = resampled
        return results
