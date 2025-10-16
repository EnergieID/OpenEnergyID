"""Module for evaluating energy simulation data."""

import typing

import numpy as np
import pandas as pd

from .. import const


def evaluate(
    data: pd.DataFrame, return_frequencies: list[str] | None = None
) -> dict[str, pd.DataFrame | pd.Series]:
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
        self.data = data.copy()
        if return_frequencies is None:
            self.return_frequencies = []
        else:
            self.return_frequencies = return_frequencies

    def evaluate(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Evaluate the data and return resampled results."""
        if const.ELECTRICITY_DELIVERED not in self.data.columns:
            self.data[const.ELECTRICITY_DELIVERED] = float("NaN")
        if const.ELECTRICITY_EXPORTED not in self.data.columns:
            self.data[const.ELECTRICITY_EXPORTED] = float("NaN")
        if const.ELECTRICITY_PRODUCED not in self.data.columns:
            self.data[const.ELECTRICITY_PRODUCED] = float("NaN")
        if const.PRICE_ELECTRICITY_DELIVERED not in self.data.columns:
            self.data[const.PRICE_ELECTRICITY_DELIVERED] = float("NaN")
        if const.PRICE_ELECTRICITY_EXPORTED not in self.data.columns:
            self.data[const.PRICE_ELECTRICITY_EXPORTED] = float("NaN")

        # Add electricy_consumed
        self.data[const.ELECTRICITY_CONSUMED] = (
            self.data[const.ELECTRICITY_DELIVERED]
            - self.data[const.ELECTRICITY_EXPORTED].fillna(0.0)
            + self.data[const.ELECTRICITY_PRODUCED].fillna(0.0)
        ).clip(lower=0)

        # Add electricy_self_consumed
        self.data[const.ELECTRICITY_SELF_CONSUMED] = (
            self.data[const.ELECTRICITY_PRODUCED] - self.data[const.ELECTRICITY_EXPORTED]
        ).clip(lower=0)

        # Add costs
        self.data[const.COST_ELECTRICITY_DELIVERED] = (
            self.data[const.ELECTRICITY_DELIVERED] * self.data[const.PRICE_ELECTRICITY_DELIVERED]
        )
        self.data[const.EARNINGS_ELECTRICITY_EXPORTED] = (
            self.data[const.ELECTRICITY_EXPORTED] * self.data[const.PRICE_ELECTRICITY_EXPORTED]
        )
        self.data[const.COST_ELECTRICITY_NET] = (
            self.data[const.COST_ELECTRICITY_DELIVERED]
            - self.data[const.EARNINGS_ELECTRICITY_EXPORTED]
        )

        # Calculate sums
        results: dict[str, pd.DataFrame | pd.Series] = {}
        results["total"] = (
            self.data[
                [
                    const.ELECTRICITY_DELIVERED,
                    const.ELECTRICITY_EXPORTED,
                    const.ELECTRICITY_PRODUCED,
                    const.ELECTRICITY_CONSUMED,
                    const.ELECTRICITY_SELF_CONSUMED,
                    const.COST_ELECTRICITY_DELIVERED,
                    const.EARNINGS_ELECTRICITY_EXPORTED,
                    const.COST_ELECTRICITY_NET,
                ]
            ]
            .dropna(axis=1, how="all")
            .sum()
        )

        for freq in self.return_frequencies:
            resampled = (
                self.data[
                    [
                        const.ELECTRICITY_DELIVERED,
                        const.ELECTRICITY_EXPORTED,
                        const.ELECTRICITY_PRODUCED,
                        const.ELECTRICITY_CONSUMED,
                        const.ELECTRICITY_SELF_CONSUMED,
                        const.COST_ELECTRICITY_DELIVERED,
                        const.EARNINGS_ELECTRICITY_EXPORTED,
                        const.COST_ELECTRICITY_NET,
                    ]
                ]
                .dropna(axis=1, how="all")
                .resample(freq)
                .sum()
            )
            results[freq] = resampled

        # Add ratios
        for _, frame in results.items():
            if const.ELECTRICITY_SELF_CONSUMED in frame:
                frame[const.RATIO_SELF_CONSUMPTION] = np.divide(
                    frame[const.ELECTRICITY_SELF_CONSUMED], frame[const.ELECTRICITY_PRODUCED]
                )
                frame[const.RATIO_SELF_SUFFICIENCY] = np.divide(
                    frame[const.ELECTRICITY_SELF_CONSUMED],
                    frame[const.ELECTRICITY_CONSUMED],
                )
        return results


def compare_results(
    res_1: dict[str, pd.DataFrame | pd.Series], res_2: dict[str, pd.DataFrame | pd.Series]
) -> dict[str, dict[str, pd.Series | pd.DataFrame]]:
    """Compare two evaluation results and return the differences."""
    results = {}
    for key in res_1.keys():
        if key in res_2:
            df_1 = res_1[key]
            df_2 = res_2[key]
            if isinstance(df_1, pd.Series) and isinstance(df_2, pd.Series):
                df_1 = df_1.to_frame().T
                df_2 = df_2.to_frame().T
            df_1, df_2 = typing.cast(pd.DataFrame, df_1), typing.cast(pd.DataFrame, df_2)
            diff = df_2 - df_1
            results[key] = {}
            results[key]["diff"] = diff.dropna(how="all", axis=1).squeeze(axis=0)
            results[key]["ratio_diff"] = (diff / df_1).dropna(how="all", axis=1).squeeze(axis=0)
    return results
