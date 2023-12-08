"""Miscelaneous helper functions for the MVLR app."""

import pandas as pd

from openenergyid.enums import Granularity

pandas_granularity_map = {Granularity.P7D: "W-MON", Granularity.P1M: "MS"}


def resample_input_data(
    data: pd.DataFrame,
    granularity: Granularity,
    aggregation_methods: dict = None,
) -> pd.DataFrame:
    """Resample input data to the given granularity.

    By default, the data is summed up for each column.
    Provide a dictionary of aggregation methods to override this behaviour.
    """
    if granularity not in pandas_granularity_map:
        raise NotImplementedError("Granularity not implemented.")
    aggregation_methods = aggregation_methods.copy() if aggregation_methods else {}

    for column in data.columns:
        if column not in aggregation_methods:
            aggregation_methods[column] = "sum"

    return data.resample(rule=pandas_granularity_map[granularity]).agg(
        aggregation_methods,
    )


def temperature_equivalent_to_degree_days(
    temperature_equivalent: pd.Series, base_temperature: float, cooling: bool = False
) -> pd.Series:
    """
    Convert temperature equivalent to degree days.

    Use cooling=True to convert cooling degree days.
    """

    if cooling:
        result = temperature_equivalent - base_temperature
    else:
        result = base_temperature - temperature_equivalent

    return result.clip(lower=0)
