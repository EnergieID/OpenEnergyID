"""Miscelaneous helper functions for the MVLR app."""

import pandas as pd

from openenergyid.enums import Granularity

pandas_granularity_map = {Granularity.P7D: "W-MON", Granularity.P1M: "MS", Granularity.P1D: "D"}


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
