"""Functions to create multi-indexed DataFrames for input and output data for energy sharing."""

import pandas as pd
from .const import GROSS_INJECTION, GROSS_OFFTAKE, KEY, NET_INJECTION, NET_OFFTAKE, SHARED_ENERGY


def create_multi_index_input_frame(
    gross_injection: pd.DataFrame,
    gross_offtake: pd.DataFrame,
    key: pd.DataFrame,
) -> pd.DataFrame:
    """Create a multi-indexed DataFrame with the input data for energy sharing."""
    gross_injection = gross_injection.copy()
    gross_offtake = gross_offtake.copy()
    key = key.copy()

    gross_injection.columns = pd.MultiIndex.from_product(
        [[GROSS_INJECTION], gross_injection.columns]
    )
    gross_offtake.columns = pd.MultiIndex.from_product([[GROSS_OFFTAKE], gross_offtake.columns])
    key.columns = pd.MultiIndex.from_product([[KEY], key.columns])

    df = pd.concat([gross_injection, gross_offtake, key], axis=1)

    return df


def create_multi_index_output_frame(
    net_injection: pd.DataFrame,
    net_offtake: pd.DataFrame,
    shared_energy: pd.DataFrame,
) -> pd.DataFrame:
    """Create a multi-indexed DataFrame with the output data for energy sharing."""
    net_injection = net_injection.copy()
    net_offtake = net_offtake.copy()
    shared_energy = shared_energy.copy()

    net_injection.columns = pd.MultiIndex.from_product([[NET_INJECTION], net_injection.columns])
    net_offtake.columns = pd.MultiIndex.from_product([[NET_OFFTAKE], net_offtake.columns])
    shared_energy.columns = pd.MultiIndex.from_product([[SHARED_ENERGY], shared_energy.columns])

    df = pd.concat([net_injection, net_offtake, shared_energy], axis=1)

    df = df.round(2)
    return df


def result_to_input_for_reiteration(result: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Create a multi-indexed DataFrame with the input data for energy sharing after the first iteration."""
    # We iterate again. The net injection of the previous result is taken as gross injection input
    # And the net offtake is taken as the gross offtake input
    # When a user's net offtake is 0, the key is set to 0; and the keys are re-normalized

    gross_injection = result[NET_INJECTION].copy()
    gross_offtake = result[NET_OFFTAKE].copy()

    # Take the original key, but replace the value with 0.0 if result[NET_OFFTAKE] is 0.0

    key = key.copy()
    key = key.where(~result[NET_OFFTAKE].eq(0), 0)

    # Re-normalize the keys

    key = key.div(key.sum(axis=1), axis=0)

    df = create_multi_index_input_frame(
        gross_injection=gross_injection, gross_offtake=gross_offtake, key=key
    )

    return df
