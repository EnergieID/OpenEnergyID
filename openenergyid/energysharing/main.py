"""Main Calcuation Module for Energy Sharing."""

import pandas as pd
from .models import CalculationMethod
from .const import GROSS_INJECTION, GROSS_OFFTAKE, KEY, NET_INJECTION, NET_OFFTAKE, SHARED_ENERGY
from .data_formatting import create_multi_index_output_frame, result_to_input_for_reiteration


def _calculate(df: pd.DataFrame, method: CalculationMethod) -> pd.DataFrame:
    """Calculate the energy sharing for the given input data. This function is not iterative."""
    # Step 1: Calculate the maximum available gross injection that can be shared
    # A participant cannot share their injection with themselves

    # Take the injection of P1, and divide it per participant as per their key

    injections_to_share = []
    rest = {}

    for participant in df.columns.levels[1]:
        injection_to_share = df[GROSS_INJECTION][participant].copy()

        if method == CalculationMethod.RELATIVE or method == CalculationMethod.OPTIMAL:
            # Set the key of the current participant to 0
            # Re-normalize the keys for the other participants
            key = df[KEY].copy()
            key.loc[:, participant] = 0
            key = key.div(key.sum(axis=1), axis=0)
        elif method == CalculationMethod.FIXED:
            key = df[KEY].copy()

        # Multiply injection_to_share with the key of each participant
        shared_by_participant = (injection_to_share * key.T).T
        shared_by_participant.fillna(0, inplace=True)
        # Set the value for the current participant to 0
        shared_by_participant.loc[:, participant] = 0

        # Put the not shared injection in the rest
        rest[participant] = injection_to_share - shared_by_participant.sum(axis=1)

        injections_to_share.append(shared_by_participant)

    # Sum the injections to share
    max_allocated_injection = sum(injections_to_share)

    # Concat the rest
    injection_that_cannot_be_shared = pd.concat(rest, axis=1)

    # Step 2: Calculate the Net Offtake, by assigning the injections to each participant
    # But, a participant cannot receive more than their offtake

    net_offtake = df[GROSS_OFFTAKE] - max_allocated_injection

    # Sum all negative values into a column "Not Shared"
    not_shared_after_assignment = net_offtake.clip(upper=0).sum(axis=1) * -1

    # Clip the values to 0
    net_offtake = net_offtake.clip(lower=0)

    # Calculate the amount of actual shared energy
    # This is the difference between the gross offtake and the net offtake
    shared_energy = df[GROSS_OFFTAKE] - net_offtake

    # Step 3: Assign the Rests back to the original injectors

    # The energy that is not shared after assignment
    # should be divided back to the original injectors
    # A ratio of the original injection should be used

    re_distributed_not_shared = (
        (df[GROSS_INJECTION].T / df[GROSS_INJECTION].sum(axis=1)) * not_shared_after_assignment
    ).T
    re_distributed_not_shared.fillna(0, inplace=True)

    # The nett injection is the sum of:
    # the injection that cannot be shared to begin with
    # (because participants cannot share with themselves)
    # and the injection that cannot be shared after assignment
    # (because participants cannot receive more than their offtake)

    net_injection = injection_that_cannot_be_shared + re_distributed_not_shared

    result = create_multi_index_output_frame(
        net_injection=net_injection, net_offtake=net_offtake, shared_energy=shared_energy
    )

    return result


def calculate(df: pd.DataFrame, method: CalculationMethod) -> pd.DataFrame:
    """Calculate the energy sharing for the given input data.

    This function is iterative if the method is optimal."""
    result = _calculate(df, method)

    if method in [CalculationMethod.FIXED, CalculationMethod.RELATIVE]:
        return result

    # Optimal method, we iterate until the amount of shared energy is 0
    final_result = result.copy()
    while not result[SHARED_ENERGY].eq(0).all().all():
        df = result_to_input_for_reiteration(result, df[KEY])
        result = _calculate(df, method)

        # Add the result to the final result
        # Overwrite NET_INJECTION and NET_OFFTAKE, Sum SHARED_ENERGY
        final_result[NET_INJECTION] = result[NET_INJECTION]
        final_result[NET_OFFTAKE] = result[NET_OFFTAKE]
        final_result[SHARED_ENERGY] += result[SHARED_ENERGY]

    return final_result
