"""PV Simulation module."""

from typing import Annotated, Union

import pandas as pd
from pydantic import Field

from ..abstractsim import SimulationSummary
from ..const import ELECTRICITY_DELIVERED, ELECTRICITY_EXPORTED, ELECTRICITY_PRODUCED
from .elia import EliaPVSimulationInput, EliaPVSimulator
from .pvlib import PVLibSimulationInput, PVLibSimulator

PVSimulationInput = Annotated[
    Union[PVLibSimulationInput, EliaPVSimulationInput], Field(discriminator="type")
]


class PVSimulationSummary(SimulationSummary):
    """Summary of a PV simulation including ex-ante, simulation results, ex-post, and comparisons."""


def get_simulator(input_: PVSimulationInput) -> PVLibSimulator | EliaPVSimulator:
    """Get an instance of the simulator based on the input data."""
    if isinstance(input_, PVLibSimulationInput):
        return PVLibSimulator.from_pydantic(input_)
    if isinstance(input_, EliaPVSimulationInput):
        return EliaPVSimulator.from_pydantic(input_)
    raise ValueError(f"Unknown simulator type: {input_.type}")


def apply_simulation(input_data: pd.DataFrame, simulation_results: pd.Series) -> pd.DataFrame:
    """Apply simulation results to input data."""
    df = input_data.copy()

    if ELECTRICITY_PRODUCED not in df.columns:
        df[ELECTRICITY_PRODUCED] = 0.0
    df[ELECTRICITY_PRODUCED] = df[ELECTRICITY_PRODUCED] + simulation_results

    new_delivered = (df[ELECTRICITY_DELIVERED] - simulation_results).clip(lower=0.0)
    self_consumed = df[ELECTRICITY_DELIVERED] - new_delivered
    df[ELECTRICITY_DELIVERED] = new_delivered

    exported = simulation_results - self_consumed

    if ELECTRICITY_EXPORTED not in df.columns:
        df[ELECTRICITY_EXPORTED] = 0.0
    df[ELECTRICITY_EXPORTED] = df[ELECTRICITY_EXPORTED] + exported

    return df
