from typing import Annotated, Union

import pandas as pd
from pydantic import Field

from .elia import EliaPVSimulationInput, EliaPVSimulator
from .pvlib import PVLibSimulationInput, PVLibSimulator

PVSimulationInput = Annotated[
    Union[PVLibSimulationInput, EliaPVSimulationInput], Field(discriminator="type")
]


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

    if "electricity_produced" not in df.columns:
        df["electricity_produced"] = 0.0
    df["electricity_produced"] = df["electricity_produced"] + simulation_results

    new_delivered = (df["electricity_delivered"] - simulation_results).clip(lower=0.0)
    self_consumed = df["electricity_delivered"] - new_delivered
    df["electricity_delivered"] = new_delivered

    exported = simulation_results - self_consumed

    if "electricity_exported" not in df.columns:
        df["electricity_exported"] = 0.0
    df["electricity_exported"] = df["electricity_exported"] + exported

    return df
