"""Battery Simulation Module."""

from typing import Annotated, Union

import pandas as pd
from pydantic import Field

from .selfconsumption import (
    SelfConsumptionBatterySimulationInput,
    SelfConsumptionBatterySimulator,
)

BatterySimulationInput = Annotated[
    Union[SelfConsumptionBatterySimulationInput], Field(discriminator="type")
]


def get_simulator(
    input_: BatterySimulationInput, data: pd.DataFrame
) -> SelfConsumptionBatterySimulator:
    """Get an instance of the battery simulator based on the input data."""
    if isinstance(input_, SelfConsumptionBatterySimulationInput):
        return SelfConsumptionBatterySimulator.from_pydantic(input_, data=data)
    raise ValueError(f"Unknown battery simulator type: {input_.type}")


def apply_simulation(input_data: pd.DataFrame, simulation_results: pd.DataFrame) -> pd.DataFrame:
    """Apply simulation results to input data."""
    df = input_data.copy()

    # Simply overwrite the relevant columns
    for col in df.columns:
        if col in simulation_results.columns:
            df[col] = simulation_results[col]

    return df
