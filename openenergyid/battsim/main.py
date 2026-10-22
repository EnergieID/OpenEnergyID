"""Battery Simulation Module."""

from typing import Annotated, Union

import pandas as pd
from pydantic import Field

from .. import const
from .selfconsumption import (
    SelfConsumptionBatterySimulationInput,
    SelfConsumptionBatterySimulator,
)

BATTERY_FLOW_COLUMNS = (const.ELECTRICITY_CHARGED, const.ELECTRICITY_DISCHARGED)

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
        if col in simulation_results.columns and col not in BATTERY_FLOW_COLUMNS:
            df[col] = simulation_results[col]

    # Carry the battery flows along, so later evaluation knows what went into storage.
    # Summed, in case the chain holds more than one battery.
    for col in BATTERY_FLOW_COLUMNS:
        if col in simulation_results.columns:
            df[col] = simulation_results[col] + (df[col] if col in df.columns else 0.0)

    return df
