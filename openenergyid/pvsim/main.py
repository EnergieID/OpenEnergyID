from typing import Annotated, Union

import pandas as pd
from pydantic import BaseModel, Field

from ..const import ELECTRICITY_DELIVERED, ELECTRICITY_EXPORTED, ELECTRICITY_PRODUCED
from ..models import TimeDataFrame, TimeSeries
from .elia import EliaPVSimulationInput, EliaPVSimulator
from .pvlib import PVLibSimulationInput, PVLibSimulator

PVSimulationInput = Annotated[
    Union[PVLibSimulationInput, EliaPVSimulationInput], Field(discriminator="type")
]


class PVSimulationSummary(BaseModel):
    """Summary of a PV simulation including ex-ante, simulation results, ex-post, and comparisons."""

    ex_ante: dict[str, TimeDataFrame | dict[str, float]]
    simulation_result: dict[str, TimeSeries | dict[str, float]]
    ex_post: dict[str, TimeDataFrame | dict[str, float]]
    comparison: dict[str, dict[str, TimeDataFrame | dict[str, float]]]

    @classmethod
    def from_simulation(
        cls,
        ex_ante: dict[str, pd.DataFrame | pd.Series],
        simulation_result: dict[str, pd.DataFrame | pd.Series],
        ex_post: dict[str, pd.DataFrame | pd.Series],
        comparison: dict[str, dict[str, pd.DataFrame | pd.Series]],
    ) -> "PVSimulationSummary":
        """Create a PVSimulationSummary from simulation data."""
        ea = {
            k: TimeDataFrame.from_pandas(v) if isinstance(v, pd.DataFrame) else v.to_dict()
            for k, v in ex_ante.items()
        }
        sr = {
            k: TimeSeries.from_pandas(v.squeeze(axis=1))  # type: ignore
            if isinstance(v, pd.DataFrame)
            else v.to_dict()
            for k, v in simulation_result.items()
        }
        ep = {
            k: TimeDataFrame.from_pandas(v) if isinstance(v, pd.DataFrame) else v.to_dict()
            for k, v in ex_post.items()
        }
        c = {
            k: {
                kk: TimeDataFrame.from_pandas(vv) if isinstance(vv, pd.DataFrame) else vv.to_dict()
                for kk, vv in v.items()
            }
            for k, v in comparison.items()
        }
        return cls(
            ex_ante=ea,
            simulation_result=sr,
            ex_post=ep,
            comparison=c,
        )


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
