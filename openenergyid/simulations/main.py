"""Generic Simulation Analysis Module."""

from typing import Annotated, Self, Union

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field

from ..models import TimeDataFrame
from ..pvsim import PVSimulationInput, apply_simulation
from ..pvsim import get_simulator as get_pv_simulator
from ..simeval import EvaluationInput, compare_results, evaluate
from .abstract import Simulator

# Here we define all types of simulations
SimulationInput = Annotated[Union[PVSimulationInput], Field(discriminator="type")]


def get_simulator(input_: SimulationInput) -> Simulator:
    """Get an instance of the simulator based on the input data."""
    # Only PV simulators for now
    return get_pv_simulator(input_)


class ExAnteData(EvaluationInput):
    """Ex-ante data for simulation analysis."""


class SimulationSummary(BaseModel):
    """Summary of a simulation including ex-ante, simulation results, ex-post, and comparisons."""

    ex_ante: dict[str, TimeDataFrame | dict[str, float]]
    simulation_result: dict[str, TimeDataFrame | dict[str, float]]
    ex_post: dict[str, TimeDataFrame | dict[str, float]]
    comparison: dict[str, dict[str, TimeDataFrame | dict[str, float]]]

    @classmethod
    def from_simulation(
        cls,
        ex_ante: dict[str, pd.DataFrame | pd.Series],
        simulation_result: dict[str, pd.DataFrame | pd.Series],
        ex_post: dict[str, pd.DataFrame | pd.Series],
        comparison: dict[str, dict[str, pd.DataFrame | pd.Series]],
    ) -> Self:
        """Create a SimulationSummary from simulation data."""
        ea = {
            k: TimeDataFrame.from_pandas(v) if isinstance(v, pd.DataFrame) else v.to_dict()
            for k, v in ex_ante.items()
        }
        sr = {
            k: TimeDataFrame.from_pandas(v) if isinstance(v, pd.DataFrame) else v.to_dict()
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


class FullSimulationInput(BaseModel):
    """Full input for running a simulation analysis."""

    ex_ante_data: ExAnteData
    simulation_parameters: SimulationInput
    timezone: str = "Europe/Brussels"
    return_frequencies: list[str] | None = Field(
        default=None,
        examples=["MS", "W-MON"],
        description="Optional list of frequencies that should be included in the analysis. Be default, only `total` is included, but you can add more here. Uses the Pandas freqstr.",
    )


async def run_simulation(
    input_: FullSimulationInput, session: aiohttp.ClientSession
) -> SimulationSummary:
    """Run the full simulation analysis workflow."""
    df = input_.ex_ante_data.to_pandas(timezone=input_.timezone)

    ex_ante_eval = evaluate(df, return_frequencies=input_.return_frequencies)

    simulator: Simulator = get_simulator(input_.simulation_parameters)
    await simulator.load_resources(session=session)

    sim_eval = evaluate(simulator.result_as_frame(), return_frequencies=input_.return_frequencies)

    df_post = apply_simulation(df, simulator.simulation_results)

    post_eval = evaluate(df_post, return_frequencies=input_.return_frequencies)

    comparison = compare_results(ex_ante_eval, post_eval)

    summary = SimulationSummary.from_simulation(
        ex_ante=ex_ante_eval,
        simulation_result=sim_eval,
        ex_post=post_eval,
        comparison=comparison,
    )

    return summary
