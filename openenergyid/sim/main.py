"""Generic Simulation Analysis Module."""

from typing import Annotated, Union

import aiohttp
from pydantic import BaseModel, Field

from ..abstractsim import SimulationSummary, Simulator
from ..pvsim import PVSimulationInput, apply_simulation
from ..pvsim import get_simulator as get_pv_simulator
from ..simeval import EvaluationInput, compare_results, evaluate
from ..simeval.models import Frequency

# Here we define all types of simulations
SimulationInput = Annotated[Union[PVSimulationInput], Field(discriminator="type")]


def get_simulator(input_: SimulationInput) -> Simulator:
    """Get an instance of the simulator based on the input data."""
    # Only PV simulators for now
    return get_pv_simulator(input_)


class ExAnteData(EvaluationInput):
    """Ex-ante data for simulation analysis."""


class FullSimulationInput(BaseModel):
    """Full input for running a simulation analysis."""

    ex_ante_data: ExAnteData
    simulation_parameters: SimulationInput
    timezone: str = "Europe/Brussels"
    return_frequencies: list[Frequency] | None = Field(
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
