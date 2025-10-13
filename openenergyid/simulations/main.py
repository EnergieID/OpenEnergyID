"""Generic Simulation Analysis Module."""

import aiohttp
from pydantic import BaseModel

from ..pvsim import (
    PVSimulationInput,
    PVSimulationSummary,
    apply_simulation,
    get_simulator,
)
from ..simeval import EvaluationInput, compare_results, evaluate


class ExAnteData(EvaluationInput):
    """Ex-ante data for simulation analysis."""


class SimulationSummary(PVSimulationSummary):
    """Summary of the simulation analysis."""


class FullSimulationInput(BaseModel):
    """Full input for running a simulation analysis."""

    ex_ante_data: ExAnteData
    simulation_parameters: PVSimulationInput
    timezone: str = "Europe/Brussels"
    return_frequencies: list[str] | None = None


async def analyze(input_: FullSimulationInput, session: aiohttp.ClientSession) -> SimulationSummary:
    """Run the full simulation analysis workflow."""
    df = input_.ex_ante_data.to_pandas(timezone=input_.timezone)

    ex_ante_eval = evaluate(df, return_frequencies=input_.return_frequencies)

    simulator = get_simulator(input_.simulation_parameters)
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
