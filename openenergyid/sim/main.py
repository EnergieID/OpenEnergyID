"""Generic Simulation Analysis Module."""

from typing import Annotated, Union, cast

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field

from ..abstractsim import SimulationSummary, Simulator
from ..battsim import BatterySimulationInput, BatterySimulator
from ..battsim import apply_simulation as apply_battery_simulation
from ..battsim import get_simulator as get_battery_simulator
from ..models import TimeDataFrame
from ..pvsim import PVSimulationInput
from ..pvsim import apply_simulation as apply_pv_simulation
from ..pvsim import get_simulator as get_pv_simulator
from ..simeval import compare_results, evaluate

# Here we define all types of simulations
SimulationInput = Annotated[
    Union[PVSimulationInput, BatterySimulationInput], Field(discriminator="type")
]


def get_simulator(input_: SimulationInput, data: pd.DataFrame | None = None) -> Simulator:
    """Get an instance of the simulator based on the input data."""
    if input_.type in ["pvlibsimulation", "eliapvsimulation"]:
        input_ = cast(PVSimulationInput, input_)
        return get_pv_simulator(input_)
    elif input_.type in ["selfconsumptionbatterysimulation"]:
        input_ = cast(BatterySimulationInput, input_)
        if data is None:
            raise ValueError("Data must be provided for battery simulations.")
        return get_battery_simulator(input_, data=data)
    else:
        raise ValueError(f"Unknown simulator type: {input_.type}")


class ExAnteData(TimeDataFrame):
    """Ex-ante data for simulation analysis."""


class FullSimulationInput(BaseModel):
    """Full input for running a simulation analysis."""

    ex_ante_data: ExAnteData
    simulation_parameters: SimulationInput | list[SimulationInput]
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

    if not isinstance(input_.simulation_parameters, list):
        parameters_list = [input_.simulation_parameters]
    else:
        parameters_list = input_.simulation_parameters

    sim_evals = {}
    for parameters in parameters_list:
        simulator: Simulator = get_simulator(parameters, data=df)
        await simulator.load_resources(session=session)
        sim_eval = evaluate(
            simulator.result_as_frame(), return_frequencies=input_.return_frequencies
        )
        sim_evals[parameters.type] = sim_eval

        if isinstance(simulator, BatterySimulator):
            df_post = apply_battery_simulation(df, simulator.simulation_results)
        else:
            df_post = apply_pv_simulation(df, simulator.simulation_results)
        df = df_post  # Update df for the next simulation if there are multiple simulations

    post_eval = evaluate(df_post, return_frequencies=input_.return_frequencies)

    comparison = compare_results(ex_ante_eval, post_eval)

    ex_ante_eval_dict = eval_to_dict(ex_ante_eval)
    sim_eval_dict = {key: eval_to_dict(value) for key, value in sim_evals.items()}
    if len(sim_evals) == 1:
        sim_eval_dict = sim_eval_dict[next(iter(sim_evals))]
    post_eval_dict = eval_to_dict(post_eval)
    comparison_dict = comparison_to_dict(comparison)

    summary = SimulationSummary(
        ex_ante=ex_ante_eval_dict,
        simulation_result=sim_eval_dict,
        ex_post=post_eval_dict,
        comparison=comparison_dict,
    )

    return summary


def eval_to_dict(eval_result: dict[str, pd.DataFrame | pd.Series]) -> dict[str, dict]:
    """Convert evaluation results to a dictionary format."""
    result = {}
    for key, value in eval_result.items():
        if isinstance(value, pd.DataFrame):
            result[key] = TimeDataFrame.from_pandas(value).model_dump(mode="json")
        elif isinstance(value, pd.Series):
            result[key] = value.to_dict()
        else:
            raise ValueError(f"Unsupported type for evaluation result: {type(value)}")

    return result


def comparison_to_dict(
    comparison_result: dict[str, dict[str, pd.DataFrame | pd.Series]],
) -> dict[str, dict[str, dict]]:
    """Convert comparison results to a dictionary format."""
    result = {}
    for key, sub_dict in comparison_result.items():
        result[key] = eval_to_dict(sub_dict)

    return result
