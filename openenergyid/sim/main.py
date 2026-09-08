"""Generic Simulation Analysis Module."""

import asyncio
from typing import Annotated, NamedTuple, Union, cast

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from .. import const
from ..abstractsim import SimulationSummary, Simulator
from ..battsim import BatterySimulationInput, BatterySimulator
from ..battsim import apply_simulation as apply_battery_simulation
from ..battsim import get_simulator as get_battery_simulator
from ..cost import ContractCostSettings, cost_simulation
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


PRICE_COLUMNS = (
    const.PRICE_ELECTRICITY_DELIVERED,
    const.PRICE_ELECTRICITY_EXPORTED,
)


def _conflicting_cost_columns(cost: ContractCostSettings | None, columns: list[str]) -> list[str]:
    """Price columns that clash with a contract-based `cost` setting, if any."""
    if cost is None:
        return []
    return sorted(set(PRICE_COLUMNS) & set(columns))


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
    cost: ContractCostSettings | None = Field(
        default=None,
        description=(
            "Optional tariff contract to cost the simulation against, producing a full "
            "bill before and after. Mutually exclusive with supplying "
            "`price_electricity_delivered` / `price_electricity_exported` columns in "
            "`ex_ante_data`, which drive the simpler price-times-volume calculation."
        ),
    )

    @model_validator(mode="after")
    def _reject_conflicting_cost_inputs(self) -> "FullSimulationInput":
        """Refuse to guess which of two cost mechanisms the caller meant."""
        clash = _conflicting_cost_columns(self.cost, self.ex_ante_data.columns)
        if clash:
            raise ValueError(
                "Conflicting cost inputs: `cost` supplies a tariff contract while "
                f"ex_ante_data also carries price columns {clash}. Pick one mechanism: "
                "remove the price columns to use contract-based costing, or drop `cost` "
                "to use per-interval prices."
            )
        return self


class SimulationFrames(NamedTuple):
    """Every frame produced by a simulation run.

    ``stage_results`` holds each simulator's own output (production, charge/discharge),
    while ``stages`` holds the cumulative grid frame after each stage. Only the latter
    is billable.
    """

    ex_ante: pd.DataFrame
    stage_results: list[pd.DataFrame]
    stages: list[pd.DataFrame]
    ex_post: pd.DataFrame


async def simulate_frames(
    input_: FullSimulationInput, session: aiohttp.ClientSession | None = None
) -> SimulationFrames:
    """Run the simulation chain and return every intermediate frame."""
    df = input_.ex_ante_data.to_pandas(timezone=input_.timezone)
    ex_ante = df.copy()

    if not isinstance(input_.simulation_parameters, list):
        parameters_list = [input_.simulation_parameters]
    else:
        parameters_list = input_.simulation_parameters

    stage_results: list[pd.DataFrame] = []
    stages: list[pd.DataFrame] = []
    for parameters in parameters_list:
        simulator: Simulator = get_simulator(parameters, data=df)
        await simulator.load_resources(session=session)
        stage_results.append(simulator.result_as_frame())

        if isinstance(simulator, BatterySimulator):
            df = apply_battery_simulation(df, simulator.simulation_results)
        else:
            df = apply_pv_simulation(df, simulator.simulation_results)
        stages.append(df.copy())

    return SimulationFrames(
        ex_ante=ex_ante,
        stage_results=stage_results,
        stages=stages,
        ex_post=df,
    )


async def run_simulation(
    input_: FullSimulationInput, session: aiohttp.ClientSession | None = None
) -> SimulationSummary:
    """Run the full simulation analysis workflow."""
    frames = await simulate_frames(input_, session=session)

    ex_ante_eval = evaluate(frames.ex_ante, return_frequencies=input_.return_frequencies)
    sim_evals = [
        evaluate(frame, return_frequencies=input_.return_frequencies)
        for frame in frames.stage_results
    ]
    post_eval = evaluate(frames.ex_post, return_frequencies=input_.return_frequencies)

    comparison = compare_results(ex_ante_eval, post_eval)

    ex_ante_eval_dict = eval_to_dict(ex_ante_eval)
    sim_eval_dict = [eval_to_dict(value) for value in sim_evals]
    if len(sim_evals) == 1 and not isinstance(input_.simulation_parameters, list):
        sim_eval_dict = sim_eval_dict[0]
    post_eval_dict = eval_to_dict(post_eval)
    comparison_dict = comparison_to_dict(comparison)

    cost = None
    if input_.cost is not None:
        # Re-check for a clash: ex_ante_data.columns is a plain mutable list, so the
        # model_validator that ran at construction time may no longer hold by now.
        clash = _conflicting_cost_columns(input_.cost, input_.ex_ante_data.columns)
        if clash:
            raise ValueError(
                "Conflicting cost inputs: `cost` supplies a tariff contract while "
                f"ex_ante_data also carries price columns {clash}. Pick one mechanism: "
                "remove the price columns to use contract-based costing, or drop `cost` "
                "to use per-interval prices."
            )
        # Contract costing is synchronous, CPU-bound pandas work; keep it off the loop.
        cost = await asyncio.to_thread(cost_simulation, frames, input_.cost)

    summary = SimulationSummary(
        ex_ante=ex_ante_eval_dict,
        simulation_result=sim_eval_dict,
        ex_post=post_eval_dict,
        comparison=comparison_dict,
        cost=cost,
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
