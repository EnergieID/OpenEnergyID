"""Re-simulate every year of an asset's life with degraded parameters, and cost it.

Year 1 is the simulation as it was run. For every later year the chain is walked again
on the same ex-ante data, with each PV stage's production and each battery's capacity
reduced by ``(1 - degradation_rate) ** (year - 1)``. The PV production is scaled from the
year-1 result, so nothing is downloaded or modelled again; the battery is simulated again,
because a smaller battery does not save proportionally less.

Costing a year against the contract takes far longer than simulating it, so not every year
is simulated: ``FinancialSettings.simulated_years`` years spread evenly over the horizon
are (``SIMULATED_YEARS`` by default), and the cost of the years in between is interpolated.
Degradation is smooth, so this loses next to nothing. Without degradation no year is
simulated again: every year costs the same as year 1.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .. import const
from ..battsim import BatterySimulationInput
from ..battsim import apply_simulation as apply_battery_simulation
from ..battsim import get_simulator as get_battery_simulator
from ..battsim.abstract import BatterySimulationInputAbstract
from ..battsim.selfconsumption import SelfConsumptionBatterySimulationInput
from ..cost import ContractCostSettings, CostSummary, compute_cost
from ..financial import (
    SIMULATED_YEARS,
    FinancialSettings,
    FinancialSummary,
    FinancialWarning,
    FinancialWarningCode,
    annualisation_factor,
    check_period,
    summarise,
)
from ..pvsim import apply_simulation as apply_pv_simulation


def degradation_factor(rate: float, year: int) -> float:
    """What is left of an asset's performance in ``year`` (year 1 is as new)."""
    return (1 - rate) ** (year - 1)


def simulated_years(horizon: int, count: int | None = SIMULATED_YEARS) -> list[int]:
    """The years to simulate: ``count`` years spread evenly from 1 to ``horizon``.

    ``None`` simulates every year.
    """
    if count is None or count >= horizon:
        return list(range(1, horizon + 1))
    return sorted({int(year) for year in np.rint(np.linspace(1, horizon, max(count, 2)))})


def _degraded_battery(parameters: BatterySimulationInput, factor: float) -> BatterySimulationInput:
    if not isinstance(parameters, SelfConsumptionBatterySimulationInput):
        raise ValueError(f"Cannot degrade a battery of type {parameters.type}.")
    capacity = parameters.capacity * factor
    return parameters.model_copy(
        update={"capacity": capacity, "initial_charge": min(parameters.initial_charge, capacity)}
    )


def simulate_year(
    ex_ante: pd.DataFrame,
    stage_results: Sequence[pd.DataFrame],
    parameters_list: Sequence,
    year: int,
) -> pd.DataFrame:
    """The ex-post frame of ``year``, with every stage degraded to that year.

    ``stage_results`` are the year-1 results of each stage, as in ``SimulationFrames``.
    """
    df = ex_ante.copy()
    for parameters, stage_result in zip(parameters_list, stage_results):
        factor = degradation_factor(parameters.degradation_rate, year)
        if isinstance(parameters, BatterySimulationInputAbstract):
            simulator = get_battery_simulator(
                _degraded_battery(parameters, factor) if factor != 1 else parameters, data=df
            )
            df = apply_battery_simulation(df, simulator.simulation_results)
        else:
            production = stage_result[const.ELECTRICITY_PRODUCED] * factor
            df = apply_pv_simulation(df, production)
    return df


def project_financials(
    frames,
    parameters_list: Sequence,
    cost_settings: ContractCostSettings,
    cost: CostSummary,
    settings: FinancialSettings,
) -> FinancialSummary:
    """Cost the years of the horizon and turn the savings into payback, NPV and IRR.

    ``frames`` is the ``SimulationFrames`` of year 1 and ``cost`` its contract cost, which
    already gives year 1. ``settings.simulated_years`` years spread over the horizon are
    simulated again and costed over the same window, with only the totals asked for
    (``None`` does every year); the cost of the years in between is interpolated. When no
    asset degrades, no year is simulated again. The ex-ante cost does not
    depend on the assets, so it is the same every year.
    """
    before, after = cost.ex_ante.total, cost.ex_post.total
    start, end = cost.ex_ante.start, cost.ex_ante.end
    if before is None or after is None or start is None or end is None:
        return FinancialSummary(
            settings=settings,
            warnings=[
                FinancialWarning(
                    code=FinancialWarningCode.SAVINGS_UNDEFINED,
                    message=(
                        "The cost before or after is undefined (see `cost.warnings`), so "
                        "there are no savings to project."
                    ),
                )
            ],
        )

    warnings = check_period(start, end)
    factor = annualisation_factor(start, end)

    horizon = settings.horizon_years
    years = list(range(1, horizon + 1))
    sampled = {1: after}
    if any(parameters.degradation_rate for parameters in parameters_list):
        year_settings = cost_settings.model_copy(
            update={
                "include_periods": False,
                "include_breakdown": False,
                "per_stage": False,
                "start": start,
                "end": end,
            }
        )
        for year in simulated_years(horizon, settings.simulated_years)[1:]:
            frame = simulate_year(frames.ex_ante, frames.stage_results, parameters_list, year)
            result, _ = compute_cost(frame, year_settings, preflight=False)
            if result.total is None:
                return FinancialSummary(
                    settings=settings,
                    warnings=[
                        *warnings,
                        FinancialWarning(
                            code=FinancialWarningCode.SAVINGS_UNDEFINED,
                            message=f"The cost of year {year} is undefined.",
                            context={"year": year},
                        ),
                    ],
                )
            sampled[year] = result.total
    costs_ex_post = np.interp(years, list(sampled), list(sampled.values()))

    summary = summarise(
        settings,
        costs_ex_ante=[before * factor] * horizon,
        costs_ex_post=[float(cost_ex_post) * factor for cost_ex_post in costs_ex_post],
        warnings=warnings,
    )
    for year in summary.yearly:
        year.simulated = year.year in sampled
    return summary
