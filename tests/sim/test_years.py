"""Tests for the year-by-year re-simulation with degraded assets, and its payback."""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from openenergyid import const
from openenergyid.battsim import BatterySimulationInput
from openenergyid.financial import FinancialWarningCode
from openenergyid.models import TimeDataFrame
from openenergyid.sim import FullSimulationInput, run_simulation, simulate_frames
from openenergyid.sim.years import degradation_factor, simulate_year, simulated_years

from ..cost.conftest import (  # noqa: F401
    isolate_index_registry,
    make_contract,
    make_frame,
)

BATTERY = {
    "type": "selfconsumptionbatterysimulation",
    "capacity": 5.0,
    "power": 2.0,
}
FINANCIAL = {"investment": 3000.0, "discount_rate": 0.03, "horizon_years": 10}


def _frame(periods: int = 96 * 365) -> pd.DataFrame:
    """Surplus at midday and offtake in the evening, so a battery has work to do."""
    frame = make_frame(start="2023-01-01", periods=periods, produced=None)
    hour = np.asarray(frame.index.hour)
    midday = (hour >= 10) & (hour < 16)
    frame[const.ELECTRICITY_DELIVERED] = np.where(midday, 0.0, 0.3)
    frame[const.ELECTRICITY_EXPORTED] = np.where(midday, 0.5, 0.0)
    return frame


def _input(battery: dict | None = None, frame: pd.DataFrame | None = None, **overrides):
    payload = {
        "ex_ante_data": TimeDataFrame.from_pandas(_frame() if frame is None else frame).model_dump(
            mode="json"
        ),
        "simulation_parameters": BATTERY if battery is None else battery,
        "timezone": "Europe/Brussels",
        "cost": {"contract": make_contract().model_dump(mode="json"), "output_resolution": "P1M"},
        "financial": FINANCIAL,
    }
    payload.update(overrides)
    return FullSimulationInput.model_validate(payload)


def _run(input_: FullSimulationInput):
    return asyncio.run(run_simulation(input_))


class TestInputs:
    def test_degradation_defaults_to_none(self) -> None:
        input_ = _input()

        assert input_.parameters_list()[0].degradation_rate == 0.0

    @pytest.mark.parametrize("rate", [-0.01, 1.0])
    def test_degradation_out_of_range_is_refused(self, rate) -> None:
        with pytest.raises(ValidationError):
            _input(battery={**BATTERY, "degradation_rate": rate})

    def test_pv_takes_a_degradation_rate(self) -> None:
        from pydantic import TypeAdapter

        from openenergyid.pvsim import PVSimulationInput

        pv = TypeAdapter(PVSimulationInput).validate_python(
            {
                "type": "eliapvsimulation",
                "start": "2023-01-01",
                "end": "2024-01-01",
                "panel_power": 4000,
                "inverter_power": 4000,
                "region": "Flanders",
                "degradation_rate": 0.005,
            }
        )
        assert pv.degradation_rate == 0.005

    def test_financial_needs_a_contract(self) -> None:
        with pytest.raises(ValidationError, match="financial"):
            _input(cost=None)

    def test_the_battery_input_type_still_resolves(self) -> None:
        from pydantic import TypeAdapter

        battery = TypeAdapter(BatterySimulationInput).validate_python(
            {**BATTERY, "degradation_rate": 0.02}
        )
        assert battery.degradation_rate == 0.02


class TestSimulatedYears:
    def test_every_year_when_the_horizon_is_short(self) -> None:
        assert simulated_years(5) == [1, 2, 3, 4, 5]

    def test_spread_over_a_long_horizon(self) -> None:
        years = simulated_years(25)

        assert years[0] == 1 and years[-1] == 25
        assert len(years) == 7

    def test_none_simulates_every_year(self) -> None:
        assert simulated_years(25, None) == list(range(1, 26))

    @pytest.mark.parametrize("count", [0, 1])
    def test_fewer_than_two_simulated_years_is_refused(self, count) -> None:
        with pytest.raises(ValidationError):
            _input(financial={**FINANCIAL, "simulated_years": count})

    def test_degradation_factor(self) -> None:
        assert degradation_factor(0.02, 1) == 1.0
        assert degradation_factor(0.02, 11) == pytest.approx(0.98**10)


class TestSimulateYear:
    def test_year_one_is_the_simulation(self) -> None:
        input_ = _input(battery={**BATTERY, "degradation_rate": 0.02})
        frames = asyncio.run(simulate_frames(input_))

        year_one = simulate_year(frames.ex_ante, frames.stage_results, input_.parameters_list(), 1)

        pd.testing.assert_frame_equal(year_one, frames.ex_post)

    def test_a_degraded_battery_shifts_less(self) -> None:
        input_ = _input(battery={**BATTERY, "degradation_rate": 0.1})
        frames = asyncio.run(simulate_frames(input_))
        parameters = input_.parameters_list()

        later = simulate_year(frames.ex_ante, frames.stage_results, parameters, 10)

        assert (
            later[const.ELECTRICITY_DELIVERED].sum()
            > frames.ex_post[const.ELECTRICITY_DELIVERED].sum()
        )


class TestFinancialSummary:
    def test_absent_without_financial_settings(self) -> None:
        summary = _run(_input(financial=None))

        assert summary.financial is None

    def test_does_not_change_the_rest_of_the_summary(self) -> None:
        without = _run(_input(financial=None))
        with_ = _run(_input(battery={**BATTERY, "degradation_rate": 0.02}))

        assert with_.model_dump(exclude={"financial"}) == without.model_dump(exclude={"financial"})

    def test_without_degradation_every_year_saves_the_same(self) -> None:
        summary = _run(_input())

        savings = [year.savings for year in summary.financial.yearly]
        assert len(savings) == FINANCIAL["horizon_years"]
        assert savings[0] > 0
        assert savings == pytest.approx([savings[0]] * len(savings))

    def test_year_one_is_the_annualised_cost_saving(self) -> None:
        summary = _run(_input())

        saving = summary.cost.ex_ante.total - summary.cost.ex_post.total
        days = (summary.cost.ex_ante.end - summary.cost.ex_ante.start).days
        assert summary.financial.yearly[0].savings == pytest.approx(saving * 365.25 / days)

    def test_a_degrading_battery_saves_less_every_year(self) -> None:
        summary = _run(_input(battery={**BATTERY, "degradation_rate": 0.1}))

        savings = [year.savings for year in summary.financial.yearly]
        assert all(later < earlier for earlier, later in zip(savings, savings[1:]))

    def test_degradation_lengthens_the_payback(self) -> None:
        financial = {**FINANCIAL, "investment": 300.0}
        new = _run(_input(financial=financial)).financial
        degrading = _run(
            _input(battery={**BATTERY, "degradation_rate": 0.1}, financial=financial)
        ).financial

        assert new.payback_years is not None and degrading.payback_years is not None
        assert degrading.payback_years > new.payback_years
        assert degrading.npv < new.npv
        assert degrading.irr < new.irr

    def test_a_short_period_is_annualised_with_a_warning(self) -> None:
        summary = _run(_input(frame=_frame(periods=96 * 59)))

        codes = [warning.code for warning in summary.financial.warnings]
        assert FinancialWarningCode.SHORT_PERIOD_ANNUALISED in codes

    def test_survives_json_serialisation(self) -> None:
        summary = _run(_input(battery={**BATTERY, "degradation_rate": 0.02}))

        restored = json.loads(summary.model_dump_json())

        assert restored["financial"]["settings"]["investment"] == FINANCIAL["investment"]
        assert len(restored["financial"]["yearly"]) == FINANCIAL["horizon_years"]

    def test_without_degradation_nothing_is_costed_again(self, monkeypatch) -> None:
        from openenergyid.sim import years

        calls = []
        monkeypatch.setattr(years, "compute_cost", lambda *args, **kwargs: calls.append(args))

        summary = _run(_input())

        assert calls == []
        assert [year.simulated for year in summary.financial.yearly] == [True] + [False] * 9

    @pytest.mark.parametrize("count, expected", [(3, 3), (None, 10)])
    def test_the_number_of_simulated_years_is_configurable(self, count, expected) -> None:
        summary = _run(
            _input(
                battery={**BATTERY, "degradation_rate": 0.1},
                financial={**FINANCIAL, "simulated_years": count},
            )
        )

        assert sum(year.simulated for year in summary.financial.yearly) == expected
