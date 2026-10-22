"""Tests for the direct self-consumption figures: solar used without going through a battery."""

import asyncio

import pandas as pd
import pytest

from openenergyid import const
from openenergyid.battsim import apply_simulation
from openenergyid.models import TimeDataFrame
from openenergyid.sim import FullSimulationInput, run_simulation
from openenergyid.simeval import evaluate

TZ = "Europe/Brussels"


def _index(periods: int, start: str = "2025-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="15min", tz=TZ)


class TestEvaluate:
    def test_without_battery_equals_self_consumption(self) -> None:
        frame = pd.DataFrame(
            {
                const.ELECTRICITY_DELIVERED: [0.2, 0.0],
                const.ELECTRICITY_EXPORTED: [0.0, 0.3],
                const.ELECTRICITY_PRODUCED: [0.1, 0.5],
            },
            index=_index(2),
        )

        total = evaluate(frame)["total"]

        assert total[const.ELECTRICITY_SELF_CONSUMED_DIRECT] == pytest.approx(0.3)
        assert total[const.ELECTRICITY_SELF_CONSUMED_DIRECT] == pytest.approx(
            total[const.ELECTRICITY_SELF_CONSUMED]
        )
        assert total[const.RATIO_SELF_CONSUMPTION_DIRECT] == pytest.approx(
            total[const.RATIO_SELF_CONSUMPTION]
        )

    def test_charged_energy_is_not_direct_use(self) -> None:
        # Charged energy is negative, as in the battery simulation results
        frame = pd.DataFrame(
            {
                const.ELECTRICITY_DELIVERED: [0.0, 0.0],
                const.ELECTRICITY_EXPORTED: [0.1, 0.0],
                const.ELECTRICITY_PRODUCED: [1.0, 0.5],
                const.ELECTRICITY_CHARGED: [-0.4, -0.1],
            },
            index=_index(2),
        )

        total = evaluate(frame)["total"]

        # produced 1.5 - exported 0.1 - charged 0.5
        assert total[const.ELECTRICITY_SELF_CONSUMED] == pytest.approx(1.4)
        assert total[const.ELECTRICITY_SELF_CONSUMED_DIRECT] == pytest.approx(0.9)
        assert total[const.RATIO_SELF_CONSUMPTION_DIRECT] == pytest.approx(0.9 / 1.5)

    def test_per_month(self) -> None:
        index = pd.DatetimeIndex(
            [pd.Timestamp("2025-01-15 12:00", tz=TZ), pd.Timestamp("2025-02-15 12:00", tz=TZ)]
        )
        frame = pd.DataFrame(
            {
                const.ELECTRICITY_DELIVERED: [0.0, 0.0],
                const.ELECTRICITY_EXPORTED: [0.0, 0.2],
                const.ELECTRICITY_PRODUCED: [1.0, 1.0],
                const.ELECTRICITY_CHARGED: [-0.5, 0.0],
            },
            index=index,
        )

        monthly = evaluate(frame, return_frequencies=["MS"])["MS"]

        assert monthly[const.ELECTRICITY_SELF_CONSUMED_DIRECT].tolist() == pytest.approx([0.5, 0.8])

    def test_absent_without_production(self) -> None:
        frame = pd.DataFrame(
            {const.ELECTRICITY_DELIVERED: [0.2], const.ELECTRICITY_EXPORTED: [0.0]},
            index=_index(1),
        )

        total = evaluate(frame)["total"]

        assert const.ELECTRICITY_SELF_CONSUMED_DIRECT not in total


class TestApplyBatterySimulation:
    def test_battery_flows_are_carried_and_summed(self) -> None:
        index = _index(2)
        data = pd.DataFrame(
            {const.ELECTRICITY_DELIVERED: [0.5, 0.5], const.ELECTRICITY_EXPORTED: [0.2, 0.2]},
            index=index,
        )
        battery = pd.DataFrame(
            {
                const.ELECTRICITY_DELIVERED: [0.4, 0.4],
                const.ELECTRICITY_EXPORTED: [0.0, 0.0],
                const.ELECTRICITY_CHARGED: [-0.2, -0.2],
                const.ELECTRICITY_DISCHARGED: [0.1, 0.1],
            },
            index=index,
        )

        once = apply_simulation(data, battery)
        twice = apply_simulation(once, battery)

        assert once[const.ELECTRICITY_CHARGED].tolist() == pytest.approx([-0.2, -0.2])
        assert once[const.ELECTRICITY_DELIVERED].tolist() == pytest.approx([0.4, 0.4])
        assert twice[const.ELECTRICITY_CHARGED].tolist() == pytest.approx([-0.4, -0.4])
        assert twice[const.ELECTRICITY_DISCHARGED].tolist() == pytest.approx([0.2, 0.2])


class TestRunSimulation:
    def test_battery_with_existing_solar(self) -> None:
        # Midday surplus, evening offtake: the battery stores part of the surplus
        index = _index(96)
        midday = (index.hour >= 11) & (index.hour < 15)
        frame = pd.DataFrame(
            {
                const.ELECTRICITY_PRODUCED: [0.6 if m else 0.0 for m in midday],
                const.ELECTRICITY_EXPORTED: [0.4 if m else 0.0 for m in midday],
                const.ELECTRICITY_DELIVERED: [0.0 if m else 0.1 for m in midday],
            },
            index=index,
        )
        payload = {
            "ex_ante_data": TimeDataFrame.from_pandas(frame).model_dump(mode="json"),
            "simulation_parameters": {
                "type": "selfconsumptionbatterysimulation",
                "capacity": 5.0,
                "power": 2.0,
                "loss_factor": 0.0,
            },
            "timezone": TZ,
        }

        summary = asyncio.run(run_simulation(FullSimulationInput.model_validate(payload)))

        ex_ante = summary.ex_ante["total"]
        ex_post = summary.ex_post["total"]
        charged = abs(ex_post[const.ELECTRICITY_CHARGED])
        assert charged > 0
        # Direct use does not change with a battery; total self-consumption does
        assert ex_post[const.ELECTRICITY_SELF_CONSUMED_DIRECT] == pytest.approx(
            ex_ante[const.ELECTRICITY_SELF_CONSUMED_DIRECT]
        )
        assert ex_post[const.ELECTRICITY_SELF_CONSUMED] == pytest.approx(
            ex_ante[const.ELECTRICITY_SELF_CONSUMED] + charged
        )
