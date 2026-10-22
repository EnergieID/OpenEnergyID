"""Tests for the self-consumption battery simulator."""

import pandas as pd
import pytest
from pydantic import ValidationError

from openenergyid import const
from openenergyid.battsim.selfconsumption import (
    SelfConsumptionBatterySimulationInput,
    SelfConsumptionBatterySimulator,
)

batt_sim = SelfConsumptionBatterySimulator.batt_sim


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2025-01-01", periods=len(values), freq="15min", tz="Europe/Brussels")
    return pd.Series(values, index=index, dtype=float)


def _simulate(offtake: list[float], injection: list[float], **kwargs) -> pd.DataFrame:
    params = {"capacity": 10.0, "initial_charge": 0.0, "loss": 0.0} | kwargs
    return batt_sim(energy_offtake=_series(offtake), energy_injection=_series(injection), **params)


class TestInput:
    def test_power_fills_both_directions(self) -> None:
        input_ = SelfConsumptionBatterySimulationInput(capacity=10, power=5)

        assert input_.charge_power == 5
        assert input_.discharge_power == 5

    def test_directional_powers_override_power(self) -> None:
        input_ = SelfConsumptionBatterySimulationInput(
            capacity=10, power=5, charge_power=4, discharge_power=3
        )

        assert input_.charge_power == 4
        assert input_.discharge_power == 3

    def test_directional_powers_without_power(self) -> None:
        input_ = SelfConsumptionBatterySimulationInput(
            capacity=10, charge_power=4, discharge_power=3
        )

        assert input_.power is None
        assert input_.charge_power == 4
        assert input_.discharge_power == 3

    @pytest.mark.parametrize("kwargs", [{}, {"charge_power": 4}, {"discharge_power": 3}])
    def test_missing_power_is_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            SelfConsumptionBatterySimulationInput(capacity=10, **kwargs)

    @pytest.mark.parametrize("field", ["power", "charge_power", "discharge_power"])
    def test_negative_power_is_rejected(self, field: str) -> None:
        kwargs = {"power": 5} | {field: -1}

        with pytest.raises(ValidationError):
            SelfConsumptionBatterySimulationInput(capacity=10, **kwargs)

    def test_simulator_from_input_uses_directional_powers(self) -> None:
        input_ = SelfConsumptionBatterySimulationInput(
            capacity=10, power=5, charge_power=4, discharge_power=3
        )
        data = pd.DataFrame(
            {
                const.ELECTRICITY_DELIVERED: _series([0.0]),
                const.ELECTRICITY_EXPORTED: _series([0.0]),
            }
        )

        simulator = SelfConsumptionBatterySimulator.from_pydantic(input_, data=data)

        assert simulator.charge_power == 4
        assert simulator.discharge_power == 3


class TestPowerLimits:
    def test_charge_is_capped_by_charge_power(self) -> None:
        result = _simulate([0.0], [5.0], charge_power=4.0, discharge_power=8.0)

        assert result[const.ELECTRICITY_CHARGED].iloc[0] == pytest.approx(1.0)

    def test_discharge_is_capped_by_discharge_power(self) -> None:
        result = _simulate([5.0], [0.0], charge_power=8.0, discharge_power=2.0, initial_charge=10.0)

        assert result[const.ELECTRICITY_DISCHARGED].iloc[0] == pytest.approx(0.5)

    def test_charging_gets_the_time_left_after_discharging(self) -> None:
        # Discharging 0.25 kWh at 2 kW takes half the interval, leaving half of it to
        # charge at 4 kW: 0.5 kWh.
        result = _simulate([0.25], [5.0], charge_power=4.0, discharge_power=2.0, initial_charge=5.0)

        assert result[const.ELECTRICITY_DISCHARGED].iloc[0] == pytest.approx(0.25)
        assert result[const.ELECTRICITY_CHARGED].iloc[0] == pytest.approx(0.5)

    def test_zero_discharge_power_still_charges(self) -> None:
        result = _simulate([1.0], [1.0], charge_power=4.0, discharge_power=0.0, initial_charge=5.0)

        assert result[const.ELECTRICITY_DISCHARGED].iloc[0] == 0.0
        assert result[const.ELECTRICITY_CHARGED].iloc[0] == pytest.approx(1.0)

    def test_equal_powers_match_a_single_power(self) -> None:
        offtake = [0.0, 0.3, 1.2, 0.1, 0.0, 2.0]
        injection = [1.5, 0.4, 0.0, 0.9, 2.5, 0.2]
        data = pd.DataFrame(
            {
                const.ELECTRICITY_DELIVERED: _series(offtake),
                const.ELECTRICITY_EXPORTED: _series(injection),
            }
        )

        single = SelfConsumptionBatterySimulator(
            capacity=2.0, power=3.0, loss_factor=0.05, data=data
        ).simulate()
        split = SelfConsumptionBatterySimulator(
            capacity=2.0, charge_power=3.0, discharge_power=3.0, loss_factor=0.05, data=data
        ).simulate()

        pd.testing.assert_frame_equal(single, split)


class TestLossFactor:
    @pytest.mark.parametrize("loss", [0.0, 0.026, 0.05, 0.081])
    def test_round_trip_efficiency(self, loss: float) -> None:
        # Charge 1 kWh from injection, then discharge everything that was stored.
        result = _simulate(
            [0.0, 5.0], [1.0, 0.0], charge_power=10.0, discharge_power=10.0, loss=loss
        )

        charged = result[const.ELECTRICITY_CHARGED].iloc[0]
        discharged = result[const.ELECTRICITY_DISCHARGED].iloc[1]
        assert charged == pytest.approx(1.0)
        assert discharged / charged == pytest.approx((1 - loss) / (1 + loss))
