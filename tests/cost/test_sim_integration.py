"""Tests for cost calculation wired into the simulation workflow."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from openenergyid import const
from openenergyid.sim import FullSimulationInput, run_simulation, simulate_frames

from .conftest import make_contract, make_frame

BATTERY = {
    "type": "selfconsumptionbatterysimulation",
    "capacity": 10.0,
    "power": 5.0,
}


def _payload(frame=None, **overrides) -> dict:
    from openenergyid.models import TimeDataFrame

    # Export must be non-zero, otherwise a self-consumption battery has no surplus to
    # store and the simulation is a no-op.
    frame = make_frame(periods=96 * 366, delivered=0.4, exported=0.1) if frame is None else frame
    payload = {
        "ex_ante_data": TimeDataFrame.from_pandas(frame).model_dump(mode="json"),
        "simulation_parameters": BATTERY,
        "timezone": "Europe/Brussels",
    }
    payload.update(overrides)
    return payload


def _cost_settings(**overrides) -> dict:
    settings = {
        "contract": make_contract().model_dump(mode="json"),
        "output_resolution": "P1M",
    }
    settings.update(overrides)
    return settings


class TestSimulateFrames:
    def test_returns_a_frame_per_stage(self) -> None:
        payload = _payload(simulation_parameters=[BATTERY, BATTERY])
        frames = asyncio.run(simulate_frames(FullSimulationInput.model_validate(payload)))

        assert len(frames.stages) == 2
        assert len(frames.stage_results) == 2

    def test_ex_post_is_the_last_stage(self) -> None:
        frames = asyncio.run(simulate_frames(FullSimulationInput.model_validate(_payload())))

        assert frames.ex_post.equals(frames.stages[-1])

    def test_ex_ante_is_not_mutated_by_the_simulation(self) -> None:
        frames = asyncio.run(simulate_frames(FullSimulationInput.model_validate(_payload())))

        # The battery changes grid offtake, so the two must differ.
        assert not frames.ex_ante.equals(frames.ex_post)


class TestCostConflictValidation:
    def test_rejects_a_contract_alongside_price_columns(self) -> None:
        frame = make_frame(periods=96)
        frame[const.PRICE_ELECTRICITY_DELIVERED] = 0.30

        with pytest.raises(ValidationError, match="Conflicting cost inputs"):
            FullSimulationInput.model_validate(_payload(frame=frame, cost=_cost_settings()))

    def test_allows_price_columns_without_a_contract(self) -> None:
        frame = make_frame(periods=96)
        frame[const.PRICE_ELECTRICITY_DELIVERED] = 0.30

        model = FullSimulationInput.model_validate(_payload(frame=frame))
        assert model.cost is None

    def test_allows_a_contract_without_price_columns(self) -> None:
        model = FullSimulationInput.model_validate(_payload(cost=_cost_settings()))
        assert model.cost is not None


class TestRunSimulationWithCost:
    def test_cost_is_absent_without_a_contract(self) -> None:
        summary = asyncio.run(run_simulation(FullSimulationInput.model_validate(_payload())))
        assert summary.cost is None

    def test_battery_reduces_the_bill(self, mock_index) -> None:
        payload = _payload(
            cost=_cost_settings(contract=make_contract(with_index=True).model_dump(mode="json")),
        )
        summary = asyncio.run(run_simulation(FullSimulationInput.model_validate(payload)))

        assert summary.cost is not None
        assert summary.cost.comparison.diff["total"] < 0

    def test_battery_shaves_the_capacity_tariff(self, mock_index) -> None:
        """Peak shaving is the structural saving a battery is sold on."""
        payload = _payload(cost=_cost_settings())
        summary = asyncio.run(run_simulation(FullSimulationInput.model_validate(payload)))

        capacity = summary.cost.comparison.diff["distributor.capacity.total"]
        assert capacity <= 0

    def test_zero_capacity_battery_saves_exactly_nothing(self) -> None:
        payload = _payload(
            simulation_parameters={**BATTERY, "capacity": 0.0, "power": 0.0},
            cost=_cost_settings(),
        )
        summary = asyncio.run(run_simulation(FullSimulationInput.model_validate(payload)))

        assert summary.cost.comparison.diff["total"] == pytest.approx(0.0)

    def test_per_stage_costs_each_cumulative_frame(self) -> None:
        payload = _payload(
            simulation_parameters=[BATTERY, BATTERY],
            cost=_cost_settings(per_stage=True),
        )
        summary = asyncio.run(run_simulation(FullSimulationInput.model_validate(payload)))

        assert summary.cost.stages is not None
        assert len(summary.cost.stages) == 2
        assert len(summary.cost.stage_comparisons) == 2
        # The final stage is the ex-post state.
        assert summary.cost.stages[-1].total == pytest.approx(summary.cost.ex_post.total)

    def test_summary_survives_json_serialisation(self) -> None:
        payload = _payload(cost=_cost_settings())
        summary = asyncio.run(run_simulation(FullSimulationInput.model_validate(payload)))

        encoded = json.dumps(summary.model_dump(mode="json"))
        assert "Infinity" not in encoded and "NaN" not in encoded


class TestSchemas:
    def test_full_simulation_input_schema_generates(self) -> None:
        """Contract nests a Formula union behind a callable Discriminator."""
        schema = FullSimulationInput.model_json_schema()
        assert "cost" in schema["properties"]
        assert "Contract" in schema["$defs"]

    def test_tariff_is_an_array_in_the_schema(self) -> None:
        """Tariff is a RootModel[list[...]], so its JSON body is a bare array."""
        schema = FullSimulationInput.model_json_schema()
        assert schema["$defs"]["Tariff"]["type"] == "array"

    def test_simulation_summary_schema_generates(self) -> None:
        from openenergyid.abstractsim import SimulationSummary

        schema = SimulationSummary.model_json_schema()
        assert "cost" in schema["properties"]
