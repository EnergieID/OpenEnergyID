"""Tests for the coverage and warnings about a simulation's ex-ante data."""

import asyncio

import numpy as np
import pandas as pd

from openenergyid import const
from openenergyid.abstractsim import SimulationWarningCode
from openenergyid.models import TimeDataFrame
from openenergyid.sim import FullSimulationInput, run_simulation
from openenergyid.sim.diagnostics import check_ex_ante

TZ = "Europe/Brussels"


def _frame(days: int = 3, exported: bool = True) -> pd.DataFrame:
    index = pd.date_range("2025-03-01", periods=96 * days, freq="15min", tz=TZ)
    data = {const.ELECTRICITY_DELIVERED: 0.2}
    if exported:
        data[const.ELECTRICITY_EXPORTED] = 0.1
    return pd.DataFrame(data, index=index)


def _codes(warnings) -> list[str]:
    return [w.code for w in warnings]


class TestCoverage:
    def test_full_data(self) -> None:
        frame = _frame()

        coverage, warnings = check_ex_ante(frame)

        assert coverage.start == frame.index[0]
        assert coverage.end == frame.index[-1]
        assert coverage.days_with_delivered == 3
        assert coverage.days_with_exported == 3
        assert coverage.intervals == 96 * 3
        assert coverage.intervals_incomplete == 0
        assert warnings == []

    def test_days_are_counted_in_the_frame_timezone(self) -> None:
        # One local day of data starts at 23:00 UTC the day before, so it spans two UTC dates
        frame = _frame(days=1)

        coverage, _ = check_ex_ante(frame)

        assert coverage.days_with_delivered == 1

    def test_a_day_without_export_is_not_counted(self) -> None:
        frame = _frame()
        frame.loc[frame.index.day == 2, const.ELECTRICITY_EXPORTED] = np.nan

        coverage, _ = check_ex_ante(frame)

        assert coverage.days_with_delivered == 3
        assert coverage.days_with_exported == 2

    def test_empty_frame(self) -> None:
        frame = pd.DataFrame(
            {const.ELECTRICITY_DELIVERED: [], const.ELECTRICITY_EXPORTED: []},
            index=pd.DatetimeIndex([], tz=TZ),
        )

        coverage, _ = check_ex_ante(frame)

        assert coverage.start is None
        assert coverage.intervals == 0


class TestWarnings:
    def test_no_export_column(self) -> None:
        coverage, warnings = check_ex_ante(_frame(exported=False))

        assert _codes(warnings) == [SimulationWarningCode.NO_INJECTION_DATA]
        assert coverage.days_with_exported == 0
        # Missing injection everywhere is reported once, not as incomplete intervals
        assert coverage.intervals_incomplete == 0

    def test_export_all_nan(self) -> None:
        frame = _frame()
        frame[const.ELECTRICITY_EXPORTED] = np.nan

        _, warnings = check_ex_ante(frame)

        assert _codes(warnings) == [SimulationWarningCode.NO_INJECTION_DATA]

    def test_gaps(self) -> None:
        frame = _frame()
        frame.iloc[:4, frame.columns.get_loc(const.ELECTRICITY_DELIVERED)] = np.nan
        frame.iloc[10:12, frame.columns.get_loc(const.ELECTRICITY_EXPORTED)] = np.nan

        coverage, warnings = check_ex_ante(frame)

        assert coverage.intervals_incomplete == 6
        assert _codes(warnings) == [SimulationWarningCode.ENERGY_DATA_INCOMPLETE]
        assert warnings[0].context == {"intervals": 96 * 3, "incomplete": 6}


class TestRunSimulation:
    @staticmethod
    def _run(frame: pd.DataFrame):
        payload = {
            "ex_ante_data": TimeDataFrame.from_pandas(frame).model_dump(mode="json"),
            "simulation_parameters": {
                "type": "selfconsumptionbatterysimulation",
                "capacity": 10.0,
                "power": 5.0,
            },
            "timezone": TZ,
        }
        return asyncio.run(run_simulation(FullSimulationInput.model_validate(payload)))

    def test_summary_carries_coverage_and_no_warnings_for_full_data(self) -> None:
        summary = self._run(_frame())

        assert summary.coverage is not None
        assert summary.coverage.days_with_delivered == 3
        assert summary.warnings == []

    def test_summary_carries_warnings_for_gaps(self) -> None:
        frame = _frame()
        frame.iloc[:4, frame.columns.get_loc(const.ELECTRICITY_EXPORTED)] = np.nan

        summary = self._run(frame)

        assert _codes(summary.warnings) == [SimulationWarningCode.ENERGY_DATA_INCOMPLETE]
        dumped = summary.model_dump(mode="json")
        assert dumped["warnings"][0]["code"] == "energy_data_incomplete"
        assert dumped["coverage"]["intervals_incomplete"] == 4
