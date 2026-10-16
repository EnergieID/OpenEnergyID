"""Measured month peaks as capacity history for the contract cost (AB#861)."""

import datetime as dt
import types

import isodate
import pandas as pd
import pytest
from energy_cost import ContractHistory, Meter, TimeseriesFrame
from pydantic import ValidationError

from openenergyid.cost import (
    ContractCostSettings,
    CostWarningCode,
    MonthPeak,
    MonthPeaks,
    compute_cost,
    cost_simulation,
    frame_to_meters,
)
from openenergyid.cost.capacity import build_capacity, first_billed_month

from .conftest import TZ, make_contract, make_frame

MONTH = isodate.Duration(months=1)
# A year starting mid-month, like a "last 12 months" request: the whole billing periods
# are Feb 2024 .. Dec 2024, so Feb 2023 .. Jan 2024 is the 12-month window's history.
START = "2024-01-10"
PERIODS = 96 * 366
CUTOFF = pd.Timestamp("2024-02-01", tz=TZ)
W_PER_MW = 1_000_000.0


def frame(delivered: float = 1.0) -> pd.DataFrame:
    """Flat 1 kWh per quarter hour is a flat 4 kW, above the tariff's minimum."""
    return make_frame(start=START, periods=PERIODS, delivered=delivered)


def history(first: str = "2023-01-01", last: str = "2024-01-01") -> list[MonthPeak]:
    """One distinct peak (W) per month, timestamped inside its month."""
    months = pd.date_range(first, last, freq="MS", tz=TZ)
    return [
        MonthPeak(
            timestamp=(m + pd.Timedelta(days=16, hours=18, minutes=15)).to_pydatetime(),
            value=3000.0 + 250.0 * i,
        )
        for i, m in enumerate(months)
    ]


def peaks(values: list[MonthPeak]) -> MonthPeaks:
    return MonthPeaks(values=values)


def settings(month_peaks=None, contract=None, **overrides) -> ContractCostSettings:
    return ContractCostSettings(
        contract=contract if contract is not None else make_contract(),
        output_resolution=MONTH,
        month_peaks=month_peaks,
        **overrides,
    )


def codes(warnings) -> set[CostWarningCode]:
    return {w.code for w in warnings}


def capacity_cost(result) -> float:
    return result.breakdown["distributor"]["capacity"]["total"]


def reference_capacity_cost(data: pd.DataFrame, month_peaks_w: dict[str, float]) -> float:
    """Capacity cost from a capacity series built by hand and handed to energy_cost.

    ``month_peaks_w`` maps month start -> W for every month of the series. This goes
    around ``build_capacity`` entirely: it is the expectation the implementation is
    held against.
    """
    contract = make_contract()
    consumption, injection, _ = frame_to_meters(data)
    series = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(m, tz=TZ) for m in month_peaks_w],
            "value": [w / W_PER_MW for w in month_peaks_w.values()],
        }
    )
    consumption = Meter(
        direction=consumption.direction,
        type=consumption.type,
        measurements=consumption.measurements,
        capacity=TimeseriesFrame(series, resolution=MONTH),
    )
    out = contract.apply(
        consumption=consumption,
        injection=injection,
        start=dt.datetime(2024, 2, 1, tzinfo=TZ),
        end=dt.datetime(2025, 1, 1, tzinfo=TZ),
        output_resolution=MONTH,
    )
    return float(out[("distributor", "capacity", "total")].sum())


def hist_dict() -> dict[str, float]:
    return {
        pd.Timestamp(p.timestamp).tz_convert(TZ).strftime("%Y-%m-01"): p.value for p in history()
    }


class TestMonthPeaksModel:
    def test_accepts_the_energyid_api_payload(self) -> None:
        payload = {
            "resolution": "P1M",
            "values": [
                {"timestamp": "2023-02-17T18:15:00+01:00", "value": 4210.0},
                {"timestamp": "2023-07-03T20:00:00+02:00", "value": 3900.5},
                {"timestamp": "2023-08-03T20:00:00Z", "value": 0},
            ],
        }
        parsed = MonthPeaks.model_validate(payload)
        assert [v.value for v in parsed.values] == [4210.0, 3900.5, 0]

    def test_resolution_defaults_to_p1m(self) -> None:
        assert MonthPeaks(values=[]).resolution == "P1M"

    def test_rejects_other_resolutions(self) -> None:
        with pytest.raises(ValidationError):
            MonthPeaks.model_validate({"resolution": "PT15M", "values": []})

    def test_rejects_a_naive_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            MonthPeak.model_validate({"timestamp": "2023-02-17T18:15:00", "value": 1.0})

    def test_rejects_a_negative_value(self) -> None:
        with pytest.raises(ValidationError):
            MonthPeak.model_validate({"timestamp": "2023-02-17T18:15:00+01:00", "value": -1.0})

    def test_rejects_two_peaks_in_one_month(self) -> None:
        values = [
            {"timestamp": "2023-02-17T18:15:00+01:00", "value": 1.0},
            {"timestamp": "2023-02-18T18:15:00+01:00", "value": 2.0},
        ]
        with pytest.raises(ValidationError, match="2023-02"):
            MonthPeaks.model_validate({"values": values})

    def test_the_settings_field_is_optional_and_documented(self) -> None:
        assert settings().month_peaks is None
        text = ContractCostSettings.model_json_schema()["properties"]["month_peaks"]
        description = str(text)
        assert "W" in description and "P1M" in description and "11" in description


class TestBuildCapacity:
    def build(self, data, values, cutoff=CUTOFF):
        contract = make_contract()
        consumption, _, _ = frame_to_meters(data)
        return build_capacity(peaks(values), consumption, contract.capacity_rule, cutoff, str(TZ))

    def test_history_is_converted_from_w_to_mw(self) -> None:
        series = self.build(frame(), history())
        assert series.iloc[0]["value"] == pytest.approx(3000.0 / W_PER_MW)

    def test_months_from_the_cutoff_on_are_the_frames_own_peaks(self) -> None:
        series = self.build(frame(), history())
        in_period = series[(series["timestamp"] >= CUTOFF) & (series["timestamp"] < "2025-01-01")]
        assert len(in_period) == 11  # Feb .. Dec 2024
        assert in_period["value"].to_numpy() == pytest.approx(4.0 / 1000)  # 4 kW in MW

    def test_provided_in_period_peaks_are_dropped(self) -> None:
        absurd = MonthPeak(timestamp=dt.datetime(2024, 5, 15, 12, tzinfo=TZ), value=99_000.0)
        series = self.build(frame(), [*history(), absurd])
        assert series["value"].max() < 0.01

    def test_the_history_before_the_cutoff_is_the_same_for_any_frame(self) -> None:
        ante = self.build(frame(1.0), history())
        post = self.build(frame(0.5), history())
        before = lambda s: s[s["timestamp"] < CUTOFF].reset_index(drop=True)  # noqa: E731
        pd.testing.assert_frame_equal(before(ante), before(post))
        assert (post["value"] <= ante["value"]).all()

    def test_a_missing_month_inside_the_window_means_no_history(self) -> None:
        values = history()
        del values[8]
        assert self.build(frame(), values) is None

    def test_too_few_prior_months_means_no_history(self) -> None:
        assert self.build(frame(), history(first="2023-04-01")) is None

    def test_exactly_the_eleven_prior_months_are_enough(self) -> None:
        assert self.build(frame(), history(first="2023-03-01", last="2024-01-01")) is not None


def test_a_naive_first_billed_datetime_is_read_in_the_contract_timezone() -> None:
    assert first_billed_month(dt.datetime(2024, 2, 17, 5), str(TZ)) == CUTOFF


class TestComputeCostWithHistory:
    def test_without_month_peaks_nothing_changes(self) -> None:
        data = frame()
        plain, plain_warnings = compute_cost(data, settings())
        none, none_warnings = compute_cost(data, settings(month_peaks=None))

        assert plain.model_dump() == none.model_dump()
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE in codes(plain_warnings)
        assert [w.model_dump() for w in plain_warnings] == [w.model_dump() for w in none_warnings]

    def test_full_history_costs_the_capacity_tariff_over_the_full_window(self) -> None:
        data = frame()
        result, warnings = compute_cost(data, settings(peaks(history())))

        series = {**hist_dict(), **{f"2024-{m:02d}-01": 4000.0 for m in range(2, 13)}}
        assert capacity_cost(result) == pytest.approx(reference_capacity_cost(data, series))

    def test_full_history_changes_the_capacity_cost(self) -> None:
        data = frame()
        without, _ = compute_cost(data, settings())
        with_history, _ = compute_cost(data, settings(peaks(history())))
        assert capacity_cost(with_history) != pytest.approx(capacity_cost(without))

    def test_the_window_warning_is_not_emitted_with_full_history(self) -> None:
        _, warnings = compute_cost(frame(), settings(peaks(history())))
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE not in codes(warnings)

    def test_other_warnings_survive(self) -> None:
        _, plain = compute_cost(frame(), settings())
        _, filled = compute_cost(frame(), settings(peaks(history())))
        assert codes(plain) - {CostWarningCode.CAPACITY_WINDOW_INCOMPLETE} == codes(filled)
        assert CostWarningCode.PARTIAL_BILLING_PERIOD in codes(filled)

    def test_incomplete_history_is_identical_to_no_history_and_still_warns(self) -> None:
        data = frame()
        values = history()
        del values[8]
        plain, plain_warnings = compute_cost(data, settings())
        short, short_warnings = compute_cost(data, settings(peaks(values)))

        assert short.model_dump() == plain.model_dump()
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE in codes(short_warnings)
        assert [w.model_dump() for w in short_warnings] == [w.model_dump() for w in plain_warnings]

    def test_provided_in_period_peaks_are_never_used(self) -> None:
        data = frame()
        absurd = [
            MonthPeak(timestamp=dt.datetime(2024, m, 15, 19, tzinfo=TZ), value=99_000.0)
            for m in range(2, 13)
        ]
        clean, _ = compute_cost(data, settings(peaks(history())))
        noisy, _ = compute_cost(data, settings(peaks([*history(), *absurd])))
        assert noisy.model_dump() == clean.model_dump()

    def test_a_lower_frame_gets_lower_in_period_peaks_on_the_same_history(self) -> None:
        low = frame(delivered=0.75)  # 3 kW, still above the minimum
        result, _ = compute_cost(low, settings(peaks(history())))

        series = {**hist_dict(), **{f"2024-{m:02d}-01": 3000.0 for m in range(2, 13)}}
        assert capacity_cost(result) == pytest.approx(reference_capacity_cost(low, series))

    def test_a_contract_history_works_too(self) -> None:
        data = frame()
        as_history = ContractHistory.model_validate([make_contract().model_dump()])
        single, _ = compute_cost(data, settings(peaks(history())))
        multi, warnings = compute_cost(data, settings(peaks(history()), contract=as_history))

        assert capacity_cost(multi) == pytest.approx(capacity_cost(single))
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE not in codes(warnings)

    def test_data_starting_on_a_month_boundary_needs_the_months_before_it(self) -> None:
        """Whole periods then start in January 2024, so Jan 2023 .. Dec 2023 is history."""
        data = make_frame(start="2024-01-01", periods=96 * 366)
        full = history(first="2023-02-01", last="2023-12-01")
        _, warnings = compute_cost(data, settings(peaks(full)))
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE not in codes(warnings)


class TestCostSimulationWithHistory:
    def frames(self) -> types.SimpleNamespace:
        ante, post = frame(1.0), frame(0.75)
        return types.SimpleNamespace(ex_ante=ante, ex_post=post, stages=[post])

    def test_ex_ante_and_ex_post_share_the_history_and_recalculate_their_own_months(self) -> None:
        frames = self.frames()
        summary = cost_simulation(frames, settings(peaks(history())))

        for result, level in ((summary.ex_ante, 4000.0), (summary.ex_post, 3000.0)):
            series = {**hist_dict(), **{f"2024-{m:02d}-01": level for m in range(2, 13)}}
            data = frames.ex_ante if level == 4000.0 else frames.ex_post
            assert capacity_cost(result) == pytest.approx(reference_capacity_cost(data, series))
        assert summary.comparison.diff["distributor.capacity.total"] < 0

    def test_the_summary_has_no_window_warning_with_full_history(self) -> None:
        summary = cost_simulation(self.frames(), settings(peaks(history())))
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE not in codes(summary.warnings)

    def test_the_summary_keeps_the_window_warning_without_history(self) -> None:
        summary = cost_simulation(self.frames(), settings())
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE in codes(summary.warnings)

    def test_per_stage_costing_uses_the_history_as_well(self) -> None:
        frames = self.frames()
        summary = cost_simulation(frames, settings(peaks(history()), per_stage=True))

        assert summary.stages is not None
        assert capacity_cost(summary.stages[0]) == pytest.approx(capacity_cost(summary.ex_post))
        series = {**hist_dict(), **{f"2024-{m:02d}-01": 3000.0 for m in range(2, 13)}}
        assert capacity_cost(summary.ex_post) == pytest.approx(
            reference_capacity_cost(frames.ex_post, series)
        )


def test_the_window_stays_incomplete_for_a_short_period_even_with_unrelated_peaks() -> None:
    """History far from the period cannot fill the window."""
    data = make_frame(start="2024-01-01", periods=96 * 31)
    _, warnings = compute_cost(
        data, settings(peaks(history(first="2020-01-01", last="2020-12-01")))
    )
    assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE in codes(warnings)
