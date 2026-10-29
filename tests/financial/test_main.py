"""Tests for payback, NPV and IRR on yearly cash flows."""

import datetime as dt

import pytest

from openenergyid.financial import (
    FinancialSettings,
    FinancialWarningCode,
    annualisation_factor,
    check_period,
    discount,
    irr,
    npv,
    payback_years,
    summarise,
)

# Worked example: 6,000 EUR invested, 600 EUR saved every year, at 3 %.
INVESTMENT = 6000.0
SAVING = 600.0
RATE = 0.03


class TestPayback:
    def test_without_degradation_is_investment_over_saving(self) -> None:
        assert payback_years(INVESTMENT, [SAVING] * 25) == pytest.approx(10.0)

    def test_is_interpolated_within_the_year(self) -> None:
        assert payback_years(1000.0, [400.0, 400.0, 400.0]) == pytest.approx(2.5)

    def test_is_none_when_the_horizon_ends_first(self) -> None:
        assert payback_years(INVESTMENT, [SAVING] * 9) is None

    def test_is_none_without_savings(self) -> None:
        assert payback_years(INVESTMENT, [0.0] * 25) is None

    def test_is_zero_without_investment(self) -> None:
        assert payback_years(0.0, [SAVING] * 5) == 0.0

    def test_waits_for_a_negative_year_to_be_made_up(self) -> None:
        assert payback_years(100.0, [-50.0, 100.0, 100.0]) == pytest.approx(2.5)

    def test_discounted_matches_the_worked_example(self) -> None:
        discounted = discount([SAVING] * 25, RATE)

        assert discounted[0] == pytest.approx(582.52, abs=0.01)
        assert discounted[9] == pytest.approx(446.46, abs=0.01)
        assert sum(discounted[:12]) == pytest.approx(5972.40, abs=0.01)
        assert payback_years(INVESTMENT, discounted) == pytest.approx(12.07, abs=0.01)


class TestNpv:
    def test_matches_the_annuity_formula(self) -> None:
        annuity = (1 - (1 + RATE) ** -25) / RATE

        assert npv(INVESTMENT, [SAVING] * 25, RATE) == pytest.approx(SAVING * annuity - INVESTMENT)
        assert npv(INVESTMENT, [SAVING] * 25, RATE) == pytest.approx(4447.8, abs=0.1)

    def test_at_zero_rate_is_the_plain_sum(self) -> None:
        assert npv(INVESTMENT, [SAVING] * 25, 0.0) == pytest.approx(9000.0)


class TestIrr:
    def test_zeroes_the_npv(self) -> None:
        rate = irr(INVESTMENT, [SAVING] * 25)

        assert rate == pytest.approx(0.0878, abs=0.0001)
        assert npv(INVESTMENT, [SAVING] * 25, rate) == pytest.approx(0.0, abs=1e-6)

    def test_is_zero_when_the_savings_just_repay_the_investment(self) -> None:
        assert irr(INVESTMENT, [SAVING] * 10) == pytest.approx(0.0, abs=1e-8)

    def test_is_negative_when_the_investment_is_not_repaid(self) -> None:
        assert irr(INVESTMENT, [SAVING] * 5) < 0

    def test_is_none_without_savings(self) -> None:
        assert irr(INVESTMENT, [0.0] * 25) is None

    def test_is_none_without_investment(self) -> None:
        assert irr(0.0, [SAVING] * 25) is None


class TestAnnualisation:
    def test_a_year_is_one(self) -> None:
        start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
        end = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)

        assert annualisation_factor(start, end) == pytest.approx(365.25 / 365)
        assert check_period(start, end) == []

    def test_a_month_is_scaled_up_and_warned_about(self) -> None:
        start = dt.datetime(2023, 3, 1, tzinfo=dt.UTC)
        end = dt.datetime(2023, 4, 1, tzinfo=dt.UTC)

        assert annualisation_factor(start, end) == pytest.approx(365.25 / 31)
        [warning] = check_period(start, end)
        assert warning.code == FinancialWarningCode.SHORT_PERIOD_ANNUALISED
        assert warning.context["days"] == pytest.approx(31)

    def test_an_empty_period_is_refused(self) -> None:
        moment = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)

        with pytest.raises(ValueError):
            annualisation_factor(moment, moment)


class TestSummarise:
    def test_worked_example(self) -> None:
        settings = FinancialSettings(investment=INVESTMENT, discount_rate=RATE)

        summary = summarise(settings, [1000.0] * 25, [400.0] * 25)

        assert len(summary.yearly) == 25
        assert summary.yearly[0].savings == pytest.approx(SAVING)
        assert summary.yearly[9].cumulative == pytest.approx(6000.0)
        assert summary.yearly[11].discounted_cumulative == pytest.approx(5972.40, abs=0.01)
        assert summary.payback_years == pytest.approx(10.0)
        assert summary.discounted_payback_years == pytest.approx(12.07, abs=0.01)
        assert summary.npv == pytest.approx(4447.8, abs=0.1)
        assert summary.irr == pytest.approx(0.0878, abs=0.0001)

    def test_maintenance_lowers_the_cash_flow(self) -> None:
        settings = FinancialSettings(investment=INVESTMENT, yearly_maintenance=100.0)

        summary = summarise(settings, [1000.0] * 25, [400.0] * 25)

        assert summary.yearly[0].savings == pytest.approx(600.0)
        assert summary.yearly[0].cash_flow == pytest.approx(500.0)
        assert summary.payback_years == pytest.approx(12.0)

    def test_without_discount_rate_leaves_out_the_discounted_figures(self) -> None:
        settings = FinancialSettings(investment=INVESTMENT)

        summary = summarise(settings, [1000.0] * 25, [400.0] * 25)

        assert summary.discounted_payback_years is None
        assert summary.npv is None
        assert summary.yearly[0].discounted_cash_flow is None
        assert summary.yearly[0].discounted_cumulative is None
        assert summary.irr is not None

    @pytest.mark.parametrize(
        "field, value",
        [
            ("investment", -1),
            ("yearly_maintenance", -1),
            ("horizon_years", 0),
            ("horizon_years", 51),
        ],
    )
    def test_settings_reject_out_of_range_values(self, field, value) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FinancialSettings.model_validate({"investment": 1000, field: value})
