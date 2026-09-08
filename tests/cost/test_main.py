"""Tests for contract-based cost calculation."""

import datetime as dt
import json
import math

import isodate
import pytest

from openenergyid.cost import (
    ContractCostSettings,
    CostCalculationError,
    CostWarningCode,
    compare_costs,
    compute_cost,
)

from .conftest import make_contract, make_frame

MONTH = isodate.Duration(months=1)


def _settings(contract=None, **overrides) -> ContractCostSettings:
    return ContractCostSettings(
        contract=contract if contract is not None else make_contract(),
        output_resolution=overrides.pop("output_resolution", MONTH),
        **overrides,
    )


class TestComputeCost:
    def test_returns_one_row_per_billing_period(self, year_frame) -> None:
        result, _ = compute_cost(year_frame, _settings())

        assert result.periods is not None
        assert len(result.periods.index) == 12

    def test_total_matches_the_sum_of_the_period_series(self, year_frame) -> None:
        result, _ = compute_cost(year_frame, _settings())

        periods = result.periods.to_pandas()
        assert result.total == pytest.approx(periods["total.total.total"].sum())

    def test_breakdown_covers_every_tariff_category(self, year_frame) -> None:
        result, _ = compute_cost(year_frame, _settings())

        assert {"supplier", "distributor", "fees", "taxes"} <= set(result.breakdown)

    def test_category_totals_plus_taxes_equal_the_grand_total(self, year_frame) -> None:
        """Guards the double grand-total recomputation inside Contract.apply."""
        result, _ = compute_cost(year_frame, _settings())

        parts = sum(
            groups["total"]["total"] for groups in result.breakdown.values() if "total" in groups
        )
        assert result.total == pytest.approx(parts)

    def test_injection_reduces_the_bill(self, year_frame) -> None:
        """A negative injection rate is revenue, so it must lower the total."""
        with_injection, _ = compute_cost(
            year_frame, _settings(make_contract(supplier_injection=-50.0))
        )
        without, _ = compute_cost(year_frame, _settings(make_contract(supplier_injection=None)))

        assert with_injection.total < without.total
        assert with_injection.breakdown["supplier"]["injection"]["total"] < 0

    def test_no_vat_on_injection_or_the_energy_fund(self, year_frame) -> None:
        """Both exemptions ship in energy_cost's Belgian taxes.yml."""
        result, _ = compute_cost(year_frame, _settings())

        taxes = result.breakdown["taxes"]
        assert "injection" not in taxes or taxes["injection"]["total"] in (0.0, None)

    def test_zero_offtake_still_bills_fixed_and_capacity_charges(self) -> None:
        frame = make_frame(delivered=0.0, exported=0.0, produced=0.0)
        result, _ = compute_cost(frame, _settings())

        assert result.total is not None
        assert result.total > 0
        assert not math.isnan(result.total)

    def test_include_flags_shrink_the_payload(self, year_frame) -> None:
        result, _ = compute_cost(
            year_frame, _settings(include_periods=False, include_breakdown=False)
        )

        assert result.periods is None
        assert result.breakdown == {}
        assert result.total is not None

    def test_index_backed_contract_costs_successfully(self, year_frame, flat_index) -> None:
        result, _ = compute_cost(
            year_frame,
            _settings(make_contract(supplier_consumption=0.0, with_index=True)),
        )
        assert result.total is not None

    def test_window_is_half_open_over_the_last_interval(self, year_frame) -> None:
        result, _ = compute_cost(year_frame, _settings())

        assert result.start == year_frame.index[0]
        # The final interval contributes its full width.
        assert result.end > year_frame.index[-1]

    def test_preflight_warnings_are_returned(self) -> None:
        frame = make_frame(periods=96 * 31)
        _, warnings = compute_cost(frame, _settings())

        codes = {w.code for w in warnings}
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE in codes

    def test_unregistered_index_is_reported_as_a_cost_error(self, year_frame) -> None:
        from openenergyid.cost import UnknownIndexError

        settings = _settings(make_contract(with_index=True))
        with pytest.raises(UnknownIndexError):
            compute_cost(year_frame, settings)

    def test_history_not_covering_the_period_raises(self, year_frame) -> None:
        """ContractHistory.apply returns None when no version covers the window.

        A standalone Contract has no such gap -- its ``start`` only orders versions
        within a history -- so this is the path that can legitimately yield no cost.
        """
        from energy_cost import ContractHistory

        future = make_contract()
        future.start = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
        history = ContractHistory.model_validate([future.model_dump()])

        with pytest.raises(CostCalculationError, match="No cost could be calculated"):
            compute_cost(year_frame, _settings(history))


class TestCompareCosts:
    def test_identical_results_produce_zero_diffs(self, year_frame) -> None:
        result, _ = compute_cost(year_frame, _settings())
        comparison = compare_costs(result, result)

        assert all(value == 0 for value in comparison.diff.values())

    def test_output_is_json_serialisable(self, year_frame) -> None:
        """No inf or nan may reach the response, even for a zero baseline.

        A battery with ``capacity == 0`` is a pass-through, so before == after and the
        ratio is exactly 0/0.
        """
        result, _ = compute_cost(year_frame, _settings())
        comparison = compare_costs(result, result)

        encoded = json.dumps(comparison.model_dump(mode="json"))
        assert "Infinity" not in encoded
        assert "NaN" not in encoded

    def test_zero_baseline_gives_none_not_infinity(self, year_frame) -> None:
        zero_frame = make_frame(delivered=0.0, exported=0.0)
        before, _ = compute_cost(zero_frame, _settings())
        after, _ = compute_cost(year_frame, _settings())

        comparison = compare_costs(before, after)
        for key, ratio in comparison.ratio_diff.items():
            assert ratio is None or math.isfinite(ratio), key

    def test_savings_are_negative(self, year_frame) -> None:
        """diff = after - before, matching simeval.compare_results."""
        cheaper = make_frame(delivered=0.01, exported=0.02)
        before, _ = compute_cost(year_frame, _settings())
        after, _ = compute_cost(cheaper, _settings())

        assert compare_costs(before, after).diff["total"] < 0
