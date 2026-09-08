"""Tests for cost pre-flight checks."""

import datetime as dt

import isodate
import pytest
from energy_cost.formula import IndexAdder, IndexFormula

from openenergyid.cost import CostWarningCode, UnknownIndexError, iter_index_names
from openenergyid.cost.diagnostics import check_indexes, check_period

from .conftest import MOCK_INDEX, TZ, make_contract, make_index

MONTH = isodate.Duration(months=1)


def _at(year: int, month: int, day: int = 1) -> dt.datetime:
    return dt.datetime(year, month, day, tzinfo=TZ)


class TestIterIndexNames:
    def test_finds_a_supplier_index(self, mock_index) -> None:
        contract = make_contract(with_index=True)
        assert iter_index_names(contract) == {MOCK_INDEX}

    def test_returns_empty_for_a_constant_only_contract(self) -> None:
        contract = make_contract()
        assert iter_index_names(contract) == set()

    def test_reaches_indexes_nested_deep_in_the_shipped_fluvius_tariff(self) -> None:
        """The generic walk must survive real nesting, not just top-level formulas.

        The shipped Fluvius capacity node is ``maximum -> minimum -> by_meter_type``,
        which is why this walks the model tree rather than switching on formula kind.

        Deep-copied first: a resolved Contract holds the *shared* Tariff objects from
        the global RegionalData registry, so mutating one in place would leak into
        every other test.
        """
        contract = make_contract().model_copy(deep=True)
        capacity = contract.distributor.root[-1].capacity["total"]
        nested = capacity.maximum[1].minimum[1]
        assert (capacity.kind, capacity.maximum[1].kind, nested.kind) == (
            "maximum",
            "minimum",
            "meter_type",
        )

        nested.by_meter_type["default"] = IndexFormula(
            constant_cost=0.0,
            variable_costs=[IndexAdder(index="DeeplyNested", scalar=1.0)],
        )
        assert "DeeplyNested" in iter_index_names(contract)


class TestCheckIndexes:
    def test_raises_for_an_unregistered_index(self) -> None:
        contract = make_contract(with_index=True)
        with pytest.raises(UnknownIndexError) as excinfo:
            check_indexes(contract, _at(2024, 6))
        assert excinfo.value.args[0] == MOCK_INDEX

    def test_raises_before_any_calculation_runs(self, monkeypatch) -> None:
        """The pre-flight exists so this failure never happens mid-calculation."""
        contract = make_contract(with_index=True)

        def _explode(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("Contract.apply must not be reached")

        monkeypatch.setattr(type(contract), "apply", _explode)
        with pytest.raises(UnknownIndexError):
            check_indexes(contract, _at(2024, 6))

    def test_silent_when_the_index_covers_the_period(self) -> None:
        make_index(start="2023-01-01", end="2025-01-01")
        contract = make_contract(with_index=True)
        assert check_indexes(contract, _at(2024, 6)) == []

    def test_warns_when_the_index_stops_early(self) -> None:
        """Registered indexes forward-fill, so running past the data is silent."""
        make_index(start="2024-01-01", end="2024-07-01")
        contract = make_contract(with_index=True)

        warnings = check_indexes(contract, _at(2025, 1))

        assert [w.code for w in warnings] == [CostWarningCode.INDEX_DATA_INCOMPLETE]
        assert warnings[0].context["index"] == MOCK_INDEX
        # Last interval start (2024-07-01T00:00) plus one resolution, in the end's zone.
        covered_until = dt.datetime.fromisoformat(warnings[0].context["covered_until"])
        assert covered_until == dt.datetime(2024, 7, 1, 0, 15, tzinfo=TZ)


class TestCheckPeriod:
    def test_clean_full_year_has_no_warnings(self) -> None:
        contract = make_contract()
        assert check_period(contract, _at(2024, 1), _at(2025, 1), MONTH) == []

    def test_does_not_warn_about_boundary_alignment(self) -> None:
        """compute_cost owns that warning; it shrinks the window before this runs."""
        contract = make_contract()
        codes = [w.code for w in check_period(contract, _at(2024, 1, 5), _at(2024, 3, 12), MONTH)]
        assert CostWarningCode.PARTIAL_BILLING_PERIOD not in codes

    def test_warns_when_shorter_than_one_billing_period(self) -> None:
        contract = make_contract()
        codes = [w.code for w in check_period(contract, _at(2024, 1), _at(2024, 1, 4), MONTH)]
        assert CostWarningCode.PERIOD_SHORTER_THAN_BILLING_PERIOD in codes

    def test_warns_when_the_capacity_window_is_incomplete(self) -> None:
        """The Flemish capacity tariff averages over a 12-month rolling window."""
        contract = make_contract()
        assert contract.capacity_rule.window_periods == 12

        codes = [w.code for w in check_period(contract, _at(2024, 1), _at(2024, 4), MONTH)]
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE in codes

    def test_no_capacity_warning_for_a_full_window(self) -> None:
        contract = make_contract()
        codes = [w.code for w in check_period(contract, _at(2024, 1), _at(2025, 1), MONTH)]
        assert CostWarningCode.CAPACITY_WINDOW_INCOMPLETE not in codes
