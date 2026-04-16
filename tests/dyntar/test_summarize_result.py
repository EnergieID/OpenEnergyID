import warnings

import pandas as pd
import pytest

from openenergyid.dyntar import summarize_result


def test_summarize_result_returns_none_ratio_when_abs_smr2_is_zero() -> None:
    df = pd.DataFrame(
        {
            "cost_electricity_delivered_smr2": [1.5, -1.5],
            "cost_electricity_exported_smr2": [-2.0, 2.0],
            "cost_electricity_delivered_smr3": [4.0, 1.0],
            "cost_electricity_exported_smr3": [-1.0, -2.0],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = summarize_result(df)

    assert result["cost_electricity_total_smr2"] == pytest.approx(0.0)
    assert result["cost_electricity_total_smr3"] == pytest.approx(2.0)
    assert result["ratio"] is None
    assert not any("divide by zero" in str(warning.message) for warning in caught)


def test_summarize_result_keeps_numeric_ratio_when_abs_smr2_is_nonzero() -> None:
    df = pd.DataFrame(
        {
            "cost_electricity_delivered_smr2": [3.0, 1.0],
            "cost_electricity_exported_smr2": [-1.0, 0.0],
            "cost_electricity_delivered_smr3": [4.0, 2.0],
            "cost_electricity_exported_smr3": [-1.0, -1.0],
        }
    )

    result = summarize_result(df)

    assert result["cost_electricity_total_smr2"] == pytest.approx(3.0)
    assert result["cost_electricity_total_smr3"] == pytest.approx(4.0)
    assert result["ratio"] == pytest.approx(0.2)
