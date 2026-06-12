"""Tests for MVLR source-data filtering."""

import math

import pandas as pd

from openenergyid.models import TimeDataFrame
from openenergyid.mvlr import (
    MultiVariableRegressionInput,
    clean_regression_frame,
    find_best_mvlr,
)

DEPENDENT = "energyProduction/solarPhotovoltaic"
SOLAR_REFERENCE = "solarPowerGeneration"


def _solar_regression_input(
    *,
    zero_slice: slice = slice(20, 30),
    spikes: dict[int, float] | None = None,
) -> MultiVariableRegressionInput:
    index = pd.date_range("2025-04-01", periods=90, freq="D", tz="Europe/Brussels")
    solar_reference = pd.Series(
        [1.0 + (i % 35) / 8.0 for i in range(len(index))],
        index=index,
        dtype=float,
    )
    production = 4.35 * solar_reference + 0.15

    if zero_slice:
        production.iloc[zero_slice] = 0.0
    for idx, value in (spikes or {45: 70.0, 60: 75.0, 75: 80.0}).items():
        production.iloc[idx] = value

    frame = pd.DataFrame(
        {
            DEPENDENT: production,
            SOLAR_REFERENCE: solar_reference,
        },
        index=index,
    )

    return MultiVariableRegressionInput.model_validate(
        {
            "timeZone": "Europe/Brussels",
            "independentVariables": [
                {
                    "name": SOLAR_REFERENCE,
                    "allowNegativeCoefficient": False,
                },
            ],
            "dependentVariable": DEPENDENT,
            "frame": TimeDataFrame.from_pandas(frame).model_dump(),
            "granularities": ["P1D"],
            "allowNegativePredictions": False,
            "validationParameters": {
                "rsquared": 0.95,
                "f_pvalue": 0.05,
                "pvalues": 0.05,
            },
        },
    )


def test_clean_regression_frame_removes_solar_source_outliers() -> None:
    """Solar production cleaning should drop zero-line and ratio outliers."""
    data = _solar_regression_input()
    frame = data.data_frame()

    cleaned, diagnostics = clean_regression_frame(frame, DEPENDENT)

    assert diagnostics.applied
    assert diagnostics.original_observation_count == 90
    assert diagnostics.removed_zero_with_solar_count == 10
    assert diagnostics.removed_ratio_outlier_count == 3
    assert diagnostics.removed_observation_count == 13
    assert len(cleaned) == 77
    assert (cleaned[DEPENDENT] > 0).all()


def test_find_best_mvlr_returns_filtering_diagnostics() -> None:
    """A model should fit after bad source observations are excluded."""
    data = _solar_regression_input()

    result = find_best_mvlr(data)

    assert result.r2 > 0.99
    assert result.outlier_filtering is not None
    assert result.outlier_filtering.applied
    assert result.outlier_filtering.removed_observation_count == 13
    assert math.isclose(result.independent_variables[0].coef, 4.35, rel_tol=0.02)


def test_clean_regression_frame_keeps_original_data_when_filtering_too_much() -> None:
    """Filtering should not apply when too little source data would remain."""
    data = _solar_regression_input(zero_slice=slice(0, 55), spikes={})
    frame = data.data_frame()

    cleaned, diagnostics = clean_regression_frame(frame, DEPENDENT)

    assert not diagnostics.applied
    assert diagnostics.reason == "too much source data would be removed"
    assert len(cleaned) == 90
