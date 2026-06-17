"""Tests for MVLR source-data filtering."""

import math

import pandas as pd

from openenergyid.models import TimeDataFrame
from openenergyid.mvlr import (
    MultiVariableRegressionInput,
    SourceDataFilteringParameters,
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
    if spikes is None:
        spikes = {45: 70.0, 60: 75.0, 75: 80.0}
    for idx, value in spikes.items():
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


def test_clean_regression_frame_uses_filtering_parameters() -> None:
    """Retained-data guardrails should be caller-configurable."""
    data = _solar_regression_input(zero_slice=slice(0, 55), spikes={})
    frame = data.data_frame()

    cleaned, diagnostics = clean_regression_frame(
        frame,
        DEPENDENT,
        SourceDataFilteringParameters(
            minimum_retained_fraction=0.30,
            ratio_robust_z_threshold=999.0,
        ),
    )

    assert diagnostics.applied
    assert diagnostics.removed_zero_with_solar_count == 55
    assert diagnostics.removed_observation_count == 55
    assert len(cleaned) == 35


def test_source_data_filtering_parameters_support_json_aliases() -> None:
    """Filtering parameters should be usable from API-shaped input."""
    parameters = SourceDataFilteringParameters.model_validate(
        {
            "enabled": False,
            "minimumRetainedRows": 12,
            "minimumRetainedFraction": 0.25,
            "solarReferenceNames": ["customSolarReference"],
            "ratioRobustZThreshold": 8.0,
        }
    )

    assert not parameters.enabled
    assert parameters.minimum_retained_rows == 12
    assert parameters.minimum_retained_fraction == 0.25
    assert parameters.solar_reference_names == ("customSolarReference",)
    assert parameters.model_dump(by_alias=True)["ratioRobustZThreshold"] == 8.0


def test_clean_regression_frame_can_be_disabled() -> None:
    """Filtering can be disabled without changing the source frame."""
    data = _solar_regression_input()
    frame = data.data_frame()

    cleaned, diagnostics = clean_regression_frame(
        frame,
        DEPENDENT,
        SourceDataFilteringParameters(enabled=False),
    )

    assert not diagnostics.enabled
    assert not diagnostics.applied
    assert diagnostics.reason == "source-data filtering disabled"
    assert cleaned is frame


def test_clean_regression_frame_removes_non_finite_rows_for_non_solar_models() -> None:
    """Generic MVLR cleaning should drop non-finite observations."""
    index = pd.date_range("2025-04-01", periods=40, freq="D", tz="Europe/Brussels")
    frame = pd.DataFrame(
        {
            "energyConsumption": [10.0] * 39 + [float("nan")],
            "temperature": [12.0 + (i % 10) for i in range(40)],
        },
        index=index,
    )

    cleaned, diagnostics = clean_regression_frame(frame, "energyConsumption")

    assert diagnostics.applied
    assert diagnostics.reason == "generic non-finite filtering only"
    assert diagnostics.removed_non_finite_count == 1
    assert diagnostics.removed_observation_count == 1
    assert len(cleaned) == 39
