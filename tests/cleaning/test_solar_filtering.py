"""Tests for solar production source-data filtering."""

import math

import pandas as pd
import pytest

from openenergyid.cleaning import (
    SolarSourceFilteringParameters,
    clean_solar_production_frame,
)
from openenergyid.models import TimeDataFrame
from openenergyid.mvlr import MultiVariableRegressionInput, find_best_mvlr

PRODUCTION = "energyProduction/solarPhotovoltaic"
SOLAR_REFERENCE = "solarPowerGeneration"


def _solar_frame(
    *,
    zero_slice: slice | None = slice(20, 30),
    spikes: dict[int, float] | None = None,
) -> pd.DataFrame:
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

    return pd.DataFrame(
        {
            PRODUCTION: production,
            SOLAR_REFERENCE: solar_reference,
        },
        index=index,
    )


def test_filtering_removes_solar_source_outliers() -> None:
    """Zero-line and ratio outliers should be identified and removed."""
    frame = _solar_frame()

    cleaned, diagnostics = clean_solar_production_frame(frame, PRODUCTION)

    assert diagnostics.status == "applied"
    assert diagnostics.original_observation_count == 90
    assert diagnostics.identified_zero_with_solar_count == 10
    assert diagnostics.identified_ratio_outlier_count == 3
    assert diagnostics.removed_observation_count == 13
    assert diagnostics.retained_observation_count == 77
    assert len(cleaned) == 77
    assert (cleaned[PRODUCTION] > 0).all()


def test_filtering_reports_nothing_to_remove_on_clean_data() -> None:
    """Clean source data should pass through untouched."""
    frame = _solar_frame(zero_slice=None, spikes={})

    cleaned, diagnostics = clean_solar_production_frame(frame, PRODUCTION)

    assert diagnostics.status == "nothing-to-remove"
    assert diagnostics.removed_observation_count == 0
    assert diagnostics.retained_observation_count == 90
    assert cleaned is frame


def test_guardrail_rejection_returns_original_frame_with_identified_counts() -> None:
    """Guardrail rejection must not mutate the frame but keep identified counts."""
    frame = _solar_frame(zero_slice=slice(0, 55), spikes={})
    frame["label"] = "not-numeric"

    cleaned, diagnostics = clean_solar_production_frame(frame, PRODUCTION)

    assert diagnostics.status == "rejected-guardrail"
    assert diagnostics.reason == "too much source data would be removed"
    assert diagnostics.identified_zero_with_solar_count == 55
    assert diagnostics.removed_observation_count == 0
    assert diagnostics.retained_observation_count == 90
    assert cleaned is frame
    assert (cleaned["label"] == "not-numeric").all()


def test_guardrail_rejects_when_too_few_rows_would_remain() -> None:
    """The minimum-retained-rows guardrail should reject independently."""
    frame = _solar_frame().iloc[:35]

    cleaned, diagnostics = clean_solar_production_frame(
        frame,
        PRODUCTION,
        parameters=SolarSourceFilteringParameters(minimum_retained_rows=30),
    )

    assert diagnostics.status == "rejected-guardrail"
    assert diagnostics.reason == "too few observations would remain after filtering"
    assert cleaned is frame


def test_guardrails_are_configurable() -> None:
    """Relaxing the retained-fraction guardrail should let filtering apply."""
    frame = _solar_frame(zero_slice=slice(0, 55), spikes={})

    cleaned, diagnostics = clean_solar_production_frame(
        frame,
        PRODUCTION,
        parameters=SolarSourceFilteringParameters(
            minimum_retained_fraction=0.30,
            ratio_robust_z_threshold=999.0,
        ),
    )

    assert diagnostics.status == "applied"
    assert diagnostics.identified_zero_with_solar_count == 55
    assert diagnostics.removed_observation_count == 55
    assert len(cleaned) == 35


def test_negative_production_is_identified() -> None:
    """Negative production values should be removed as physically impossible."""
    frame = _solar_frame(zero_slice=None, spikes={})
    frame.iloc[5, frame.columns.get_loc(PRODUCTION)] = -4.0

    cleaned, diagnostics = clean_solar_production_frame(frame, PRODUCTION)

    assert diagnostics.status == "applied"
    assert diagnostics.identified_negative_count == 1
    assert diagnostics.removed_observation_count == 1
    assert len(cleaned) == 89


def test_non_finite_production_is_filtered() -> None:
    """NaN/inf production values must be removed explicitly, not slip through."""
    frame = _solar_frame(zero_slice=None, spikes={})
    frame.iloc[5, frame.columns.get_loc(PRODUCTION)] = float("nan")
    frame.iloc[9, frame.columns.get_loc(PRODUCTION)] = float("inf")

    cleaned, diagnostics = clean_solar_production_frame(frame, PRODUCTION)

    assert diagnostics.status == "applied"
    assert diagnostics.identified_non_finite_count == 2
    assert diagnostics.removed_observation_count == 2
    assert len(cleaned) == 88
    assert cleaned[PRODUCTION].notna().all()


def test_explicit_reference_column_is_used() -> None:
    """An explicitly named reference column should take precedence."""
    frame = _solar_frame().rename(columns={SOLAR_REFERENCE: "customReference"})

    cleaned, diagnostics = clean_solar_production_frame(
        frame,
        PRODUCTION,
        reference_column="customReference",
    )

    assert diagnostics.status == "applied"
    assert diagnostics.identified_zero_with_solar_count == 10
    assert len(cleaned) == 77


def test_missing_columns_raise() -> None:
    """Unknown production or reference columns are caller errors."""
    frame = _solar_frame()

    with pytest.raises(ValueError, match="Production column"):
        clean_solar_production_frame(frame, "unknownColumn")

    with pytest.raises(ValueError, match="Reference column"):
        clean_solar_production_frame(frame, PRODUCTION, reference_column="unknownColumn")


def test_without_reference_column_only_negative_filtering_applies() -> None:
    """Without any solar reference, only the negative-production filter runs."""
    frame = _solar_frame(zero_slice=None, spikes={}).drop(columns=[SOLAR_REFERENCE])
    frame.iloc[3, frame.columns.get_loc(PRODUCTION)] = -1.0
    frame.iloc[7, frame.columns.get_loc(PRODUCTION)] = 0.0

    cleaned, diagnostics = clean_solar_production_frame(frame, PRODUCTION)

    assert diagnostics.status == "applied"
    assert diagnostics.identified_negative_count == 1
    assert diagnostics.identified_zero_with_solar_count == 0
    assert len(cleaned) == 89


def test_parameters_support_json_aliases() -> None:
    """Filtering parameters should be usable from API-shaped input."""
    parameters = SolarSourceFilteringParameters.model_validate(
        {
            "minimumRetainedRows": 12,
            "minimumRetainedFraction": 0.25,
            "solarReferenceNames": ["customSolarReference"],
            "ratioRobustZThreshold": 8.0,
        }
    )

    assert parameters.minimum_retained_rows == 12
    assert parameters.minimum_retained_fraction == 0.25
    assert parameters.solar_reference_names == ("customSolarReference",)
    assert parameters.model_dump(by_alias=True)["ratioRobustZThreshold"] == 8.0


def test_diagnostics_serialize_with_camel_case_aliases() -> None:
    """Diagnostics must serialize to the wire format used by callers."""
    frame = _solar_frame()

    _, diagnostics = clean_solar_production_frame(frame, PRODUCTION)
    payload = diagnostics.model_dump(by_alias=True)

    assert payload["status"] == "applied"
    assert payload["originalObservationCount"] == 90
    assert payload["identifiedZeroWithSolarCount"] == 10
    assert payload["removedObservationCount"] == 13


def test_cleaning_as_pre_step_enables_valid_mvlr() -> None:
    """A model should fit on the cleaned frame, with MVLR itself staying pure."""
    frame = _solar_frame()

    cleaned, diagnostics = clean_solar_production_frame(frame, PRODUCTION)

    data = MultiVariableRegressionInput.model_validate(
        {
            "timeZone": "Europe/Brussels",
            "independentVariables": [
                {
                    "name": SOLAR_REFERENCE,
                    "allowNegativeCoefficient": False,
                },
            ],
            "dependentVariable": PRODUCTION,
            "frame": TimeDataFrame.from_pandas(cleaned).model_dump(),
            "granularities": ["P1D"],
            "allowNegativePredictions": False,
            "validationParameters": {
                "rsquared": 0.95,
                "f_pvalue": 0.05,
                "pvalues": 0.05,
            },
        },
    )
    result = find_best_mvlr(data)

    assert diagnostics.status == "applied"
    assert result.r2 > 0.99
    assert math.isclose(result.independent_variables[0].coef, 4.35, rel_tol=0.02)
