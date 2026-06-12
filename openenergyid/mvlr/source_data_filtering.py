"""Source-data filtering helpers for multi-variable regression."""

import numpy as np
import pandas as pd

from .models import OutlierFilteringDiagnostics

MINIMUM_RETAINED_FRACTION = 0.50
MINIMUM_RETAINED_ROWS = 30
SOLAR_REFERENCE_NAMES = ("solarPowerGeneration", "solarRadiation")


def _solar_reference_column(frame: pd.DataFrame) -> str | None:
    for name in SOLAR_REFERENCE_NAMES:
        if name in frame.columns:
            return name

    for column in frame.columns:
        lower = column.lower()
        if "solar" in lower and ("generation" in lower or "radiation" in lower):
            return column

    return None


def _is_solar_production_model(dependent_variable: str, frame: pd.DataFrame) -> bool:
    dependent = dependent_variable.lower()
    if "solarphotovoltaic" in dependent:
        return True
    if "solar" in dependent and "production" in dependent:
        return True
    return "production" in dependent and _solar_reference_column(frame) is not None


def _positive_reference_threshold(series: pd.Series) -> float:
    positive = series[series > 0]
    if positive.empty:
        return 0.0
    return max(float(positive.median()) * 0.10, 0.05)


def _robust_ratio_outlier_mask(ratio: pd.Series) -> pd.Series:
    if len(ratio) < MINIMUM_RETAINED_ROWS:
        return pd.Series(False, index=ratio.index)

    median = float(ratio.median())
    mad = float((ratio - median).abs().median())
    if not np.isfinite(mad) or mad <= 0:
        q1 = float(ratio.quantile(0.25))
        q3 = float(ratio.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            return pd.Series(False, index=ratio.index)
        return (ratio < q1 - 3.0 * iqr) | (ratio > q3 + 3.0 * iqr)

    robust_z = 0.6745 * (ratio - median).abs() / mad
    return robust_z > 4.5


def clean_regression_frame(
    frame: pd.DataFrame,
    dependent_variable: str,
) -> tuple[pd.DataFrame, OutlierFilteringDiagnostics]:
    """Remove obvious bad source observations before fitting a regression model."""

    original_count = len(frame)
    diagnostics = OutlierFilteringDiagnostics(
        originalObservationCount=original_count,
        retainedObservationCount=original_count,
        removedObservationCount=0,
        applied=False,
    )

    if original_count == 0 or dependent_variable not in frame.columns:
        diagnostics.reason = "empty frame or missing dependent variable"
        return frame, diagnostics

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    keep = pd.Series(True, index=numeric_frame.index)

    finite_mask = np.isfinite(numeric_frame).all(axis=1)
    diagnostics.removed_non_finite_count = int((keep & ~finite_mask).sum())
    keep &= finite_mask

    if not _is_solar_production_model(dependent_variable, numeric_frame):
        cleaned = numeric_frame.loc[keep].copy()
        diagnostics.retained_observation_count = len(cleaned)
        diagnostics.removed_observation_count = original_count - len(cleaned)
        diagnostics.applied = diagnostics.removed_observation_count > 0
        diagnostics.reason = "generic non-finite filtering only"
        return cleaned, diagnostics

    y = numeric_frame[dependent_variable]
    negative_mask = y < 0
    diagnostics.removed_negative_count = int((keep & negative_mask).sum())
    keep &= ~negative_mask

    solar_column = _solar_reference_column(numeric_frame)
    if solar_column is not None:
        solar_reference = numeric_frame[solar_column]
        solar_threshold = _positive_reference_threshold(solar_reference[keep])

        zero_with_solar_mask = (y <= 0) & (solar_reference > solar_threshold)
        diagnostics.removed_zero_with_solar_count = int((keep & zero_with_solar_mask).sum())
        keep &= ~zero_with_solar_mask

        ratio_candidates = keep & (y > 0) & (solar_reference > solar_threshold)
        ratios = y[ratio_candidates] / solar_reference[ratio_candidates]
        ratio_outliers = _robust_ratio_outlier_mask(ratios)
        diagnostics.removed_ratio_outlier_count = int(ratio_outliers.sum())
        keep.loc[ratio_outliers[ratio_outliers].index] = False

    cleaned = numeric_frame.loc[keep].copy()
    retained_count = len(cleaned)
    removed_count = original_count - retained_count

    if retained_count < MINIMUM_RETAINED_ROWS:
        diagnostics.reason = "too few observations retained after filtering"
        return numeric_frame, diagnostics

    if retained_count / original_count < MINIMUM_RETAINED_FRACTION:
        diagnostics.reason = "too much source data would be removed"
        return numeric_frame, diagnostics

    diagnostics.retained_observation_count = retained_count
    diagnostics.removed_observation_count = removed_count
    diagnostics.applied = removed_count > 0
    diagnostics.reason = "solar production source-data filtering"
    return cleaned, diagnostics
