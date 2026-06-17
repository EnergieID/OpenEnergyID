"""Source-data filtering helpers for multi-variable regression."""

import numpy as np
import pandas as pd

from .models import OutlierFilteringDiagnostics, SourceDataFilteringParameters


def _solar_reference_column(
    frame: pd.DataFrame,
    parameters: SourceDataFilteringParameters,
) -> str | None:
    for name in parameters.solar_reference_names:
        if name in frame.columns:
            return name

    for column in frame.columns:
        lower = column.lower()
        if "solar" in lower and ("generation" in lower or "radiation" in lower):
            return column

    return None


def _is_solar_production_model(
    dependent_variable: str,
    frame: pd.DataFrame,
    parameters: SourceDataFilteringParameters,
) -> bool:
    dependent = dependent_variable.lower()
    if "solarphotovoltaic" in dependent:
        return True
    if "solar" in dependent and "production" in dependent:
        return True
    return "production" in dependent and _solar_reference_column(frame, parameters) is not None


def _positive_reference_threshold(
    series: pd.Series,
    parameters: SourceDataFilteringParameters,
) -> float:
    positive = series[series > 0]
    if positive.empty:
        return 0.0
    return max(
        float(positive.median()) * parameters.positive_reference_median_fraction,
        parameters.minimum_positive_reference,
    )


def _robust_ratio_outlier_mask(
    ratio: pd.Series,
    parameters: SourceDataFilteringParameters,
) -> pd.Series:
    if len(ratio) < parameters.minimum_retained_rows:
        return pd.Series(False, index=ratio.index)

    median = float(ratio.median())
    mad = float((ratio - median).abs().median())
    if not np.isfinite(mad) or mad <= 0:
        q1 = float(ratio.quantile(0.25))
        q3 = float(ratio.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            return pd.Series(False, index=ratio.index)
        return (ratio < q1 - parameters.ratio_iqr_multiplier * iqr) | (
            ratio > q3 + parameters.ratio_iqr_multiplier * iqr
        )

    robust_z = 0.6745 * (ratio - median).abs() / mad
    return robust_z > parameters.ratio_robust_z_threshold


def clean_regression_frame(
    frame: pd.DataFrame,
    dependent_variable: str,
    parameters: SourceDataFilteringParameters | None = None,
) -> tuple[pd.DataFrame, OutlierFilteringDiagnostics]:
    """Remove obvious bad source observations before fitting a regression model."""

    parameters = parameters or SourceDataFilteringParameters()
    original_count = len(frame)
    diagnostics = OutlierFilteringDiagnostics(
        enabled=parameters.enabled,
        originalObservationCount=original_count,
        retainedObservationCount=original_count,
        removedObservationCount=0,
        applied=False,
    )

    if not parameters.enabled:
        diagnostics.reason = "source-data filtering disabled"
        return frame, diagnostics

    if original_count == 0 or dependent_variable not in frame.columns:
        diagnostics.reason = "empty frame or missing dependent variable"
        return frame, diagnostics

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    keep = pd.Series(True, index=numeric_frame.index)

    finite_mask = np.isfinite(numeric_frame).all(axis=1)
    diagnostics.removed_non_finite_count = int((keep & ~finite_mask).sum())
    keep &= finite_mask

    if not _is_solar_production_model(dependent_variable, numeric_frame, parameters):
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

    solar_column = _solar_reference_column(numeric_frame, parameters)
    if solar_column is not None:
        solar_reference = numeric_frame[solar_column]
        solar_threshold = _positive_reference_threshold(solar_reference[keep], parameters)

        zero_with_solar_mask = (y <= 0) & (solar_reference > solar_threshold)
        diagnostics.removed_zero_with_solar_count = int((keep & zero_with_solar_mask).sum())
        keep &= ~zero_with_solar_mask

        ratio_candidates = keep & (y > 0) & (solar_reference > solar_threshold)
        ratios = y[ratio_candidates] / solar_reference[ratio_candidates]
        ratio_outliers = _robust_ratio_outlier_mask(ratios, parameters)
        diagnostics.removed_ratio_outlier_count = int(ratio_outliers.sum())
        keep.loc[ratio_outliers[ratio_outliers].index] = False

    cleaned = numeric_frame.loc[keep].copy()
    retained_count = len(cleaned)
    removed_count = original_count - retained_count

    if retained_count < parameters.minimum_retained_rows:
        diagnostics.reason = "too few observations retained after filtering"
        return numeric_frame, diagnostics

    if retained_count / original_count < parameters.minimum_retained_fraction:
        diagnostics.reason = "too much source data would be removed"
        return numeric_frame, diagnostics

    diagnostics.retained_observation_count = retained_count
    diagnostics.removed_observation_count = removed_count
    diagnostics.applied = removed_count > 0
    diagnostics.reason = "solar production source-data filtering"
    return cleaned, diagnostics
