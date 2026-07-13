"""Solar production source-data filtering.

Removes physically implausible solar production observations from a source
frame before analysis: negative production, zero production while a solar
reference signal indicates meaningful production, and robust
production-to-reference ratio outliers.

The filter removes rows; it never mutates values. Run it on raw (pre-resample)
data so aggregated energy totals are computed from the retained observations.
"""

import numpy as np
import pandas as pd

from .models import FilteringDiagnostics, SolarSourceFilteringParameters


def _resolve_reference_column(
    frame: pd.DataFrame,
    reference_column: str | None,
    parameters: SolarSourceFilteringParameters,
) -> str | None:
    if reference_column is not None:
        if reference_column not in frame.columns:
            raise ValueError(f"Reference column '{reference_column}' not found in frame.")
        return reference_column

    for name in parameters.solar_reference_names:
        if name in frame.columns:
            return name

    for column in frame.columns:
        lower = column.lower()
        if "solar" in lower and ("generation" in lower or "radiation" in lower):
            return column

    return None


def _positive_reference_threshold(
    series: pd.Series,
    parameters: SolarSourceFilteringParameters,
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
    parameters: SolarSourceFilteringParameters,
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


def clean_solar_production_frame(
    frame: pd.DataFrame,
    production_column: str,
    reference_column: str | None = None,
    parameters: SolarSourceFilteringParameters | None = None,
) -> tuple[pd.DataFrame, FilteringDiagnostics]:
    """Remove invalid solar production observations from a source frame.

    Returns the filtered frame and diagnostics. When guardrails reject the
    filtering (too few or too small a fraction of observations would remain),
    the original frame is returned untouched and the diagnostics keep the
    identified counts so callers can see what would have been removed.
    """

    parameters = parameters or SolarSourceFilteringParameters()

    if production_column not in frame.columns:
        raise ValueError(f"Production column '{production_column}' not found in frame.")

    original_count = len(frame)
    diagnostics = FilteringDiagnostics(
        status="nothing-to-remove",
        originalObservationCount=original_count,
        retainedObservationCount=original_count,
        removedObservationCount=0,
    )

    if original_count == 0:
        diagnostics.reason = "empty frame"
        return frame, diagnostics

    reference = _resolve_reference_column(frame, reference_column, parameters)

    # Numeric view used for mask computation only; the returned frame keeps
    # the original values.
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    y = numeric[production_column]
    keep = pd.Series(True, index=numeric.index)

    # NaN/inf production (e.g. non-numeric values coerced above) would slip
    # past the comparison masks below unnoticed, so drop it explicitly first.
    non_finite_mask = ~np.isfinite(y)
    diagnostics.identified_non_finite_count = int(non_finite_mask.sum())
    keep &= ~non_finite_mask

    negative_mask = y < 0
    diagnostics.identified_negative_count = int(negative_mask.sum())
    keep &= ~negative_mask

    if reference is not None:
        reference_series = numeric[reference]
        reference_threshold = _positive_reference_threshold(reference_series[keep], parameters)

        zero_with_solar_mask = (y <= 0) & (reference_series > reference_threshold)
        diagnostics.identified_zero_with_solar_count = int((keep & zero_with_solar_mask).sum())
        keep &= ~zero_with_solar_mask

        ratio_candidates = (
            keep
            & np.isfinite(y)
            & np.isfinite(reference_series)
            & (y > 0)
            & (reference_series > reference_threshold)
        )
        ratios = y[ratio_candidates] / reference_series[ratio_candidates]
        ratio_outliers = _robust_ratio_outlier_mask(ratios, parameters)
        diagnostics.identified_ratio_outlier_count = int(ratio_outliers.sum())
        keep.loc[ratio_outliers[ratio_outliers].index] = False

    retained_count = int(keep.sum())
    identified_count = original_count - retained_count

    if identified_count == 0:
        diagnostics.reason = "no invalid solar observations identified"
        return frame, diagnostics

    if retained_count < parameters.minimum_retained_rows:
        diagnostics.status = "rejected-guardrail"
        diagnostics.reason = "too few observations would remain after filtering"
        return frame, diagnostics

    if retained_count / original_count < parameters.minimum_retained_fraction:
        diagnostics.status = "rejected-guardrail"
        diagnostics.reason = "too much source data would be removed"
        return frame, diagnostics

    diagnostics.status = "applied"
    diagnostics.retained_observation_count = retained_count
    diagnostics.removed_observation_count = identified_count
    diagnostics.reason = "solar production source-data filtering applied"
    return frame.loc[keep].copy(), diagnostics
