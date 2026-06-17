"""Main module for the MultiVariableLinearRegression class."""

from .helpers import resample_input_data
from .models import MultiVariableRegressionInput, MultiVariableRegressionResult
from .mvlr import MultiVariableLinearRegression
from .source_data_filtering import clean_regression_frame, clean_solar_source_frame


def find_best_mvlr(
    data: MultiVariableRegressionInput,
) -> MultiVariableRegressionResult:
    """Cycle through multiple granularities and return the best model."""
    best_rsquared = 0
    best_filtering = None
    for granularity in data.granularities:
        frame = data.data_frame()
        frame, solar_filtering = clean_solar_source_frame(
            frame,
            data.dependent_variable,
            data.source_data_filtering,
        )
        frame = resample_input_data(data=frame, granularity=granularity)
        frame, finite_filtering = clean_regression_frame(
            frame,
            data.dependent_variable,
            data.source_data_filtering,
        )
        filtering = solar_filtering if solar_filtering.applied else finite_filtering
        best_filtering = filtering
        mvlr = MultiVariableLinearRegression(
            data=frame,
            y=data.dependent_variable,
            granularity=granularity,
            allow_negative_predictions=data.allow_negative_predictions,
            single_use_exog_prefixes=data.single_use_exog_prefixes or [],
            exogs__disallow_negative_coefficient=data.get_disallowed_negative_coefficients(),
        )
        mvlr.do_analysis()
        if mvlr.validate(
            min_rsquared=data.validation_parameters.rsquared,
            max_f_pvalue=data.validation_parameters.f_pvalue,
            max_pvalues=data.validation_parameters.pvalues,
        ):
            result = MultiVariableRegressionResult.from_mvlr(mvlr)
            result.outlier_filtering = filtering
            return result
        best_rsquared = max(best_rsquared, mvlr.fit.rsquared_adj)
    detail = (
        f"No valid model found. Best R²: {best_rsquared:.3f} "
        f"(need ≥{data.validation_parameters.rsquared})"
    )
    if best_filtering and best_filtering.applied:
        detail += (
            f"; outlier filtering removed {best_filtering.removed_observation_count}/"
            f"{best_filtering.original_observation_count} observations"
        )
    raise ValueError(detail)
