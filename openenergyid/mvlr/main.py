"""Main module for the MultiVariableLinearRegression class."""

from .models import MultiVariableRegressionInput, MultiVariableRegressionResult
from .helpers import resample_input_data
from .mvlr import MultiVariableLinearRegression


def find_best_mvlr(
    data: MultiVariableRegressionInput,
) -> MultiVariableRegressionResult:
    """Cycle through multiple granularities and return the best model."""
    for granularity in data.granularities:
        frame = data.data_frame()
        frame = resample_input_data(data=frame, granularity=granularity)
        mvlr = MultiVariableLinearRegression(
            data=frame,
            y=data.dependent_variable,
            granularity=granularity,
            allow_negative_predictions=data.allow_negative_predictions,
            single_use_exog_prefixes=data.single_use_exog_prefixes,
            exogs__disallow_negative_coefficient=data.get_disallowed_negative_coefficients(),
        )
        mvlr.do_analysis()
        if mvlr.validate(
            min_rsquared=data.validation_parameters.rsquared,
            max_f_pvalue=data.validation_parameters.f_pvalue,
            max_pvalues=data.validation_parameters.pvalues,
        ):
            return MultiVariableRegressionResult.from_mvlr(mvlr)
    raise ValueError("No valid model found.")
