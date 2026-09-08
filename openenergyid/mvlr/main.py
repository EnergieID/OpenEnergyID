"""Main module for the MultiVariableLinearRegression class."""

import math

from .helpers import resample_input_data
from .models import (
    DegenerateModelError,
    MultiVariableRegressionInput,
    MultiVariableRegressionResult,
)
from .mvlr import MultiVariableLinearRegression


def find_best_mvlr(
    data: MultiVariableRegressionInput,
) -> MultiVariableRegressionResult:
    """Cycle through multiple granularities and return the best model.

    If at least one fit passes the caller's validation parameters, the first
    passing fit is returned with `is_valid=True`.

    Otherwise the best *representable* fit across granularities is returned
    with `is_valid=False` and a `validation_message` naming the actual numbers,
    so callers that deliberately relax the thresholds (e.g. to render a weak
    model with a warning) can do so. A fit is "representable" when its core
    statistics are finite; a fully degenerate fit (typically `df_resid == 0`
    after resampling) has no representable form and raises
    `DegenerateModelError`.
    """
    best_representable: MultiVariableLinearRegression | None = None
    best_rsquared_adj = -math.inf

    for granularity in data.granularities:
        frame = data.data_frame()
        frame = resample_input_data(data=frame, granularity=granularity)
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
            return MultiVariableRegressionResult.from_mvlr(mvlr)

        rsq_adj = float(mvlr.fit.rsquared_adj)
        if math.isfinite(rsq_adj) and rsq_adj > best_rsquared_adj:
            best_rsquared_adj = rsq_adj
            best_representable = mvlr

    if best_representable is None:
        # Every granularity produced a degenerate fit. Raise from the last one
        # so the caller can read nobs / df_model / df_resid.
        MultiVariableRegressionResult.from_mvlr(mvlr)  # always raises
        raise DegenerateModelError(  # unreachable; keeps the type checker honest
            "Every granularity produced a degenerate fit.",
            nobs=float("nan"),
            df_resid=float("nan"),
            df_model=float("nan"),
        )

    message = (
        f"No fit met the validation thresholds. Best adjusted R²: {best_rsquared_adj:.3f} "
        f"(need ≥{data.validation_parameters.rsquared}). "
        "Returned the best representable fit with is_valid=False."
    )
    return MultiVariableRegressionResult.from_mvlr(
        best_representable,
        is_valid=False,
        validation_message=message,
    )
