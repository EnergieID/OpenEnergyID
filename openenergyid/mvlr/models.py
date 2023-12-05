"""Models for multivariable linear regression."""
from typing import Optional

from pydantic import BaseModel
import statsmodels.formula.api as fm

from openenergyid.enums import Granularity
from openenergyid.models import TimeSeries

from .mvlr import MultiVariableLinearRegression


class IndependentVariable(BaseModel):
    """Independent variable for a multivariable linear regression model."""

    name: str
    coef: float
    t_stat: Optional[float] = None
    p_value: Optional[float] = None
    std_err: Optional[float] = None
    confidence_interval: Optional[dict[str, float]] = None

    @classmethod
    def from_fit(cls, fit: fm.ols, name: str) -> "IndependentVariable":
        """Create an IndependentVariable from a fit."""
        return cls(
            name=name,
            coef=fit.params[name],
            t_stat=fit.tvalues[name],
            p_value=fit.pvalues[name],
            std_err=fit.bse[name],
            confidence_interval={
                "confidence": 0.95,
                "lower": fit.conf_int().transpose()[name][0],
                "upper": fit.conf_int().transpose()[name][1],
            },
        )


class MultiVariableRegressionResult(BaseModel):
    """Result of a multivariable regression model."""

    dependent_variable: str
    independent_variables: list[IndependentVariable]
    r2: float
    r2_adj: float
    f_stat: float
    prob_f_stat: float
    intercept: IndependentVariable
    granularity: Granularity
    frame: TimeSeries

    @classmethod
    def from_mvlr(cls, mvlr: MultiVariableLinearRegression) -> "MultiVariableRegressionResult":
        """Create a MultiVariableRegressionResult from a MultiVariableLinearRegression."""

        # Get independent variables
        param_keys = mvlr.fit.params.keys().tolist()
        param_keys.remove("Intercept")
        independent_variables = []
        for k in param_keys:
            independent_variables.append(IndependentVariable.from_fit(mvlr.fit, k))

        # Create resulting TimeSeries
        cols_to_keep = list(param_keys)
        cols_to_keep.append(mvlr.y)
        cols_to_remove = list(filter(lambda v: v not in cols_to_keep, mvlr.data.columns.values))
        frame = mvlr.data.drop(cols_to_remove, axis=1)

        return cls(
            dependent_variable=mvlr.y,
            independent_variables=independent_variables,
            r2=mvlr.fit.rsquared,
            r2_adj=mvlr.fit.rsquared_adj,
            f_stat=mvlr.fit.fvalue,
            prob_f_stat=mvlr.fit.f_pvalue,
            intercept=IndependentVariable.from_fit(mvlr.fit, "Intercept"),
            granularity=mvlr.granularity,
            frame=TimeSeries.from_pandas(frame),
        )
