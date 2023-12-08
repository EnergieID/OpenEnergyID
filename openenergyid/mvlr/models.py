"""Models for multivariable linear regression."""
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict
import statsmodels.formula.api as fm

from openenergyid.enums import Granularity
from openenergyid.models import TimeSeries

from .mvlr import MultiVariableLinearRegression


class ConfidenceInterval(BaseModel):
    """Confidence interval for a coefficient."""

    confidence: float = Field(ge=0, le=1)
    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)


class IndependentVariable(BaseModel):
    """Independent variable for a multivariable linear regression model."""

    name: str
    coef: float
    t_stat: Optional[float] = Field(default=None, alias="tStat")
    p_value: Optional[float] = Field(ge=0, le=1, default=None, alias="pValue")
    std_err: Optional[float] = Field(default=None, alias="stdErr")
    confidence_interval: Optional[ConfidenceInterval] = Field(
        default=None, alias="confidenceInterval"
    )

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_fit(cls, fit: fm.ols, name: str) -> "IndependentVariable":
        """Create an IndependentVariable from a fit."""
        return cls(
            name=name,
            coef=fit.params[name],
            t_stat=fit.tvalues[name],
            p_value=fit.pvalues[name],
            std_err=fit.bse[name],
            confidence_interval=ConfidenceInterval(
                confidence=0.95,
                lower=fit.conf_int().transpose()[name][0],
                upper=fit.conf_int().transpose()[name][1],
            ),
        )


class MultiVariableRegressionResult(BaseModel):
    """Result of a multivariable regression model."""

    dependent_variable: str = Field(alias="dependentVariable")
    independent_variables: list[IndependentVariable] = Field(alias="independentVariables")
    r2: float = Field(ge=0, le=1, alias="rSquared")
    r2_adj: float = Field(ge=0, le=1, alias="rSquaredAdjusted")
    f_stat: float = Field(ge=0, alias="fStat")
    prob_f_stat: float = Field(ge=0, le=1, alias="probFStat")
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
