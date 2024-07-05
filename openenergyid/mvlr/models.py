"""Models for multivariable linear regression."""

from typing import Any, List, Optional
import pandas as pd

from pydantic import BaseModel, Field, ConfigDict
import statsmodels.formula.api as fm

from openenergyid.enums import Granularity
from openenergyid.models import TimeSeriesCollection

from .mvlr import MultiVariableLinearRegression


COLUMN_TEMPERATUREEQUIVALENT = "temperatureEquivalent"


######################
# MVLR Input Models #
######################


class ValidationParameters(BaseModel):
    """Parameters for validation of a multivariable linear regression model."""

    rsquared: float = Field(
        0.75, ge=0, le=1, description="Minimum acceptable value for the adjusted R-squared"
    )
    f_pvalue: float = Field(
        0.05, ge=0, le=1, description="Maximum acceptable value for the F-statistic"
    )
    pvalues: float = Field(
        0.05, ge=0, le=1, description="Maximum acceptable value for the p-values of the t-statistic"
    )


class IndependentVariableInput(BaseModel):
    """
    Independent variable.

    Has to corresponds to a column in the data frame.
    """

    name: str = Field(
        description="Name of the independent variable. "
        "If the name is `temperatureEquivalent`, "
        "it will be unpacked into columns according to the variants."
    )
    variants: Optional[list[str]] = Field(
        default=None,
        description="Variants of the `temperatureEquivalent` independent variable. "
        "Eg. `HDD_16.5` will be Heating Degree Days with a base temperature of 16.5°C, "
        "`CDD_0` will be Cooling Degree Days with a base temperature of 0°C.",
    )
    allow_negative_coefficient: bool = Field(
        default=True,
        alias="allowNegativeCoefficient",
        description="Whether the coefficient can be negative.",
    )


class MultiVariableRegressionInput(BaseModel):
    """Multi-variable regression input."""

    timezone: str = Field(alias="timeZone")
    independent_variables: List[IndependentVariableInput] = Field(
        alias="independentVariables", min_length=1
    )
    dependent_variable: str = Field(alias="dependentVariable")
    frame: TimeSeriesCollection
    granularities: list[Granularity]
    allow_negative_predictions: bool = Field(alias="allowNegativePredictions", default=False)
    validation_parameters: ValidationParameters = Field(
        alias="validationParameters", default=ValidationParameters()
    )
    single_use_exog_prefixes: Optional[List[str]] = Field(
        # default=["HDD", "CDD", "FDD"],
        default=None,
        alias="singleUseExogPrefixes",
        description="List of prefixes to be used as single-use exogenous variables.",
    )

    def model_post_init(self, __context: Any) -> None:
        """Post init hook."""
        # Check if all independent variables are present in the data frame
        for iv in self.independent_variables:  # pylint: disable=not-an-iterable
            if iv.name not in self.frame.columns:
                raise ValueError(f"Independent variable {iv.name} not found in the data frame.")

        return super().model_post_init(__context)

    def _data_frame(self) -> pd.DataFrame:
        """Convert the data to a pandas DataFrame."""
        return self.frame.to_pandas(timezone=self.timezone)

    def data_frame(self) -> pd.DataFrame:
        """
        Return the data frame ready for analysis.

        Unpacks degree days and removes unnecessary columns.

        If an independent variable named `temperatureEquivalent` is present,
        it will be unpacked into columns according to the variants.
        Eg. Variant "HDD_16.5" will be Heating Degree Days
        with a base temperature of 16.5°C,
        "CDD_0" will be Cooling Degree Days with a base temperature of 0°C.
        """
        frame = self._data_frame()
        columns_to_retain = [self.dependent_variable]
        for iv in self.independent_variables:  # pylint: disable=not-an-iterable
            if iv.name == COLUMN_TEMPERATUREEQUIVALENT and iv.variants is not None:
                for variant in iv.variants:
                    prefix, base_temperature = variant.split("_")
                    if prefix == "CDD":
                        frame[variant] = frame[COLUMN_TEMPERATUREEQUIVALENT] - float(
                            base_temperature
                        )
                    else:
                        frame[variant] = (
                            float(base_temperature) - frame[COLUMN_TEMPERATUREEQUIVALENT]
                        )
                    frame[variant] = frame[variant].clip(lower=0)
                    columns_to_retain.append(variant)
                frame.drop(columns=[COLUMN_TEMPERATUREEQUIVALENT], inplace=True)
            else:
                columns_to_retain.append(iv.name)

        frame = frame[columns_to_retain].copy()

        return frame

    def get_disallowed_negative_coefficients(self) -> List[str]:
        """Get independent variables that are not allowed to have a negative coefficient."""
        result = []
        for iv in self.independent_variables:  # pylint: disable=not-an-iterable
            if iv.name == COLUMN_TEMPERATUREEQUIVALENT and iv.variants is not None:
                if not iv.allow_negative_coefficient:
                    result.extend(iv.variants)
            elif not iv.allow_negative_coefficient:
                result.append(iv.name)
        return result


######################
# MVLR Result Models #
######################


class ConfidenceInterval(BaseModel):
    """Confidence interval for a coefficient."""

    confidence: float = Field(ge=0, le=1)
    lower: float
    upper: float


class IndependentVariableResult(BaseModel):
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
    def from_fit(cls, fit: fm.ols, name: str) -> "IndependentVariableResult":
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
    independent_variables: list[IndependentVariableResult] = Field(alias="independentVariables")
    r2: float = Field(ge=0, le=1, alias="rSquared")
    r2_adj: float = Field(ge=0, le=1, alias="rSquaredAdjusted")
    f_stat: float = Field(ge=0, alias="fStat")
    prob_f_stat: float = Field(ge=0, le=1, alias="probFStat")
    intercept: IndependentVariableResult
    granularity: Granularity
    frame: TimeSeriesCollection

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_mvlr(cls, mvlr: MultiVariableLinearRegression) -> "MultiVariableRegressionResult":
        """Create a MultiVariableRegressionResult from a MultiVariableLinearRegression."""

        # Get independent variables
        param_keys = mvlr.fit.params.keys().tolist()
        param_keys.remove("Intercept")
        independent_variables = []
        for k in param_keys:
            independent_variables.append(IndependentVariableResult.from_fit(mvlr.fit, k))

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
            intercept=IndependentVariableResult.from_fit(mvlr.fit, "Intercept"),
            granularity=mvlr.granularity,
            frame=TimeSeriesCollection.from_pandas(frame),
        )
