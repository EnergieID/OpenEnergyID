"""Models for multivariable linear regression."""

import math
from typing import Any

import pandas as pd
import statsmodels.formula.api as fm
from pydantic import BaseModel, ConfigDict, Field

from openenergyid.enums import Granularity
from openenergyid.models import TimeDataFrame

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
    variants: list[str] | None = Field(
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
    independent_variables: list[IndependentVariableInput] = Field(
        alias="independentVariables", min_length=1
    )
    dependent_variable: str = Field(alias="dependentVariable")
    frame: TimeDataFrame
    granularities: list[Granularity]
    allow_negative_predictions: bool = Field(alias="allowNegativePredictions", default=False)
    validation_parameters: ValidationParameters = Field(
        alias="validationParameters", default_factory=ValidationParameters
    )
    single_use_exog_prefixes: list[str] | None = Field(
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

    def get_disallowed_negative_coefficients(self) -> list[str]:
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
    t_stat: float | None = Field(default=None, alias="tStat")
    p_value: float | None = Field(ge=0, le=1, default=None, alias="pValue")
    std_err: float | None = Field(default=None, alias="stdErr")
    confidence_interval: ConfidenceInterval | None = Field(default=None, alias="confidenceInterval")

    model_config = ConfigDict(populate_by_name=True)

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        """Return None if value is NaN or infinite, otherwise the value itself."""
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)

    @classmethod
    def from_fit(cls, fit: fm.ols, name: str) -> "IndependentVariableResult":
        """Create an IndependentVariable from a fit."""
        ci_lower = float(fit.conf_int().transpose()[name][0])
        ci_upper = float(fit.conf_int().transpose()[name][1])
        has_valid_ci = not (
            math.isnan(ci_lower)
            or math.isinf(ci_lower)
            or math.isnan(ci_upper)
            or math.isinf(ci_upper)
        )
        return cls(
            name=name,
            coef=fit.params[name],
            t_stat=cls._finite_or_none(fit.tvalues[name]),
            p_value=cls._finite_or_none(fit.pvalues[name]),
            std_err=cls._finite_or_none(fit.bse[name]),
            confidence_interval=ConfidenceInterval(
                confidence=0.95,
                lower=ci_lower,
                upper=ci_upper,
            )
            if has_valid_ci
            else None,
        )


class DegenerateModelError(ValueError):
    """A regression fit whose core statistics are not finite (typically df_resid == 0).

    Kept as a `ValueError` subclass so callers doing a broad `except ValueError`
    keep working; new callers can distinguish it and read the observation counts.
    """

    def __init__(self, message: str, *, nobs: float, df_resid: float, df_model: float) -> None:
        super().__init__(message)
        self.nobs = nobs
        self.df_resid = df_resid
        self.df_model = df_model


class MultiVariableRegressionResult(BaseModel):
    """Result of a multivariable regression model."""

    dependent_variable: str = Field(alias="dependentVariable")
    independent_variables: list[IndependentVariableResult] = Field(alias="independentVariables")
    r2: float = Field(ge=0, le=1, alias="rSquared")
    # r2_adj can be negative for a model worse than the mean; only the upper bound is
    # meaningful. See AB#820 and AB#667: it must not be null (v3's C# DTO is
    # non-nullable double), so a negative value is used to signal "worse than intercept".
    r2_adj: float = Field(le=1, alias="rSquaredAdjusted")
    f_stat: float = Field(ge=0, alias="fStat")
    prob_f_stat: float = Field(ge=0, le=1, alias="probFStat")
    intercept: IndependentVariableResult
    granularity: Granularity
    frame: TimeDataFrame
    is_valid: bool = Field(default=True, alias="isValid")
    validation_message: str | None = Field(default=None, alias="validationMessage")

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_mvlr(
        cls,
        mvlr: MultiVariableLinearRegression,
        *,
        is_valid: bool = True,
        validation_message: str | None = None,
    ) -> "MultiVariableRegressionResult":
        """Create a MultiVariableRegressionResult from a MultiVariableLinearRegression.

        Raises `DegenerateModelError` if the core statistics (r2, r2_adj, f_stat,
        prob_f_stat) are not finite — typically because `df_resid == 0` after
        resampling, leaving no residual degrees of freedom for the F distribution.
        """
        fit = mvlr.fit
        core = {
            "r2": float(fit.rsquared),
            "r2_adj": float(fit.rsquared_adj),
            "f_stat": float(fit.fvalue),
            "prob_f_stat": float(fit.f_pvalue),
        }
        non_finite = [name for name, value in core.items() if not math.isfinite(value)]
        if non_finite:
            raise DegenerateModelError(
                "Regression fit is degenerate: non-finite "
                f"{', '.join(non_finite)} (nobs={fit.nobs}, df_model={fit.df_model}, "
                f"df_resid={fit.df_resid}). Typically df_resid == 0 after resampling.",
                nobs=float(fit.nobs),
                df_resid=float(fit.df_resid),
                df_model=float(fit.df_model),
            )

        # Get independent variables
        param_keys = fit.params.keys().tolist()
        param_keys.remove("Intercept")
        independent_variables = []
        for k in param_keys:
            independent_variables.append(IndependentVariableResult.from_fit(fit, k))

        # Create resulting TimeSeries
        cols_to_keep = list(param_keys)
        cols_to_keep.append(mvlr.y)
        cols_to_remove = list(filter(lambda v: v not in cols_to_keep, mvlr.data.columns.values))
        frame = mvlr.data.drop(cols_to_remove, axis=1)

        return cls(
            dependent_variable=mvlr.y,
            independent_variables=independent_variables,
            r2=core["r2"],
            r2_adj=core["r2_adj"],
            f_stat=core["f_stat"],
            prob_f_stat=core["prob_f_stat"],
            intercept=IndependentVariableResult.from_fit(fit, "Intercept"),
            granularity=mvlr.granularity,
            frame=TimeDataFrame.from_pandas(frame),
            is_valid=is_valid,
            validation_message=validation_message,
        )
