"""Pydantic models for long-term PV evolution analysis."""

import datetime as dt
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openenergyid import TimeDataFrame, const


class PVLongTermAnalysisInput(TimeDataFrame):
    """Input model for long-term PV evolution analysis."""

    reference: str | None = None
    baseline_start: dt.date | None = Field(
        default=None,
        validation_alias="baselineStart",
        serialization_alias="baselineStart",
    )
    baseline_end: dt.date | None = Field(
        default=None,
        validation_alias="baselineEnd",
        serialization_alias="baselineEnd",
    )
    timezone: str = Field(
        validation_alias="timeZone",
        serialization_alias="timeZone",
    )

    model_config = ConfigDict(populate_by_name=True)

    def _datetime_index(self, frame: pd.DataFrame) -> pd.DatetimeIndex:
        """Return `frame.index` as a `DatetimeIndex` for static type checkers."""
        return pd.DatetimeIndex(frame.index)

    @model_validator(mode="after")
    def validate_frame(self) -> "PVLongTermAnalysisInput":
        """Validate required columns, daily cadence, and baseline coverage."""
        required_columns = {const.SOLAR_RADIATION, const.ELECTRICITY_PRODUCED}
        missing = required_columns.difference(self.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if (self.baseline_start is None) != (self.baseline_end is None):
            raise ValueError("Provide both 'baselineStart' and 'baselineEnd', or neither.")

        if (
            self.baseline_start is not None
            and self.baseline_end is not None
            and self.baseline_start > self.baseline_end
        ):
            raise ValueError("'baselineStart' must be on or before 'baselineEnd'.")

        frame = self.to_pandas(timezone=self.timezone).sort_index()
        if frame.empty:
            raise ValueError("Input must contain at least one daily row.")

        frame_index = self._datetime_index(frame)
        local_days = frame_index.normalize().tz_localize(None)  # type: ignore

        if local_days.has_duplicates:
            raise ValueError(
                "Input must contain exactly one row per local calendar day; duplicate days "
                "were found."
            )

        expected_days = pd.date_range(start=local_days[0], end=local_days[-1], freq="D")
        if not local_days.equals(expected_days):
            raise ValueError(
                "Input must contain contiguous daily data with no missing local dates."
            )

        first_day = local_days[0].date()
        last_day = local_days[-1].date()

        if self.baseline_start is not None and self.baseline_end is not None:
            if self.baseline_start < first_day or self.baseline_end > last_day:
                raise ValueError(
                    "Explicit baseline period must be fully covered by the input data."
                )
        else:
            default_baseline_end = (
                pd.Timestamp(first_day) + pd.DateOffset(months=12) - pd.Timedelta(days=1)
            ).date()
            if default_baseline_end > last_day:
                raise ValueError("Input must cover the default 12-month baseline period.")

        if len(local_days) < 150:
            raise ValueError(
                "Input must cover at least 150 daily rows to build a month-like baseline."
            )

        return self


class PVWarningCode(StrEnum):
    """Machine-readable reasons why part of the input was not used."""

    HORIZONTAL_IRRADIANCE_INCOMPLETE = "horizontal_irradiance_incomplete"


class PVWarning(BaseModel):
    """A caveat attached to the analysis."""

    code: PVWarningCode
    message: str
    context: dict = Field(default_factory=dict)


class PVBaselinePeriod(BaseModel):
    """Actual inclusive baseline period used for the analysis."""

    start: dt.date
    end: dt.date


class PVYearResult(BaseModel):
    """Per-calendar-year PV analysis result."""

    year: int
    actual_production: float = Field(serialization_alias="actualProduction")
    predicted_production: float = Field(serialization_alias="predictedProduction")
    error: float
    relative_error: float | None = Field(
        serialization_alias="relativeError",
        description=(
            "(actual - predicted) / predicted, so -1 means no production at all. "
            "None when the prediction is zero (no irradiance) and there was production."
        ),
    )
    complete_year: bool = Field(serialization_alias="completeYear")
    relative_error_uncertainty: float | None = Field(
        default=None,
        serialization_alias="relativeErrorUncertainty",
        description=(
            "One standard deviation of `relative_error`, in the same unit. Combines this "
            "year's noise with that of the reference fit, both from the residual noise of "
            "the reference buckets. Wider for incomplete years. None when `relative_error` "
            "is None."
        ),
    )

    model_config = ConfigDict(populate_by_name=True)


class PVTrend(BaseModel):
    """Linear trend of the yearly relative error: the change in performance per year."""

    slope: float = Field(
        description="Change of `relative_error` per year, so -0.01 is a loss of 1% per year."
    )
    slope_standard_error: float = Field(
        serialization_alias="slopeStandardError",
        description="One standard deviation of `slope`.",
    )
    intercept: float = Field(
        description="Fitted `relative_error` in `first_year`. The trend line is "
        "intercept + slope * (year - first_year)."
    )
    first_year: int = Field(serialization_alias="firstYear")
    last_year: int = Field(serialization_alias="lastYear")
    year_count: int = Field(
        serialization_alias="yearCount",
        description="Number of years in the fit, including the reference year when it counts.",
    )

    model_config = ConfigDict(populate_by_name=True)


class PVRegressionDiagnostics(BaseModel):
    """Linear regression diagnostics for the reference model."""

    coefficient: float = Field(
        description="Weight of `solar_radiation` (tilted-plane irradiance) in the model."
    )
    coefficient_horizontal: float | None = Field(
        default=None,
        serialization_alias="coefficientHorizontal",
        description=(
            "Weight of `solar_radiation_horizontal`. None when the input has no "
            "horizontal irradiance and the model uses the tilted series only."
        ),
    )
    intercept: float
    r_squared: float = Field(serialization_alias="rSquared")

    model_config = ConfigDict(populate_by_name=True)


class PVLongTermAnalysisOutput(BaseModel):
    """Output model for long-term PV evolution analysis."""

    reference: str | None = None
    baseline_period: PVBaselinePeriod = Field(serialization_alias="baselinePeriod")
    yearly_results: list[PVYearResult] = Field(serialization_alias="yearlyResults")
    regression_diagnostics: PVRegressionDiagnostics = Field(
        serialization_alias="regressionDiagnostics"
    )
    trend: PVTrend | None = Field(
        default=None,
        description="Trend over the complete years. None with fewer than three usable years.",
    )
    warnings: list[PVWarning] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)
