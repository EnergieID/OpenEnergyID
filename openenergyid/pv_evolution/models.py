"""Pydantic models for long-term PV evolution analysis."""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openenergyid import TimeDataFrame, const


class PVLongTermAnalysisInput(TimeDataFrame):
    """Input model for long-term PV evolution analysis."""

    reference: str | None = None
    timezone: str = Field(
        validation_alias="timeZone",
        serialization_alias="timeZone",
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_frame(self) -> "PVLongTermAnalysisInput":
        """Validate required columns, monthly cadence, and regression-ready values."""
        required_columns = {const.SOLAR_RADIATION, const.ELECTRICITY_PRODUCED}
        missing = required_columns.difference(self.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        frame = pd.DataFrame(self.data, columns=self.columns, index=self.index).sort_index()
        required_column_names = sorted(required_columns)
        if frame.empty:
            raise ValueError("At least 12 monthly rows are required for the reference period.")

        month_periods = (
            pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_localize(None).to_period("M")
        )

        if month_periods.has_duplicates:
            raise ValueError(
                "Input must contain exactly one row per calendar month; duplicate months "
                "were found."
            )

        expected_months = pd.period_range(start=month_periods[0], end=month_periods[-1], freq="M")
        if not month_periods.equals(expected_months):
            raise ValueError(
                "Input must contain contiguous monthly data with no missing calendar months."
            )

        if len(self.index) < 12:
            raise ValueError("At least 12 monthly rows are required for the reference period.")

        missing_mask = frame[required_column_names].isna()  # pylint: disable=unsubscriptable-object
        if missing_mask.any().any():
            missing_required_columns = sorted(missing_mask.columns[missing_mask.any()].tolist())
            invalid_row_count = int(missing_mask.any(axis=1).sum())
            raise ValueError(
                "Missing values found in required columns "
                f"{missing_required_columns} across {invalid_row_count} row(s)."
            )

        return self


class PVYearResult(BaseModel):
    """Per-calendar-year PV analysis result."""

    year: int
    actual_production: float = Field(serialization_alias="actualProduction")
    predicted_production: float = Field(serialization_alias="predictedProduction")
    error: float
    relative_error: float | None = Field(serialization_alias="relativeError")
    complete_year: bool = Field(serialization_alias="completeYear")

    model_config = ConfigDict(populate_by_name=True)


class PVRegressionDiagnostics(BaseModel):
    """Linear regression diagnostics for the reference model."""

    coefficient: float
    intercept: float
    r_squared: float = Field(serialization_alias="rSquared")

    model_config = ConfigDict(populate_by_name=True)


class PVLongTermAnalysisOutput(BaseModel):
    """Output model for long-term PV evolution analysis."""

    reference: str | None = None
    yearly_results: list[PVYearResult] = Field(serialization_alias="yearlyResults")
    regression_diagnostics: PVRegressionDiagnostics = Field(
        serialization_alias="regressionDiagnostics"
    )

    model_config = ConfigDict(populate_by_name=True)
