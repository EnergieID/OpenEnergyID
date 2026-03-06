"""Pydantic models for long-term PV evolution analysis."""

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
        """Validate required columns and minimum reference data size."""
        required_columns = {const.SOLAR_RADIATION, const.ELECTRICITY_PRODUCED}
        missing = required_columns.difference(self.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if len(self.index) < 12:
            raise ValueError("At least 12 monthly rows are required for the reference period.")

        return self


class PVYearResult(BaseModel):
    """Per-calendar-year PV analysis result."""

    year: int
    actual_production: float = Field(serialization_alias="actualProduction")
    predicted_production: float = Field(serialization_alias="predictedProduction")
    error: float
    relative_error: float = Field(serialization_alias="relativeError")
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
