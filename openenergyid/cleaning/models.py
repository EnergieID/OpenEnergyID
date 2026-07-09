"""Models for source-data cleaning."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FilteringStatus = Literal["applied", "rejected-guardrail", "nothing-to-remove"]


class SolarSourceFilteringParameters(BaseModel):
    """Parameters for solar production source-data filtering."""

    minimum_retained_fraction: float = Field(
        0.50,
        ge=0,
        le=1,
        alias="minimumRetainedFraction",
        description="Minimum fraction of original observations that must remain after filtering.",
    )
    minimum_retained_rows: int = Field(
        30,
        ge=0,
        alias="minimumRetainedRows",
        description="Minimum number of observations that must remain after filtering.",
    )
    solar_reference_names: tuple[str, ...] = Field(
        ("solarPowerGeneration", "solarRadiation"),
        alias="solarReferenceNames",
        description="Preferred source column names used as solar production/radiation references.",
    )
    positive_reference_median_fraction: float = Field(
        0.10,
        ge=0,
        alias="positiveReferenceMedianFraction",
        description="Fraction of the positive solar-reference median used as meaningful-solar threshold.",
    )
    minimum_positive_reference: float = Field(
        0.05,
        ge=0,
        alias="minimumPositiveReference",
        description="Minimum meaningful-solar threshold for the solar reference column.",
    )
    ratio_iqr_multiplier: float = Field(
        3.0,
        gt=0,
        alias="ratioIqrMultiplier",
        description="IQR multiplier used when MAD-based ratio filtering is unavailable.",
    )
    ratio_robust_z_threshold: float = Field(
        4.5,
        gt=0,
        alias="ratioRobustZThreshold",
        description="Robust z-score threshold for production-to-reference ratio outliers.",
    )

    model_config = ConfigDict(populate_by_name=True)


class FilteringDiagnostics(BaseModel):
    """Diagnostics for a source-data filtering run.

    The `identified*` counts always describe what the filter identified as
    invalid, even when guardrails rejected the filtering. The observation
    counts describe what actually happened to the returned frame.
    """

    status: FilteringStatus
    original_observation_count: int = Field(alias="originalObservationCount")
    retained_observation_count: int = Field(alias="retainedObservationCount")
    removed_observation_count: int = Field(alias="removedObservationCount")
    identified_negative_count: int = Field(0, alias="identifiedNegativeCount")
    identified_zero_with_solar_count: int = Field(0, alias="identifiedZeroWithSolarCount")
    identified_ratio_outlier_count: int = Field(0, alias="identifiedRatioOutlierCount")
    reason: str | None = None

    model_config = ConfigDict(populate_by_name=True)
