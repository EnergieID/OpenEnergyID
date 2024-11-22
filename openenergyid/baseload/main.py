"""
This module provides functionality for loading, validating, and analyzing energy usage data.

Classes:
    BaseLoadMetrics: A NamedTuple container for base load analysis metrics.
    EnergySchema: A pandera DataFrameModel for validating energy usage data.

Functions:
    load_data(path: str) -> pl.LazyFrame:
        Loads and validates energy usage data from an NDJSON file.

    calculate_base_load(lf: pl.LazyFrame) -> BaseLoadMetrics:
        Calculates base load metrics from energy usage data.

    main(file_path: str) -> BaseLoadMetrics:
        Processes energy data and returns base load metrics.

    test_energy_validation():
        Tests various data validation scenarios using pytest.
"""

from typing import NamedTuple
import polars as pl
import pandera.polars as pa
## VERY important to use pandera.polars instead of pandera to avoid pandas errors


class BaseLoadMetrics(NamedTuple):
    """Container for base load analysis metrics"""

    base_load_watts: float  # Average base load in watts
    daily_usage_kwh: float  # Average daily usage in kWh
    base_percentage: float  # Base load as percentage of total


class EnergySchema(pa.DataFrameModel):
    """Schema for energy usage data validation"""

    timestamp: pl.Datetime = pa.Field(
        nullable=False,
        coerce=True,
        title="Measurement Timestamp",
        description="Time of energy measurement in Europe/Brussels timezone",
    )
    total: float = pa.Field(
        ge=0,  # Power should be non-negative
        nullable=False,
        title="Total Power",
        description="Total power measurement in kW",
    )

    # Add example of pandera validation: dataframe-level validation
    @pa.dataframe_check
    def timestamps_are_ordered(self, data: pl.DataFrame) -> bool:
        """Check if timestamps are in chronological order"""
        return data["timestamp"].is_sorted()


def load_data(path: str) -> pl.LazyFrame:
    """Load and validate energy usage data from NDJSON file"""
    lf = pl.scan_ndjson(
        path,
        schema={"timestamp": pl.Datetime(time_zone="Europe/Brussels"), "total": pl.Float64},
    )
    # Convert to DataFrame for data-level validation, then back to LazyFrame for processing
    validated_df = EnergySchema.validate(lf).collect()  # type: ignore
    return pl.LazyFrame(validated_df)


def calculate_base_load(lf: pl.LazyFrame) -> BaseLoadMetrics:
    """
    Calculate base load metrics from energy usage data.

    Takes lowest 10 totals per day to determine base load.
    Returns watts, kwh, and percentage metrics.
    """
    metrics_df = (
        lf.filter(pl.col("total") >= 0)
        .sort("timestamp")
        .group_by_dynamic("timestamp", every="1d")
        .agg(
            [
                pl.col("total").sum().alias("total_daily_usage"),
                (pl.col("total").sort().head(10).mean() * 4 * 24).alias("base_load_daily_kwh"),
            ]
        )
        .with_columns(
            [
                (pl.col("base_load_daily_kwh") / pl.col("total_daily_usage") * 100).alias(
                    "base_percentage"
                )
            ]
        )
        .select(
            [
                pl.col("base_load_daily_kwh").mean().alias("avg_daily_kwh"),
                (pl.col("base_load_daily_kwh") * 1000 / 24).mean().alias("avg_watts"),
                pl.col("base_percentage").mean().alias("avg_percentage"),
            ]
        )
        .collect()  # TODO add validation for input data: correct format, not null, etc.
    )

    return BaseLoadMetrics(
        base_load_watts=metrics_df[0, "avg_watts"],
        daily_usage_kwh=metrics_df[0, "avg_daily_kwh"],
        base_percentage=metrics_df[0, "avg_percentage"],
    )


def main(file_path: str) -> BaseLoadMetrics:
    """Process energy data and return base load metrics"""
    lf = load_data(file_path)
    return calculate_base_load(lf)
