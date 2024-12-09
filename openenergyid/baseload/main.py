"""
This module provides functionality for loading, validating, and analyzing energy usage data.

Classes:
    TimeFrame: An enumeration of different timeframes for data aggregation.
    BaseLoadMetrics: A NamedTuple container for base load analysis metrics.
    EnergySchema: A pandera DataFrameModel for validating energy usage data.

Functions:
    load_data(path: str) -> pl.LazyFrame:
        Loads and validates energy usage data from an NDJSON file.

    calculate_base_load(lf: pl.LazyFrame, granularity: Granularity = Granularity.DAILY) -> pl.DataFrame:
        Calculates base load metrics from energy usage data aggregated by the specified granularity.

    main(file_path: str, granularity: Granularity) -> pl.DataFrame:
        Processes energy data and returns base load metrics for the specified granularity.
"""

from typing import NamedTuple
import polars as pl
import pandera.polars as pa
from openenergyid.enums import Granularity
## VERY important to use pandera.polars instead of pandera to avoid pandas errors

# Map Granularity to polars format
GRANULARITY_TO_POLARS = {
    Granularity.PT15M: "15m",
    Granularity.PT1H: "1h",
    Granularity.P1D: "1d",
    Granularity.P1M: "1mo",
    Granularity.P1Y: "1y",
}


class BaseLoadMetrics(NamedTuple):
    """Container for base load analysis metrics"""

    base_load_watts: float  # Base load in watts
    usage_kwh: float  # Usage in kWh for the period
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


def calculate_base_load(
    lf: pl.LazyFrame, granularity: Granularity = Granularity.P1D
) -> pl.DataFrame:
    """Calculate base load metrics aggregated by specified granularity"""
    polars_interval = GRANULARITY_TO_POLARS[granularity]
    return (
        lf.filter(pl.col("total") >= 0)
        .sort("timestamp")
        .group_by_dynamic("timestamp", every=polars_interval)
        .agg(
            [
                pl.col("total").sum().alias("total_usage"),
                (pl.col("total").sort().head(10).mean() * 4 * 24).alias("base_load_kwh"),
                pl.col("timestamp").first().alias("period_start"),
            ]
        )
        .with_columns(
            [
                (pl.col("base_load_kwh") / pl.col("total_usage") * 100).alias("base_percentage"),
                (pl.col("base_load_kwh") * 1000 / 24).alias("base_load_watts"),
            ]
        )
        .sort("period_start")
        .collect()
    )


def main(file_path: str, granularity: Granularity) -> pl.DataFrame:
    """Process energy data and return base load metrics for specified granularity"""
    return calculate_base_load(load_data(file_path), granularity)


# Example usage:
if __name__ == "__main__":
    results = main("data/PP/energy_use_test1.ndjson", Granularity.P1M)
    print(results)
