"""Baseload Power Consumption Analysis Module

This module provides tools for analyzing electrical power consumption patterns to identify
and quantify baseload - the continuous background power usage in electrical systems.
It uses sophisticated time-series analysis to detect consistent minimum power draws
that represent always-on devices and systems.
"""

import polars as pl


class BaseloadAnalyzer:
    """Analyzes power consumption data to determine baseload characteristics.

    The BaseloadAnalyzer helps identify the minimum continuous power consumption in
    an electrical system by analyzing regular energy readings. It uses a statistical
    approach to determine baseload, which represents power used by devices that run
    continuously (like refrigerators, standby electronics, or network equipment).

    The analyzer works by:
    1. Converting 15-minute energy readings to instantaneous power values
    2. Analyzing daily patterns to identify consistent minimum usage
    3. Aggregating results into configurable time periods

    Parameters
    ----------
    quantile : float, default=0.05
        Defines what portion of lowest daily readings to consider as baseload.
        The default 0.05 (5%) corresponds to roughly 72 minutes of lowest
        consumption per day, which helps filter out brief power dips while
        capturing true baseload patterns.

    timezone : str
        Timezone for analysis. All timestamps will be converted to this timezone
        to ensure correct daily boundaries and consistent reporting periods.

    Example Usage
    ------------
    >>> analyzer = BaseloadAnalyzer(quantile=0.05)
    >>> power_data = analyzer.prepare_power_seriespolars(energy_readings)
    >>> hourly_analysis = analyzer.analyze(power_data, "1h")
    >>> monthly_analysis = analyzer.analyze(power_data, "1mo")
    """

    def __init__(self, timezone: str, quantile: float = 0.05):
        self.quantile = quantile
        self.timezone = timezone

    def prepare_power_seriespolars(self, energy_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Converts energy readings into a power consumption time series.

        Transforms 15-minute energy readings (kilowatt-hours) into instantaneous
        power readings (watts) while handling timezone conversion.

        Parameters
        ----------
        energy_lf : pl.LazyFrame
            Input energy data with columns:
            - timestamp: Datetime with timezone (e.g. "2023-01-01T00:00:00+01:00")
            - total: Energy readings in kilowatt-hours (kWh)

        Returns
        -------
        pl.LazyFrame
            Power series with columns:
            - timestamp: Timezone-adjusted timestamps
            - power: Power readings in watts

        Notes
        -----
        The conversion from kWh/15min to watts uses the formula:
            watts = kWh * 4000
        where:
        - Multiply by 4 to convert from 15-minute to hourly rate
        - Multiply by 1000 to convert from kilowatts to watts
        """
        return (
            energy_lf.with_columns(
                [
                    # Convert timezone
                    pl.col("timestamp")
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone(self.timezone)
                    .alias("timestamp"),
                    # Convert to watts and clip negative values
                    (pl.col("total") * 4000).clip(0).alias("power"),
                ]
            )
            .drop("total")
            .sort("timestamp")
        )

    def analyze(self, power_lf: pl.LazyFrame, reporting_granularity: str = "1h") -> pl.LazyFrame:
        """Analyze power consumption data to calculate baseload and total energy metrics.

        Takes power readings (in watts) with 15-minute intervals and calculates:
        - Daily baseload power using a percentile threshold
        - Energy consumption from baseload vs total consumption
        - Average power metrics

        The analysis happens in three steps:
        1. Calculate the daily baseload power level using the configured percentile
        2. Join this daily baseload with the original power readings
        3. Aggregate the combined data into the requested reporting periods

        Parameters
        ----------
        power_lf : pl.LazyFrame
            Power consumption data with columns:
            - timestamp: Datetime in configured timezone
            - power: Power readings in watts

        reporting_granularity : str, default="1h"
            Time period for aggregating results. Must be a valid Polars interval string
            like "1h", "1d", "1mo" etc.

        Returns
        -------
        pl.LazyFrame
            Analysis results with metrics per reporting period:
            - timestamp: Start of reporting period
            - consumption_due_to_baseload_in_kilowatthour: Baseload energy
            - total_consumption_in_kilowatthour: Total energy
            - consumption_not_due_to_baseload_in_kilowatthour: Non-baseload energy
            - average_daily_baseload_in_watt: Average baseload power level
            - average_power_in_watt: Average total power
            - baseload_ratio: Fraction of energy from baseload
        """
        # Step 1: Calculate the daily baseload level
        # Group power readings by day and find the threshold power level that represents baseload
        daily_baseload = power_lf.group_by_dynamic("timestamp", every="1d").agg(
            pl.col("power").quantile(self.quantile).alias("daily_baseload")
        )

        # Step 2 & 3: Join baseload data and aggregate metrics
        return (
            # Join the daily baseload level with original power readings
            # Using asof join since baseload changes daily but readings are every 15min
            power_lf.join_asof(daily_baseload, on="timestamp")
            # Group into requested reporting periods
            .group_by_dynamic("timestamp", every=reporting_granularity)
            .agg(
                [
                    # Energy calculations:
                    # Each 15min power reading (watts) represents 0.25 hours
                    # Convert to kWh: watts * 0.25h * (1kW/1000W)
                    (pl.col("daily_baseload").sum() * 0.25 / 1000).alias(
                        "consumption_due_to_baseload_in_kilowatthour"
                    ),
                    (pl.col("power").sum() * 0.25 / 1000).alias(
                        "total_consumption_in_kilowatthour"
                    ),
                    # Average power levels during the period
                    pl.col("daily_baseload").mean().alias("average_daily_baseload_in_watt"),
                    pl.col("power").mean().alias("average_power_in_watt"),
                ]
            )
            # Calculate derived metrics
            .with_columns(
                [
                    # Energy consumed above baseload level
                    (
                        pl.col("total_consumption_in_kilowatthour")
                        - pl.col("consumption_due_to_baseload_in_kilowatthour")
                    ).alias("consumption_not_due_to_baseload_in_kilowatthour"),
                    # What fraction of total energy was from baseload
                    (
                        pl.col("consumption_due_to_baseload_in_kilowatthour")
                        / pl.col("total_consumption_in_kilowatthour")
                    ).alias("baseload_ratio"),
                ]
            )
        )
