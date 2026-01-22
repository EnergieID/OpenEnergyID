"""Baseload Power Consumption Analysis Module.

Provides tools for detecting continuous background power usage using Polars LazyFrames,
keeping transformations lazy to support larger time-series datasets.
"""

from dataclasses import dataclass

import pandas as pd
import polars as pl
import pvlib


@dataclass
class BaseloadAnalysisResult:
    """Complete result from baseload analysis.

    Attributes
    ----------
    results : pl.LazyFrame
        Per-period metrics (hourly, daily, monthly, etc.) with columns:
        - timestamp: Start of reporting period
        - consumption_due_to_baseload_in_kilowatthour
        - total_consumption_in_kilowatthour
        - consumption_not_due_to_baseload_in_kilowatthour
        - average_daily_baseload_in_watt
        - average_power_in_watt
        - baseload_ratio
        - consumption_due_to_median_baseload_in_kilowatthour

    global_median_baseload : float
        The overall median of daily baseload values across the entire period, in watts.

    monthly_median_baseloads : pl.LazyFrame
        Monthly aggregated median baseload values with columns:
        - timestamp: Start of month
        - monthly_median_baseload_in_watt: Median of daily baseload values for that month
    """

    results: pl.LazyFrame
    global_median_baseload: float
    monthly_median_baseloads: pl.LazyFrame


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
    timezone : str
        Timezone for analysis. All timestamps will be converted to this timezone
        to ensure correct daily boundaries and consistent reporting periods.

    quantile : float, default=0.05
        Defines what portion of lowest daily readings to consider as baseload.
        The default 0.05 (5%) corresponds to roughly 72 minutes of lowest
        consumption per day, which helps filter out brief power dips while
        capturing true baseload patterns.

    nighttime_only : bool, default=False
        If True, filter to only nighttime readings before analysis.
        Use this when PV production exists but is not metered, as daytime
        import readings will be artificially low due to unmeasured solar production.

    location : tuple[float, float] | None, default=None
        Latitude and longitude for solar position calculation, as (lat, lon).
        Only used when nighttime_only=True. If None, defaults to Brussels (50.85, 4.35).

    solar_elevation_threshold : float, default=0.0
        Sun elevation angle (degrees) below which is considered night.
        Only used when nighttime_only=True.
        - 0.0: Geometric sunset (sun at horizon) - recommended default
        - -6.0: Civil twilight (sky visibly dark)
        - -12.0: Nautical twilight (conservative)

    Example
    -------
    >>> analyzer = BaseloadAnalyzer(timezone="Europe/Brussels", quantile=0.05)
    >>> power_data = analyzer.prepare_power_series(energy_readings)
    >>> hourly_analysis, _ = analyzer.analyze(power_data, "1h")

    >>> # For homes with unmeasured PV production
    >>> analyzer = BaseloadAnalyzer(
    ...     timezone="Europe/Brussels",
    ...     nighttime_only=True,
    ... )
    """

    # Default location: Brussels, Belgium
    DEFAULT_LOCATION = (50.85, 4.35)

    def __init__(
        self,
        timezone: str,
        quantile: float = 0.05,
        nighttime_only: bool = False,
        location: tuple[float, float] | None = None,
        solar_elevation_threshold: float = 0.0,
    ):
        self.quantile = quantile
        self.timezone = timezone
        self.nighttime_only = nighttime_only
        self.location = location if location is not None else self.DEFAULT_LOCATION
        self.solar_elevation_threshold = solar_elevation_threshold

    def prepare_power_series(self, energy_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Converts energy readings into a power consumption time series.

        Transforms 15-minute energy readings (kilowatt-hours) into instantaneous
        power readings (watts) while handling timezone conversion. Input must be a
        Polars LazyFrame with columns:
            - timestamp: timezone-aware datetime (e.g. "2025-01-01T00:00:00+01:00")
            - total: kWh for the 15-minute interval
        Example row (kWh input):
            {"timestamp": "2025-01-01T00:00:00+01:00", "total": 0.031}

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
                    pl.col("timestamp")
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone(self.timezone)
                    .alias("timestamp"),
                    (pl.col("total") * 4000).alias("power"),
                ]
            )
            .drop("total")
            .filter(pl.col("timestamp").is_not_null() & pl.col("power").is_not_null())
            .with_columns(pl.col("power").clip(lower_bound=0))
            .sort("timestamp")
        )

    def analyze(
        self, power_lf: pl.LazyFrame, reporting_granularity: str = "1h"
    ) -> BaseloadAnalysisResult:
        """
        Analyze power consumption data to calculate baseload and total energy metrics.

        Accepts either prepared power readings (timestamp/power) or raw energy readings
        (timestamp/total). Calculates:
        - Daily baseload power using a percentile threshold
        - Energy consumption from baseload vs total consumption
        - Average power metrics
        - Global median baseload value for the entire period
        - Monthly median baseload values

        The analysis happens in three steps:
        1. Calculate the daily baseload power level using the configured percentile
        2. Join this daily baseload with the original power readings
        3. Aggregate the combined data into the requested reporting periods

        Parameters
        ----------
        power_lf : pl.LazyFrame
            Data with columns:
                - timestamp: Datetime in configured timezone
                - power: Power readings in watts
            Alternatively accepts:
                - timestamp and total (kWh/15min) and will convert to power.

        reporting_granularity : str, default="1h"
            Time period for aggregating results. Must be a valid Polars interval string
            like "1h", "1d", "1mo" etc.

        Returns
        -------
        BaseloadAnalysisResult
            A dataclass containing:
            - results: pl.LazyFrame with metrics per reporting period
            - global_median_baseload: The overall median baseload in watts
            - monthly_median_baseloads: pl.LazyFrame with monthly median baseload values
        """
        power_lf = self._ensure_power_frame(power_lf)
        power_lf = (
            power_lf.select(["timestamp", "power"])
            .filter(pl.col("timestamp").is_not_null() & pl.col("power").is_not_null())
            .with_columns(pl.col("power").clip(lower_bound=0))
            .sort("timestamp")
        )

        # Apply nighttime filter if enabled
        if self.nighttime_only:
            power_lf = self._apply_nighttime_filter(power_lf)

        row_count = power_lf.select(pl.len()).collect().item()
        if row_count == 0:
            return self._empty_result()

        daily_baseload = (
            power_lf.group_by_dynamic("timestamp", every="1d")
            .agg(
                pl.col("power")
                .filter(pl.col("power") > 0)
                .quantile(self.quantile)
                .alias("daily_baseload")
            )
            .filter(pl.col("daily_baseload").is_not_null())
        )

        baseload_count = daily_baseload.select(pl.len()).collect().item()
        if baseload_count == 0:
            return self._empty_result()

        global_median_baseload = (
            daily_baseload.select(pl.col("daily_baseload").median()).collect().item()
        ) or 0.0

        # Compute monthly median baseloads from daily values
        monthly_median_baseloads = daily_baseload.group_by_dynamic("timestamp", every="1mo").agg(
            pl.col("daily_baseload").median().alias("monthly_median_baseload_in_watt")
        )

        results = (
            power_lf.join_asof(daily_baseload, on="timestamp")
            .group_by_dynamic("timestamp", every=reporting_granularity)
            .agg(
                [
                    (pl.col("daily_baseload").fill_null(0).sum() * 0.25 / 1000).alias(
                        "consumption_due_to_baseload_in_kilowatthour"
                    ),
                    (pl.col("power").sum() * 0.25 / 1000).alias(
                        "total_consumption_in_kilowatthour"
                    ),
                    pl.col("daily_baseload").mean().alias("average_daily_baseload_in_watt"),
                    pl.col("power").mean().alias("average_power_in_watt"),
                    (pl.len() * 0.25 * global_median_baseload / 1000).alias(
                        "consumption_due_to_median_baseload_in_kilowatthour"
                    ),
                ]
            )
            # First: cap baseload to not exceed total consumption
            .with_columns(
                pl.min_horizontal(
                    pl.col("consumption_due_to_baseload_in_kilowatthour"),
                    pl.col("total_consumption_in_kilowatthour"),
                ).alias("consumption_due_to_baseload_in_kilowatthour")
            )
            # Second: calculate derived columns using the capped baseload
            .with_columns(
                [
                    (
                        pl.col("total_consumption_in_kilowatthour")
                        - pl.col("consumption_due_to_baseload_in_kilowatthour")
                    )
                    .clip(lower_bound=0)
                    .alias("consumption_not_due_to_baseload_in_kilowatthour"),
                    pl.when(pl.col("total_consumption_in_kilowatthour") > 0)
                    .then(
                        pl.col("consumption_due_to_baseload_in_kilowatthour")
                        / pl.col("total_consumption_in_kilowatthour")
                    )
                    .otherwise(None)
                    .alias("baseload_ratio"),
                ]
            )
        )

        return BaseloadAnalysisResult(
            results=results,
            global_median_baseload=global_median_baseload,
            monthly_median_baseloads=monthly_median_baseloads,
        )

    def _empty_result(self) -> BaseloadAnalysisResult:
        """Return an empty BaseloadAnalysisResult for edge cases with no valid data."""
        empty_results = pl.LazyFrame(
            schema={
                "timestamp": pl.Datetime(time_zone=self.timezone),
                "consumption_due_to_baseload_in_kilowatthour": pl.Float64,
                "total_consumption_in_kilowatthour": pl.Float64,
                "consumption_not_due_to_baseload_in_kilowatthour": pl.Float64,
                "average_daily_baseload_in_watt": pl.Float64,
                "average_power_in_watt": pl.Float64,
                "baseload_ratio": pl.Float64,
                "consumption_due_to_median_baseload_in_kilowatthour": pl.Float64,
            }
        )
        empty_monthly = pl.LazyFrame(
            schema={
                "timestamp": pl.Datetime(time_zone=self.timezone),
                "monthly_median_baseload_in_watt": pl.Float64,
            }
        )
        return BaseloadAnalysisResult(
            results=empty_results,
            global_median_baseload=0.0,
            monthly_median_baseloads=empty_monthly,
        )

    def _ensure_power_frame(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Ensure a LazyFrame contains power data; convert from energy if needed."""
        cols = set(frame.collect_schema().names())
        if {"timestamp", "power"} <= cols:
            return frame
        if {"timestamp", "total"} <= cols:
            return self.prepare_power_series(frame)
        raise ValueError(
            "Expected LazyFrame with columns 'timestamp' and 'power' or 'timestamp' and 'total'."
        )

    def _apply_nighttime_filter(self, power_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Filter power readings to only include nighttime hours.

        Uses pvlib to calculate solar position for each timestamp and filters
        to only include readings where the sun is below the configured threshold.

        Parameters
        ----------
        power_lf : pl.LazyFrame
            Power data with timestamp and power columns.

        Returns
        -------
        pl.LazyFrame
            Filtered power data containing only nighttime readings.
        """
        # Collect timestamps to calculate solar position
        # This requires materializing the timestamps, but solar calc is fast
        timestamps_df = power_lf.select("timestamp").collect()

        if timestamps_df.is_empty():
            return power_lf

        # Convert to pandas DatetimeIndex for pvlib
        timestamps_pd = pd.DatetimeIndex(timestamps_df["timestamp"].to_list())

        # Calculate solar position
        latitude, longitude = self.location
        solar_pos = pvlib.solarposition.get_solarposition(timestamps_pd, latitude, longitude)

        # Create mask for nighttime (elevation below threshold)
        elevation: pd.Series = solar_pos["elevation"]  # type: ignore[assignment]
        is_night = elevation < self.solar_elevation_threshold

        # Convert mask to polars and join back
        night_mask_df = pl.DataFrame(
            {"timestamp": timestamps_df["timestamp"], "is_night": is_night.to_list()}
        )

        # Filter to nighttime only
        return (
            power_lf.join(night_mask_df.lazy(), on="timestamp", how="inner")
            .filter(pl.col("is_night"))
            .drop("is_night")
        )
