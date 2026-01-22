"""Tests for nighttime-only baseload analysis feature."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from openenergyid.baseload.analysis import BaseloadAnalysisResult, BaseloadAnalyzer


class TestNighttimeFiltering:
    """Test nighttime filtering functionality."""

    @pytest.fixture
    def summer_week_data(self) -> pl.LazyFrame:
        """Generate a week of 15-minute power data simulating unmeasured PV.

        This simulates the problem scenario: a home with PV but no production meter.
        - True baseload: 200W (what we want to find)
        - Daytime import: artificially LOW because PV covers part of consumption
        - Nighttime import: shows true consumption (baseload + nighttime activities)
        """
        start = datetime(2024, 6, 20, 0, 0)
        timestamps = [start + timedelta(minutes=15 * i) for i in range(7 * 96)]

        # Simulate unmeasured PV scenario:
        # - True baseload is 200W (always running)
        # - Daytime: PV produces, so import shows LOWER than actual consumption
        # - Nighttime: No PV, import shows true consumption
        power_values = []
        for ts in timestamps:
            hour = ts.hour
            if 6 <= hour <= 21:
                # Daytime: PV covers most consumption, import appears LOW
                # This is the misleading data we want to filter out
                power_values.append(50.0)  # Artificially low due to PV
            else:
                # Nighttime: True consumption visible (baseload ~200W)
                power_values.append(200.0)

        return (
            pl.DataFrame({"timestamp": timestamps, "power": power_values})
            .with_columns(pl.col("timestamp").dt.replace_time_zone("Europe/Brussels"))
            .lazy()
        )

    @pytest.fixture
    def winter_week_data(self) -> pl.LazyFrame:
        """Generate a week of 15-minute power data in winter with unmeasured PV."""
        start = datetime(2024, 12, 20, 0, 0)
        timestamps = [start + timedelta(minutes=15 * i) for i in range(7 * 96)]

        # Same unmeasured PV scenario, but winter has shorter daylight hours
        power_values = []
        for ts in timestamps:
            hour = ts.hour
            if 9 <= hour <= 16:
                power_values.append(100.0)  # Daytime - PV reduces import (less than summer)
            else:
                power_values.append(200.0)  # Nighttime - true baseload visible

        return (
            pl.DataFrame({"timestamp": timestamps, "power": power_values})
            .with_columns(pl.col("timestamp").dt.replace_time_zone("Europe/Brussels"))
            .lazy()
        )

    def test_nighttime_only_default_false(self, summer_week_data):
        """Verify nighttime_only defaults to False."""
        analyzer = BaseloadAnalyzer(timezone="Europe/Brussels")
        assert analyzer.nighttime_only is False

    def test_nighttime_only_filters_daytime(self, summer_week_data):
        """Nighttime-only analysis should find true baseload when daytime has PV interference.

        In the unmeasured PV scenario:
        - Standard analysis sees the artificially LOW daytime values (50W) and uses those as baseload
        - Nighttime-only correctly ignores daytime and finds true baseload (200W)
        """
        # Standard analysis will be fooled by low daytime values
        analyzer_standard = BaseloadAnalyzer(timezone="Europe/Brussels", quantile=0.05)
        result_standard = analyzer_standard.analyze(summer_week_data, "1d")

        # Nighttime-only excludes misleading daytime data
        analyzer_night = BaseloadAnalyzer(
            timezone="Europe/Brussels", quantile=0.05, nighttime_only=True
        )
        result_night = analyzer_night.analyze(summer_week_data, "1d")

        # Standard analysis is fooled by low daytime values (expects ~50W)
        assert result_standard.global_median_baseload < 100

        # Nighttime-only finds the true baseload (200W)
        assert result_night.global_median_baseload > result_standard.global_median_baseload
        assert 195 <= result_night.global_median_baseload <= 210

    def test_default_location_is_brussels(self):
        """Verify default location is Brussels."""
        analyzer = BaseloadAnalyzer(timezone="Europe/Brussels", nighttime_only=True)
        assert analyzer.location == (50.85, 4.35)

    def test_custom_location(self):
        """Verify custom location is used."""
        amsterdam = (52.37, 4.90)
        analyzer = BaseloadAnalyzer(
            timezone="Europe/Amsterdam", nighttime_only=True, location=amsterdam
        )
        assert analyzer.location == amsterdam

    def test_solar_elevation_threshold_default(self):
        """Verify default solar elevation threshold is 0.0 (geometric sunset)."""
        analyzer = BaseloadAnalyzer(timezone="Europe/Brussels", nighttime_only=True)
        assert analyzer.solar_elevation_threshold == 0.0

    def test_civil_twilight_threshold(self, summer_week_data):
        """Civil twilight (-6°) should give fewer nighttime hours but similar baseload."""
        # Geometric sunset
        analyzer_geometric = BaseloadAnalyzer(
            timezone="Europe/Brussels",
            nighttime_only=True,
            solar_elevation_threshold=0.0,
        )
        result_geometric = analyzer_geometric.analyze(summer_week_data, "1d")

        # Civil twilight (more conservative)
        analyzer_civil = BaseloadAnalyzer(
            timezone="Europe/Brussels",
            nighttime_only=True,
            solar_elevation_threshold=-6.0,
        )
        result_civil = analyzer_civil.analyze(summer_week_data, "1d")

        # Both should find approximately the same baseload (~200W)
        # because both are filtering to nighttime-only data
        assert 195 <= result_geometric.global_median_baseload <= 210
        assert 195 <= result_civil.global_median_baseload <= 210

    def test_winter_longer_nights(self, winter_week_data):
        """Winter should have more nighttime readings than summer."""
        analyzer = BaseloadAnalyzer(timezone="Europe/Brussels", nighttime_only=True, quantile=0.05)
        result = analyzer.analyze(winter_week_data, "1d")

        # Should still find the 200W baseload
        assert 195 <= result.global_median_baseload <= 210


class TestBackwardCompatibility:
    """Verify backward compatibility when nighttime_only=False."""

    @pytest.fixture
    def simple_power_data(self) -> pl.LazyFrame:
        """Simple constant power data for testing."""
        start = datetime(2024, 1, 1, 0, 0)
        timestamps = [start + timedelta(minutes=15 * i) for i in range(96 * 7)]  # 7 days
        power_values = [300.0] * len(timestamps)  # Constant 300W

        return (
            pl.DataFrame({"timestamp": timestamps, "power": power_values})
            .with_columns(pl.col("timestamp").dt.replace_time_zone("Europe/Brussels"))
            .lazy()
        )

    def test_default_params_unchanged(self, simple_power_data):
        """Standard analysis with default params should work as before."""
        analyzer = BaseloadAnalyzer(timezone="Europe/Brussels", quantile=0.05)
        result = analyzer.analyze(simple_power_data, "1d")

        assert isinstance(result, BaseloadAnalysisResult)
        assert result.global_median_baseload == pytest.approx(300.0, rel=0.01)

    def test_nighttime_false_same_as_no_param(self, simple_power_data):
        """Explicitly setting nighttime_only=False should be same as not setting it."""
        analyzer_default = BaseloadAnalyzer(timezone="Europe/Brussels")
        analyzer_explicit = BaseloadAnalyzer(timezone="Europe/Brussels", nighttime_only=False)

        result_default = analyzer_default.analyze(simple_power_data, "1d")
        result_explicit = analyzer_explicit.analyze(simple_power_data, "1d")

        assert result_default.global_median_baseload == result_explicit.global_median_baseload


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_returns_empty_result(self):
        """Empty data should return empty result, not crash."""
        empty_lf = pl.LazyFrame(
            schema={
                "timestamp": pl.Datetime(time_zone="Europe/Brussels"),
                "power": pl.Float64,
            }
        )
        analyzer = BaseloadAnalyzer(timezone="Europe/Brussels", nighttime_only=True)
        result = analyzer.analyze(empty_lf, "1d")

        assert result.global_median_baseload == 0.0

    def test_all_daytime_data_returns_empty(self):
        """If all data is daytime (filtered out), should return empty result."""
        # Create data only during midday hours
        start = datetime(2024, 6, 21, 11, 0)  # Start at 11:00
        timestamps = [start + timedelta(minutes=15 * i) for i in range(16)]  # 4 hours
        power_values = [300.0] * len(timestamps)

        daytime_only = (
            pl.DataFrame({"timestamp": timestamps, "power": power_values})
            .with_columns(pl.col("timestamp").dt.replace_time_zone("Europe/Brussels"))
            .lazy()
        )

        analyzer = BaseloadAnalyzer(timezone="Europe/Brussels", nighttime_only=True)
        result = analyzer.analyze(daytime_only, "1d")

        # All data filtered out = empty result
        assert result.global_median_baseload == 0.0

    def test_nighttime_only_data_unchanged(self):
        """Data that's already all nighttime should give same result."""
        # Create data only during nighttime hours (2-4 AM)
        start = datetime(2024, 6, 21, 2, 0)
        timestamps = [start + timedelta(minutes=15 * i) for i in range(8)]  # 2 hours
        power_values = [250.0] * len(timestamps)

        nighttime_only_data = (
            pl.DataFrame({"timestamp": timestamps, "power": power_values})
            .with_columns(pl.col("timestamp").dt.replace_time_zone("Europe/Brussels"))
            .lazy()
        )

        analyzer_standard = BaseloadAnalyzer(timezone="Europe/Brussels")
        analyzer_night = BaseloadAnalyzer(timezone="Europe/Brussels", nighttime_only=True)

        result_standard = analyzer_standard.analyze(nighttime_only_data, "1d")
        result_night = analyzer_night.analyze(nighttime_only_data, "1d")

        # Both should find the same baseload since all data is already nighttime
        assert result_standard.global_median_baseload == pytest.approx(
            result_night.global_median_baseload, rel=0.01
        )

    def test_unsorted_input_data(self):
        """Unsorted input data should still work correctly.

        Regression test for sort order bug: the nighttime filter join
        doesn't preserve sort order, which breaks group_by_dynamic.
        """
        import random

        # Create nighttime data points in random order
        start = datetime(2024, 6, 21, 1, 0)  # 1 AM - definitely nighttime
        timestamps = [start + timedelta(minutes=15 * i) for i in range(24)]  # 6 hours
        power_values = [200.0] * len(timestamps)

        # Shuffle the data to simulate unsorted input
        combined = list(zip(timestamps, power_values))
        random.seed(42)
        random.shuffle(combined)
        shuffled_timestamps, shuffled_power = zip(*combined)

        unsorted_data = (
            pl.DataFrame({"timestamp": list(shuffled_timestamps), "power": list(shuffled_power)})
            .with_columns(pl.col("timestamp").dt.replace_time_zone("Europe/Brussels"))
            .lazy()
        )

        analyzer = BaseloadAnalyzer(timezone="Europe/Brussels", nighttime_only=True)
        # This would raise "argument in operation 'group_by_dynamic' is not sorted"
        # if the sort fix is missing
        result = analyzer.analyze(unsorted_data, "1d")

        assert 195 <= result.global_median_baseload <= 210
