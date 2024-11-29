"""Test the TimeFrame feature of the calculate_base_load function."""

from datetime import datetime
import polars as pl
from openenergyid.baseload.main import TimeFrame, calculate_base_load


def test_timeframe_feature():
    """
    Test the calculate_base_load function with different timeframes.

    This function creates a sample LazyFrame with test data and tests the
    calculate_base_load function with DAILY and HOURLY timeframes. It verifies
    the shape of the resulting DataFrame and checks for the presence of the
    'base_load_watts' column.

    Test Cases:
    - DAILY timeframe: Expects 2 rows in the result.
    - HOURLY timeframe: Expects 6 rows in the result.

    Raises:
        AssertionError: If the shape of the resulting DataFrame does not match
                        the expected number of rows or if the 'base_load_watts'
                        column is not present in the result.
    """
    # Create a sample LazyFrame with test data
    data = {
        "timestamp": [
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 1, 1, 1, 0),
            datetime(2023, 1, 1, 2, 0),
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 2, 1, 0),
            datetime(2023, 1, 2, 2, 0),
        ],
        "total": [10, 20, 30, 40, 50, 60],
    }
    lf = pl.LazyFrame(data)

    # Test with DAILY timeframe
    result_daily = calculate_base_load(lf, TimeFrame.DAILY)
    print("Daily Timeframe Result:")
    print(result_daily)
    assert result_daily.shape[0] == 2, "Expected 2 rows for DAILY timeframe"
    assert "base_load_watts" in result_daily.columns, "Expected 'base_load_watts' column in result"

    # Test with HOURLY timeframe
    result_hourly = calculate_base_load(lf, TimeFrame.HOURLY)
    print("\nHourly Timeframe Result:")
    print(result_hourly)
    assert result_hourly.shape[0] == 6, "Expected 6 rows for HOURLY timeframe"
    assert "base_load_watts" in result_hourly.columns, "Expected 'base_load_watts' column in result"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_timeframe_feature()
