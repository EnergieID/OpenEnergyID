"""tests for main"""

from datetime import datetime
import polars as pl
from openenergyid.baseload.main import Granularity, analyze_base_load


def test_granularity_feature():
    """
    Test the analyze_base_load function with different granularities.

    This function creates a sample LazyFrame with test data and tests the
    analyze_base_load function with DAILY and HOURLY granularities. It verifies
    the shape of the resulting DataFrame and checks for the presence of the
    'base_load_watts' column.

    Test Cases:
    - DAILY granularity: Expects 2 rows in the result.
    - HOURLY granularity: Expects 6 rows in the result.

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

    # Test with DAILY granularity
    result_daily = analyze_base_load(lf, Granularity.P1D)
    print("Daily Granularity Result:")
    print(result_daily)
    assert result_daily.shape[0] == 2, "Expected 2 rows for DAILY granularity"
    assert "base_load_watts" in result_daily.columns, "Expected 'base_load_watts' column in result"

    # Test with HOURLY granularity
    result_hourly = analyze_base_load(lf, Granularity.PT1H)
    print("\nHourly Granularity Result:")
    print(result_hourly)
    assert result_hourly.shape[0] == 6, "Expected 6 rows for HOURLY granularity"
    assert "base_load_watts" in result_hourly.columns, "Expected 'base_load_watts' column in result"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_granularity_feature()
