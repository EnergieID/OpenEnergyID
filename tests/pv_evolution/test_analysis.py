import pandas as pd
import pytest

from openenergyid import TimeDataFrame, const
from openenergyid.pv_evolution import LongTermPVAnalyzer, PVLongTermAnalysisInput


def _make_monthly_frame(start: str, periods: int, production_factor: float = 2.0) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq="MS", tz="UTC")
    radiation = pd.Series(range(1, periods + 1), index=index, dtype=float)
    production = radiation * production_factor
    return pd.DataFrame(
        {
            const.SOLAR_RADIATION: radiation,
            const.ELECTRICITY_PRODUCED: production,
        },
        index=index,
    )


def test_long_term_analysis_happy_path_and_diagnostics() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=36, production_factor=2.0)

    # Add drift in the final year so yearly errors are not all zero.
    year_mask = frame.index.map(lambda timestamp: timestamp.year) == 2023
    frame.loc[year_mask, const.ELECTRICITY_PRODUCED] = (
        frame.loc[year_mask, const.ELECTRICITY_PRODUCED].astype(float) * 1.1
    )

    input_model = PVLongTermAnalysisInput(
        frame=TimeDataFrame.from_pandas(frame),
        timezone="Europe/Brussels",
    )

    output = LongTermPVAnalyzer().analyze(input_model)
    output_dump = output.model_dump()

    assert [result.year for result in output.yearly_results] == [2021, 2022, 2023]
    assert output_dump["regression_diagnostics"]["coefficient"] == pytest.approx(2.0)
    assert output_dump["regression_diagnostics"]["intercept"] == pytest.approx(0.0)
    assert output_dump["regression_diagnostics"]["r_squared"] == pytest.approx(1.0)

    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]
    assert result_2022.error == pytest.approx(0.0)
    assert result_2022.relative_error == pytest.approx(0.0)
    assert result_2022.complete_year is True

    result_2023 = [result for result in output.yearly_results if result.year == 2023][0]
    assert result_2023.error > 0
    assert result_2023.relative_error > 0


def test_first_incomplete_year_is_skipped() -> None:
    frame = _make_monthly_frame("2021-03-01", periods=34, production_factor=2.0)

    input_model = PVLongTermAnalysisInput(
        frame=TimeDataFrame.from_pandas(frame),
        timezone="Europe/Brussels",
    )

    output = LongTermPVAnalyzer().analyze(input_model)

    years = [result.year for result in output.yearly_results]
    assert years == [2022, 2023]
    assert output.yearly_results[1].complete_year is True


def test_incomplete_non_first_year_is_retained() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=27, production_factor=2.0)

    input_model = PVLongTermAnalysisInput(
        frame=TimeDataFrame.from_pandas(frame),
        timezone="Europe/Brussels",
    )

    output = LongTermPVAnalyzer().analyze(input_model)

    assert [result.year for result in output.yearly_results] == [2021, 2022, 2023]
    incomplete = [result for result in output.yearly_results if result.year == 2023][0]
    assert incomplete.complete_year is False


def test_missing_required_columns_raises() -> None:
    index = pd.date_range(start="2021-01-01", periods=12, freq="MS", tz="UTC")
    frame = pd.DataFrame({const.SOLAR_RADIATION: range(12)}, index=index)

    with pytest.raises(ValueError, match="Missing required columns"):
        PVLongTermAnalysisInput(
            frame=TimeDataFrame.from_pandas(frame),
            timezone="Europe/Brussels",
        )


def test_reference_period_needs_minimum_12_rows() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=11)

    with pytest.raises(ValueError, match="At least 12 monthly rows"):
        PVLongTermAnalysisInput(
            frame=TimeDataFrame.from_pandas(frame),
            timezone="Europe/Brussels",
        )
