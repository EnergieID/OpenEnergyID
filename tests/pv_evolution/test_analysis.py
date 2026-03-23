import pandas as pd
import pytest

from openenergyid import TimeDataFrame, const
from openenergyid.pv_evolution import LongTermPVAnalyzer, PVLongTermAnalysisInput


def _make_frame(
    start: str,
    periods: int,
    freq: str,
    production_factor: float = 2.0,
) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    radiation = pd.Series(range(1, periods + 1), index=index, dtype=float)
    production = radiation * production_factor
    return pd.DataFrame(
        {
            const.SOLAR_RADIATION: radiation,
            const.ELECTRICITY_PRODUCED: production,
        },
        index=index,
    )


def _make_monthly_frame(start: str, periods: int, production_factor: float = 2.0) -> pd.DataFrame:
    return _make_frame(start=start, periods=periods, freq="MS", production_factor=production_factor)


def _make_input_model(
    frame: pd.DataFrame,
    timezone: str = "Europe/Brussels",
    reference: str | None = None,
) -> PVLongTermAnalysisInput:
    tdf = TimeDataFrame.from_pandas(frame)
    return PVLongTermAnalysisInput(
        index=tdf.index,
        columns=tdf.columns,
        data=tdf.data,
        timezone=timezone,
        reference=reference,
    )


def test_long_term_analysis_happy_path_and_diagnostics() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=36, production_factor=2.0)

    # Add drift in the final year so yearly errors are not all zero.
    year_mask = frame.index.year == 2023  # type: ignore
    frame.loc[year_mask, const.ELECTRICITY_PRODUCED] = (
        frame.loc[year_mask, const.ELECTRICITY_PRODUCED].astype(float) * 1.1  # type: ignore
    )

    input_model = _make_input_model(frame)

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
    assert result_2023.relative_error > 0  # type: ignore


def test_first_incomplete_year_is_skipped() -> None:
    frame = _make_monthly_frame("2021-03-01", periods=34, production_factor=2.0)

    input_model = _make_input_model(frame)

    output = LongTermPVAnalyzer().analyze(input_model)

    years = [result.year for result in output.yearly_results]
    assert years == [2022, 2023]
    assert output.yearly_results[1].complete_year is True


def test_incomplete_non_first_year_is_retained() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=27, production_factor=2.0)

    input_model = _make_input_model(frame)

    output = LongTermPVAnalyzer().analyze(input_model)

    assert [result.year for result in output.yearly_results] == [2021, 2022, 2023]
    incomplete = [result for result in output.yearly_results if result.year == 2023][0]
    assert incomplete.complete_year is False


def test_missing_required_columns_raises() -> None:
    index = pd.date_range(start="2021-01-01", periods=12, freq="MS", tz="UTC")
    frame = pd.DataFrame({const.SOLAR_RADIATION: range(12)}, index=index)
    tdf = TimeDataFrame.from_pandas(frame)

    with pytest.raises(ValueError, match="Missing required columns"):
        PVLongTermAnalysisInput(
            index=tdf.index,
            columns=tdf.columns,
            data=tdf.data,
            timezone="Europe/Brussels",
        )


def test_daily_input_raises_duplicate_months_error() -> None:
    frame = _make_frame("2021-01-01", periods=12, freq="D")

    with pytest.raises(ValueError, match="exactly one row per calendar month"):
        _make_input_model(frame)


def test_quarterly_input_raises_missing_months_error() -> None:
    frame = _make_frame("2021-01-01", periods=12, freq="QS")

    with pytest.raises(ValueError, match="contiguous monthly data"):
        _make_input_model(frame)


def test_input_with_missing_month_raises() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=13).drop(pd.Timestamp("2021-06-01", tz="UTC"))

    with pytest.raises(ValueError, match="contiguous monthly data"):
        _make_input_model(frame)


def test_input_with_duplicate_month_raises() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=12)
    duplicate_month_row = pd.DataFrame(
        {
            const.SOLAR_RADIATION: [100.0],
            const.ELECTRICITY_PRODUCED: [200.0],
        },
        index=[pd.Timestamp("2021-01-15", tz="UTC")],
    )

    with pytest.raises(ValueError, match="exactly one row per calendar month"):
        _make_input_model(pd.concat([frame, duplicate_month_row]))


@pytest.mark.parametrize("column", [const.SOLAR_RADIATION, const.ELECTRICITY_PRODUCED])
def test_nan_in_required_columns_raises_clear_error(column: str) -> None:
    frame = _make_monthly_frame("2021-01-01", periods=12)
    frame.loc[frame.index[4], column] = float("nan")

    with pytest.raises(ValueError, match="Missing values found in required columns"):
        _make_input_model(frame)


def test_reference_period_needs_minimum_12_rows() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=11)

    with pytest.raises(ValueError, match="At least 12 monthly rows"):
        _make_input_model(frame)


def test_zero_actual_with_nonzero_prediction_has_null_relative_error() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=24, production_factor=2.0)
    frame.loc[frame.index.year == 2022, const.ELECTRICITY_PRODUCED] = 0.0  # type: ignore

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))

    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]
    assert result_2022.actual_production == pytest.approx(0.0)
    assert result_2022.predicted_production > 0
    assert result_2022.relative_error is None


def test_reference_dataset_sorts_unsorted_input() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=24).iloc[::-1]

    reference = LongTermPVAnalyzer()._reference_dataset(frame)

    assert list(reference.index) == list(frame.sort_index().index[:12])


def test_optional_reference_is_reflected_in_output() -> None:
    frame = _make_monthly_frame("2021-01-01", periods=12)

    input_model = _make_input_model(frame, reference="PV-SITE-123")

    output = LongTermPVAnalyzer().analyze(input_model)

    assert output.reference == "PV-SITE-123"
