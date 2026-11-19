import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from openenergyid import TimeDataFrame, const
from openenergyid.pv_evolution import (
    LongTermPVAnalyzer,
    PVLongTermAnalysisInput,
    PVWarningCode,
)


def _make_daily_frame(
    start: str,
    periods: int,
    production_factor: float = 2.0,
) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq="D", tz="UTC")
    radiation = pd.Series(range(1, periods + 1), index=index, dtype=float)
    production = radiation * production_factor
    return pd.DataFrame(
        {
            const.SOLAR_RADIATION: radiation,
            const.ELECTRICITY_PRODUCED: production,
        },
        index=index,
    )


def _make_input_model(
    frame: pd.DataFrame,
    timezone: str = "Europe/Brussels",
    reference: str | None = None,
    baseline_start: dt.date | None = None,
    baseline_end: dt.date | None = None,
) -> PVLongTermAnalysisInput:
    tdf = TimeDataFrame.from_pandas(frame)
    return PVLongTermAnalysisInput(
        index=tdf.index,
        columns=tdf.columns,
        data=tdf.data,
        timezone=timezone,
        reference=reference,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )


def test_long_term_analysis_happy_path_and_diagnostics() -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 3, production_factor=2.0)

    year_mask = frame.index.year == 2023  # type: ignore[assignment]
    frame.loc[year_mask, const.ELECTRICITY_PRODUCED] = (
        frame.loc[year_mask, const.ELECTRICITY_PRODUCED].astype(float) * 1.1  # type: ignore[index]
    )

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))
    output_dump = output.model_dump()

    assert output.baseline_period.start == dt.date(2021, 1, 1)
    assert output.baseline_period.end == dt.date(2021, 12, 31)
    assert [result.year for result in output.yearly_results] == [2022, 2023]
    assert output_dump["regression_diagnostics"]["coefficient"] == pytest.approx(2.0)
    assert output_dump["regression_diagnostics"]["intercept"] == pytest.approx(0.0)
    assert output_dump["regression_diagnostics"]["r_squared"] == pytest.approx(1.0)

    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]
    assert result_2022.error == pytest.approx(0.0, abs=1e-9)
    assert result_2022.relative_error == pytest.approx(0.0, abs=1e-9)
    assert result_2022.complete_year is True

    result_2023 = [result for result in output.yearly_results if result.year == 2023][0]
    assert result_2023.error > 0
    # 10% above the prediction is +10%, measured against the prediction.
    assert result_2023.relative_error == pytest.approx(0.1)


def test_default_baseline_starts_at_first_available_day_and_skips_first_partial_year() -> None:
    frame = _make_daily_frame("2021-03-15", periods=365 * 2, production_factor=2.0)

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))

    assert output.baseline_period.start == dt.date(2021, 3, 15)
    assert output.baseline_period.end == dt.date(2022, 3, 14)
    assert [result.year for result in output.yearly_results] == [2022, 2023]
    assert output.yearly_results[0].complete_year is True
    assert output.yearly_results[1].complete_year is False


def test_reference_year_is_skipped_when_it_is_the_first_reporting_year() -> None:
    frame = _make_daily_frame("2020-07-01", periods=365 * 3, production_factor=2.0)

    output = LongTermPVAnalyzer().analyze(
        _make_input_model(
            frame,
            baseline_start=dt.date(2021, 1, 1),
            baseline_end=dt.date(2021, 12, 31),
        )
    )

    assert [result.year for result in output.yearly_results] == [2022, 2023]


def test_explicit_irregular_baseline_uses_requested_period() -> None:
    analyzer = LongTermPVAnalyzer()
    frame = _make_daily_frame("2020-01-01", periods=365 * 4, production_factor=2.0)
    input_model = _make_input_model(
        frame,
        baseline_start=dt.date(2021, 2, 10),
        baseline_end=dt.date(2023, 2, 9),
    )

    prepared = analyzer._prepare_daily_frame(input_model)
    reference = analyzer._reference_dataset(
        prepared,
        baseline_start=input_model.baseline_start,  # type: ignore[arg-type]
        baseline_end=input_model.baseline_end,  # type: ignore[arg-type]
    )

    assert analyzer._choose_bucket_count(730) == 24
    assert len(reference) == 24
    assert reference.index[0].date() == dt.date(2021, 2, 10)


def test_bucket_sizes_are_balanced_and_month_like() -> None:
    analyzer = LongTermPVAnalyzer()
    bucket_count = analyzer._choose_bucket_count(730)
    bucket_sizes = analyzer._bucket_sizes(730, bucket_count)

    assert bucket_count == 24
    assert min(bucket_sizes) >= 25
    assert max(bucket_sizes) <= 35
    assert max(bucket_sizes) - min(bucket_sizes) <= 1


def test_exact_one_year_complete_baseline_uses_calendar_month_resampling() -> None:
    analyzer = LongTermPVAnalyzer()
    frame = _make_daily_frame("2021-03-15", periods=365 * 2, production_factor=2.0)
    input_model = _make_input_model(frame)

    prepared = analyzer._prepare_daily_frame(input_model)
    reference = analyzer._reference_dataset(
        prepared,
        baseline_start=dt.date(2021, 3, 15),
        baseline_end=dt.date(2022, 3, 14),
    )

    expected_index = pd.date_range("2021-03-01", "2022-03-01", freq="MS", tz="Europe/Brussels")
    assert reference.index.equals(expected_index)
    assert len(reference) == 13


def test_exact_one_year_incomplete_baseline_falls_back_to_balanced_buckets() -> None:
    analyzer = LongTermPVAnalyzer()
    frame = _make_daily_frame("2021-03-15", periods=365 * 2, production_factor=2.0)
    frame.loc[pd.Timestamp("2021-09-01", tz="UTC"), const.ELECTRICITY_PRODUCED] = float("nan")
    input_model = _make_input_model(frame)

    prepared = analyzer._prepare_daily_frame(input_model)
    reference = analyzer._reference_dataset(
        prepared,
        baseline_start=dt.date(2021, 3, 15),
        baseline_end=dt.date(2022, 3, 14),
    )

    month_starts = pd.date_range("2021-03-01", "2022-03-01", freq="MS", tz="Europe/Brussels")
    assert not reference.index.equals(month_starts)
    assert any(timestamp.day != 1 for timestamp in reference.index[1:])
    assert len(reference) == analyzer._choose_bucket_count(364)


@pytest.mark.parametrize("column", [const.SOLAR_RADIATION, const.ELECTRICITY_PRODUCED])
def test_baseline_nan_days_are_dropped_before_bucket_aggregation(column: str) -> None:
    analyzer = LongTermPVAnalyzer()
    frame = _make_daily_frame("2021-01-01", periods=365 * 3, production_factor=2.0)
    frame.loc[pd.date_range("2021-06-01", periods=20, freq="D", tz="UTC"), column] = float("nan")

    input_model = _make_input_model(frame)
    prepared = analyzer._prepare_daily_frame(input_model)
    reference = analyzer._reference_dataset(
        prepared,
        baseline_start=dt.date(2021, 1, 1),
        baseline_end=dt.date(2021, 12, 31),
    )
    output = analyzer.analyze(input_model)

    assert len(reference) == analyzer._choose_bucket_count(365 - 20) == 11
    assert output.regression_diagnostics.coefficient == pytest.approx(2.0)


def test_outside_baseline_production_nan_is_zero_filled_in_yearly_actuals() -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)
    missing_day = pd.Timestamp("2022-05-17", tz="UTC")
    original_value = float(frame.loc[missing_day, const.ELECTRICITY_PRODUCED])
    frame.loc[missing_day, const.ELECTRICITY_PRODUCED] = float("nan")

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))
    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]

    expected_actual = float(
        frame.loc[frame.index.year == 2022, const.ELECTRICITY_PRODUCED].fillna(0.0).sum()  # type: ignore[index]
    )

    assert result_2022.actual_production == pytest.approx(expected_actual)
    assert result_2022.actual_production == pytest.approx(
        float(
            _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)
            .loc[lambda value: value.index.year == 2022, const.ELECTRICITY_PRODUCED]
            .sum()
        )
        - original_value
    )


def test_missing_solar_day_is_dropped_from_actual_and_prediction() -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)
    missing_day = pd.Timestamp("2022-06-03", tz="UTC")
    missing_radiation = float(frame.loc[missing_day, const.SOLAR_RADIATION])
    original_production = float(frame.loc[missing_day, const.ELECTRICITY_PRODUCED])
    frame.loc[missing_day, const.SOLAR_RADIATION] = float("nan")

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))
    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]

    expected_actual = float(
        frame.loc[frame.index.year == 2022, const.ELECTRICITY_PRODUCED]
        .drop(index=missing_day)
        .sum()
    )
    expected_prediction = float(
        2.0 * frame.loc[frame.index.year == 2022, const.SOLAR_RADIATION].dropna().sum()  # type: ignore[index]
    )

    assert result_2022.actual_production == pytest.approx(expected_actual)
    assert result_2022.predicted_production == pytest.approx(expected_prediction)
    assert result_2022.actual_production == pytest.approx(
        float(
            _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)
            .loc[lambda value: value.index.year == 2022, const.ELECTRICITY_PRODUCED]
            .sum()
        )
        - original_production
    )
    assert result_2022.predicted_production == pytest.approx(
        float(
            _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)
            .loc[lambda value: value.index.year == 2022, const.SOLAR_RADIATION]
            .sum()
            * 2.0
        )
        - (missing_radiation * 2.0)
    )


def test_bucket_count_has_no_upper_limit_for_longer_baselines() -> None:
    analyzer = LongTermPVAnalyzer()
    assert analyzer._choose_bucket_count(365 * 2) == 24


def test_explicit_baseline_that_cannot_be_partitioned_raises() -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)

    with pytest.raises(ValueError, match="balanced month-like buckets"):
        LongTermPVAnalyzer().analyze(
            _make_input_model(
                frame,
                baseline_start=dt.date(2021, 1, 1),
                baseline_end=dt.date(2021, 5, 29),
            )
        )


def test_too_few_valid_baseline_days_after_dropping_nans_raises_clear_error() -> None:
    baseline_start = dt.date(2021, 1, 1)
    baseline_end = dt.date(2021, 6, 30)
    frame = _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)

    # 181 baseline days minus 33 missing days leaves 148 valid days, which is
    # insufficient for 6 balanced month-like buckets of at least 25 days each.
    missing_mask = (frame.index.date >= baseline_start) & (
        frame.index.date <= baseline_start + dt.timedelta(days=32)
    )
    frame.loc[missing_mask, [const.SOLAR_RADIATION, const.ELECTRICITY_PRODUCED]] = float("nan")

    with pytest.raises(ValueError, match="enough valid days"):
        LongTermPVAnalyzer().analyze(
            _make_input_model(
                frame,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
            )
        )


def test_duplicate_local_day_raises() -> None:
    frame = pd.DataFrame(
        {
            const.SOLAR_RADIATION: [10.0, 11.0],
            const.ELECTRICITY_PRODUCED: [20.0, 22.0],
        },
        index=[
            pd.Timestamp("2021-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2021-01-01 12:00:00", tz="UTC"),
        ],
    )

    with pytest.raises(ValueError, match="exactly one row per local calendar day"):
        _make_input_model(frame)


def test_missing_local_day_raises() -> None:
    frame = _make_daily_frame("2021-01-01", periods=200).drop(pd.Timestamp("2021-04-15", tz="UTC"))

    with pytest.raises(ValueError, match="contiguous daily data"):
        _make_input_model(frame)


def test_monthly_input_is_rejected() -> None:
    index = pd.date_range(start="2021-01-01", periods=12, freq="MS", tz="UTC")
    radiation = pd.Series(range(1, 13), index=index, dtype=float)
    production = radiation * 2.0
    frame = pd.DataFrame(
        {
            const.SOLAR_RADIATION: radiation,
            const.ELECTRICITY_PRODUCED: production,
        },
        index=index,
    )

    with pytest.raises(ValueError, match="contiguous daily data"):
        _make_input_model(frame)


def test_explicit_baseline_must_be_fully_covered() -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 2)

    with pytest.raises(ValueError, match="fully covered"):
        _make_input_model(
            frame,
            baseline_start=dt.date(2020, 12, 1),
            baseline_end=dt.date(2021, 12, 1),
        )


def test_input_must_cover_default_twelve_month_baseline() -> None:
    frame = _make_daily_frame("2021-01-01", periods=149)

    with pytest.raises(ValueError, match="default 12-month baseline"):
        _make_input_model(frame)


def test_zero_actual_with_nonzero_prediction_is_minus_one() -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)
    frame.loc[frame.index.year == 2022, const.ELECTRICITY_PRODUCED] = 0.0  # type: ignore[index]

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))

    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]
    assert result_2022.actual_production == pytest.approx(0.0)
    assert result_2022.predicted_production > 0
    assert result_2022.relative_error == pytest.approx(-1.0)


def test_relative_error_is_bounded_below_by_minus_one() -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)
    # A year with a long outage: only the first ten days produce.
    outage = (frame.index.year == 2022) & (frame.index.dayofyear > 10)  # type: ignore[attr-defined]
    frame.loc[outage, const.ELECTRICITY_PRODUCED] = 0.0  # type: ignore[index]

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))

    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]
    assert result_2022.actual_production > 0
    assert -1.0 < result_2022.relative_error < -0.9  # type: ignore[operator]


@pytest.mark.parametrize(
    ("actual_factor", "expected"),
    [(0.0, 0.0), (1.0, None)],
    ids=["no-production", "production"],
)
def test_zero_prediction_has_no_ratio(actual_factor: float, expected: float | None) -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)
    # An irradiance sensor stuck at zero: the only way to predict nothing, since the
    # fit has no intercept and non-negative weights.
    frame.loc[frame.index.year == 2022, const.SOLAR_RADIATION] = 0.0  # type: ignore[index]
    frame.loc[frame.index.year == 2022, const.ELECTRICITY_PRODUCED] *= actual_factor  # type: ignore[index]

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))

    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]
    assert result_2022.predicted_production == 0
    assert result_2022.relative_error == expected


def test_optional_reference_and_api_aliases_are_reflected_in_output() -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 2)

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame, reference="PV-SITE-123"))
    output_dump = output.model_dump(by_alias=True)

    assert output.reference == "PV-SITE-123"
    assert output_dump["baselinePeriod"] == {
        "start": dt.date(2021, 1, 1),
        "end": dt.date(2021, 12, 31),
    }
    assert "yearlyResults" in output_dump
    assert "regressionDiagnostics" in output_dump


# Two irradiance series: tilted (35° south) and horizontal.


def _make_two_irradiance_frame(
    start: str,
    periods: int,
    tilted_weight: float,
    horizontal_weight: float,
) -> pd.DataFrame:
    """Daily frame with seasonal tilted and horizontal irradiance.

    Horizontal irradiance swings more over the year than tilted irradiance, as in
    reality: a south-facing tilt gains most in winter. Production is an exact
    weighted sum of the two.
    """
    index = pd.date_range(start=start, periods=periods, freq="D", tz="UTC")
    season = (1 - np.cos(2 * np.pi * (index.dayofyear - 1) / 365.25)) / 2  # 0 in Jan, 1 in Jul
    tilted = pd.Series(2000.0 + 3000.0 * season, index=index)
    horizontal = pd.Series(500.0 + 5000.0 * season, index=index)
    return pd.DataFrame(
        {
            const.SOLAR_RADIATION: tilted,
            const.SOLAR_RADIATION_HORIZONTAL: horizontal,
            const.ELECTRICITY_PRODUCED: tilted_weight * tilted + horizontal_weight * horizontal,
        },
        index=index,
    )


def test_two_irradiance_weights_are_recovered() -> None:
    frame = _make_two_irradiance_frame(
        "2021-01-01", periods=365 * 2, tilted_weight=0.6, horizontal_weight=0.9
    )

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))
    diagnostics = output.regression_diagnostics

    assert diagnostics.coefficient == pytest.approx(0.6)
    assert diagnostics.coefficient_horizontal == pytest.approx(0.9)
    assert diagnostics.intercept == pytest.approx(0.0)
    assert diagnostics.r_squared == pytest.approx(1.0)
    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]
    assert result_2022.relative_error == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("tilted_weight", "horizontal_weight"),
    [(2.0, 0.0), (0.0, 2.0)],
    ids=["south-facing", "flat-or-east-west"],
)
def test_two_irradiance_weights_are_never_negative(
    tilted_weight: float, horizontal_weight: float
) -> None:
    frame = _make_two_irradiance_frame(
        "2021-01-01",
        periods=365 * 2,
        tilted_weight=tilted_weight,
        horizontal_weight=horizontal_weight,
    )

    diagnostics = LongTermPVAnalyzer().analyze(_make_input_model(frame)).regression_diagnostics

    assert diagnostics.coefficient == pytest.approx(tilted_weight, abs=1e-6)
    assert diagnostics.coefficient_horizontal == pytest.approx(horizontal_weight, abs=1e-6)
    assert diagnostics.coefficient >= 0
    assert diagnostics.coefficient_horizontal >= 0  # type: ignore[operator]


def test_noisy_collinear_irradiance_gets_no_negative_weight() -> None:
    # Clouds move both series together, so they are nearly collinear, and production
    # follows the tilted series with 15% daily noise. On this reference year an
    # unconstrained fit trades a negative horizontal weight against a larger tilted one.
    rng = np.random.default_rng(5)
    index = pd.date_range("2021-01-01", periods=365 * 2, freq="D", tz="UTC")
    season = (1 - np.cos(2 * np.pi * (index.dayofyear - 1) / 365.25)) / 2
    cloud = rng.uniform(0.3, 1.0, len(index))
    tilted = (2000.0 + 3000.0 * season) * cloud
    horizontal = (500.0 + 5000.0 * season) * cloud
    frame = pd.DataFrame(
        {
            const.SOLAR_RADIATION: tilted,
            const.SOLAR_RADIATION_HORIZONTAL: horizontal,
            const.ELECTRICITY_PRODUCED: tilted * (1 + rng.normal(0, 0.15, len(index))),
        },
        index=index,
    )

    local = frame.tz_convert("Europe/Brussels")
    local.index = local.index.normalize()
    reference = local.loc["2021-01-01":"2021-12-31"].resample("MS").sum()
    predictors = [const.SOLAR_RADIATION, const.SOLAR_RADIATION_HORIZONTAL]
    unconstrained = LinearRegression(fit_intercept=False).fit(
        reference[predictors], reference[const.ELECTRICITY_PRODUCED]
    )
    assert unconstrained.coef_[1] < -0.05

    diagnostics = LongTermPVAnalyzer().analyze(_make_input_model(frame)).regression_diagnostics

    assert diagnostics.coefficient_horizontal == 0
    assert diagnostics.coefficient == pytest.approx(1.0, abs=0.01)


def test_horizontal_irradiance_removes_seasonal_bias_of_incomplete_year() -> None:
    # A system that follows horizontal irradiance, observed until the end of February.
    frame = _make_two_irradiance_frame(
        "2021-01-01", periods=365 + 31 + 28, tilted_weight=0.0, horizontal_weight=1.0
    )
    tilted_only = frame.drop(columns=[const.SOLAR_RADIATION_HORIZONTAL])

    with_horizontal = LongTermPVAnalyzer().analyze(_make_input_model(frame))
    without_horizontal = LongTermPVAnalyzer().analyze(_make_input_model(tilted_only))

    ytd_with = with_horizontal.yearly_results[-1]
    ytd_without = without_horizontal.yearly_results[-1]
    assert ytd_with.year == ytd_without.year == 2022
    assert ytd_with.complete_year is False
    # The tilted series overweights winter, so the winter months read as low.
    assert ytd_without.relative_error < -0.2  # type: ignore[operator]
    assert ytd_with.relative_error == pytest.approx(0.0, abs=1e-6)


def test_day_missing_horizontal_irradiance_is_dropped() -> None:
    frame = _make_two_irradiance_frame(
        "2021-01-01", periods=365 * 2, tilted_weight=0.6, horizontal_weight=0.9
    )
    missing_day = pd.Timestamp("2022-06-03", tz="UTC")
    frame.loc[missing_day, const.SOLAR_RADIATION_HORIZONTAL] = float("nan")

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))
    result_2022 = [result for result in output.yearly_results if result.year == 2022][0]

    expected_actual = float(
        frame.loc[frame.index.year == 2022, const.ELECTRICITY_PRODUCED]
        .drop(index=missing_day)
        .sum()
    )
    assert result_2022.actual_production == pytest.approx(expected_actual)
    assert result_2022.predicted_production == pytest.approx(expected_actual)
    assert result_2022.complete_year is False
    (warning,) = output.warnings
    assert warning.code == PVWarningCode.HORIZONTAL_IRRADIANCE_INCOMPLETE
    assert warning.context["days_per_year"] == {2022: 1}


def test_horizontal_series_starting_later_warns_about_the_years_it_drops() -> None:
    frame = _make_two_irradiance_frame(
        "2020-01-01", periods=366 + 365 * 2, tilted_weight=0.6, horizontal_weight=0.9
    )
    # The horizontal series only starts in 2021: 2020 cannot be used with both.
    frame.loc[frame.index.year == 2020, const.SOLAR_RADIATION_HORIZONTAL] = float("nan")  # type: ignore[index]

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))

    assert output.baseline_period.start == dt.date(2021, 1, 1)
    (warning,) = output.warnings
    assert warning.code == PVWarningCode.HORIZONTAL_IRRADIANCE_INCOMPLETE
    assert warning.context["days"] == 366
    assert warning.context["days_per_year"] == {2020: 366}
    assert "2020: 366" in warning.message


def test_complete_input_has_no_warnings() -> None:
    two_series = _make_two_irradiance_frame(
        "2021-01-01", periods=365 * 2, tilted_weight=0.6, horizontal_weight=0.9
    )
    tilted_only = _make_daily_frame("2021-01-01", periods=365 * 2)

    assert LongTermPVAnalyzer().analyze(_make_input_model(two_series)).warnings == []
    assert LongTermPVAnalyzer().analyze(_make_input_model(tilted_only)).warnings == []


def test_empty_horizontal_column_falls_back_to_tilted_only() -> None:
    frame = _make_daily_frame("2021-01-01", periods=365 * 2, production_factor=2.0)
    reference = LongTermPVAnalyzer().analyze(_make_input_model(frame))
    frame[const.SOLAR_RADIATION_HORIZONTAL] = float("nan")

    output = LongTermPVAnalyzer().analyze(_make_input_model(frame))

    assert output.regression_diagnostics.coefficient == pytest.approx(2.0)
    assert output.regression_diagnostics.coefficient_horizontal is None
    assert output.yearly_results == reference.yearly_results


def test_horizontal_coefficient_serialises_with_api_alias() -> None:
    frame = _make_two_irradiance_frame(
        "2021-01-01", periods=365 * 2, tilted_weight=0.6, horizontal_weight=0.9
    )

    dumped = LongTermPVAnalyzer().analyze(_make_input_model(frame)).model_dump(by_alias=True)

    assert dumped["regressionDiagnostics"]["coefficientHorizontal"] == pytest.approx(0.9)
    tilted_only = _make_daily_frame("2021-01-01", periods=365 * 2)
    dumped = LongTermPVAnalyzer().analyze(_make_input_model(tilted_only)).model_dump(by_alias=True)
    assert dumped["regressionDiagnostics"]["coefficientHorizontal"] is None
