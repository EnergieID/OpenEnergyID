"""Long-term PV evolution analysis implementation.

The method is described in ``docs/specs/pv-long-term-evolution.md``.
"""

import datetime as dt

import pandas as pd

from openenergyid import const
from openenergyid.utils import (
    fit_linear_regression,
    linear_regression_coefficients,
    linear_regression_diagnostics,
    predict_linear_regression,
)

from .models import (
    PVBaselinePeriod,
    PVLongTermAnalysisInput,
    PVLongTermAnalysisOutput,
    PVRegressionDiagnostics,
    PVYearResult,
)


class LongTermPVAnalyzer:
    """Analyze long-term PV production using a month-like baseline regression model.

    Production in a reference (baseline) period is regressed on irradiance, summed
    per month-like bucket, without intercept and with non-negative coefficients.
    Each calendar year is then predicted from its own monthly irradiance and
    compared with the actual production.

    The predictor is `solar_radiation` (tilted-plane irradiance). When the input
    also carries `solar_radiation_horizontal`, both series are used, so each
    system gets a non-negative mix of the two. The mix lets the model follow the
    system's own seasonal profile, which a single tilted series cannot, and
    removes most of the seasonal bias of an incomplete year. See
    ``docs/specs/pv-long-term-evolution.md``.
    """

    minimum_bucket_days = 25
    maximum_bucket_days = 35
    minimum_usable_buckets = 6
    target_bucket_days = 30.4375

    def _prepare_daily_frame(self, input_data: PVLongTermAnalysisInput) -> pd.DataFrame:
        """Normalize validated input into daily local-midnight rows."""
        frame = input_data.to_pandas(timezone=input_data.timezone).sort_index().copy()
        frame.index = frame.index.normalize()  # type: ignore
        return frame

    def _predictor_columns(self, frame: pd.DataFrame) -> list[str]:
        """Return the irradiance columns used as regression predictors.

        The horizontal series is used only when present with at least one value;
        otherwise the model falls back to the single tilted series.
        """
        if (
            const.SOLAR_RADIATION_HORIZONTAL in frame.columns
            and frame[const.SOLAR_RADIATION_HORIZONTAL].notna().any()
        ):
            return [const.SOLAR_RADIATION, const.SOLAR_RADIATION_HORIZONTAL]
        return [const.SOLAR_RADIATION]

    def _default_baseline_period(self, first_day: dt.date) -> tuple[dt.date, dt.date]:
        """Return the default first-12-month baseline period."""
        baseline_end = (
            pd.Timestamp(first_day) + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        ).date()
        return first_day, baseline_end

    def _is_exact_year_period(self, start: dt.date, end: dt.date) -> bool:
        """Return whether `start` to `end` spans exactly 12 calendar months inclusive."""
        expected_end = (
            pd.Timestamp(start) + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        ).date()
        return end == expected_end

    def _is_calendar_year_period(self, start: dt.date, end: dt.date) -> bool:
        """Return whether `start` to `end` matches one full calendar year."""
        return start == dt.date(start.year, 1, 1) and end == dt.date(start.year, 12, 31)

    def _resolve_baseline_period(
        self,
        input_data: PVLongTermAnalysisInput,
        frame: pd.DataFrame,
    ) -> tuple[dt.date, dt.date]:
        """Resolve the inclusive baseline period from input data."""
        first_day = frame.index[0].date()

        if input_data.baseline_start is not None and input_data.baseline_end is not None:
            return input_data.baseline_start, input_data.baseline_end

        return self._default_baseline_period(first_day)

    def _choose_bucket_count(self, span_days: int) -> int:
        """Choose how many month-like buckets to create for a baseline span.

        The regression is still intentionally a "monthly point" regression, but the
        baseline period is no longer required to line up with true calendar months.
        A site can start mid-month, and a user can also provide an explicit baseline
        start/end range. To keep the model month-like without creating awkward edge
        buckets, we:

        1. First remove baseline days that are unusable for regression because
           an irradiance predictor or `electricity_produced` is missing.
        2. Partition the remaining valid baseline days, in chronological order,
           into contiguous buckets.
        3. Require those bucket sizes to stay in a month-like range of 25-35 days.
        4. Require the bucket sizes to be as balanced as possible, so adjacent
           buckets differ by at most one day and we never end up with a tiny
           dangling tail bucket.

        This ordering matters. If we bucketed first and only then dropped missing
        baseline days, a long outage could leave one regression point representing
        just a handful of valid days. By choosing the bucket count from the
        remaining valid baseline days instead, each regression point still
        represents a month-like amount of usable data.

        When multiple bucket counts satisfy the size constraints, we choose the one
        whose average bucket length is closest to :attr:`target_bucket_days`
        (roughly one twelfth of a year). If there is still a tie, we prefer the
        larger bucket count so that longer baselines can naturally produce more
        month-like regression points, such as ~24 buckets for a two-year baseline.
        """
        feasible_counts: list[int] = []

        for bucket_count in range(self.minimum_usable_buckets, span_days + 1):
            minimum_size = span_days // bucket_count
            maximum_size = minimum_size + (1 if span_days % bucket_count else 0)
            # The balanced split produced by `_bucket_sizes` can only yield these
            # two sizes, so checking their bounds is enough to validate the whole
            # partition for this candidate bucket count.
            if (
                minimum_size >= self.minimum_bucket_days
                and maximum_size <= self.maximum_bucket_days
            ):
                feasible_counts.append(bucket_count)

        if not feasible_counts:
            raise ValueError(
                "Baseline period must contain enough valid days to form at least 6 "
                "balanced month-like buckets between 25 and 35 days."
            )

        return min(
            feasible_counts,
            key=lambda bucket_count: (
                # Prefer the candidate whose average size is closest to a
                # real-world month length.
                abs((span_days / bucket_count) - self.target_bucket_days),
                # If two candidates are equally close, keep more buckets so
                # longer baselines retain more month-like regression points.
                -bucket_count,
            ),
        )

    def _bucket_sizes(self, valid_day_count: int, bucket_count: int) -> list[int]:
        """Split valid baseline days into contiguous balanced bucket sizes."""
        minimum_size, extra_days = divmod(valid_day_count, bucket_count)
        # Spread the remainder over the earliest buckets so all bucket sizes stay
        # within one day of each other.
        return [minimum_size + 1] * extra_days + [minimum_size] * (bucket_count - extra_days)

    def _has_complete_baseline_data(
        self,
        baseline_frame: pd.DataFrame,
        baseline_start: dt.date,
        baseline_end: dt.date,
        predictors: list[str] | None = None,
    ) -> bool:
        """Return whether the baseline slice has every day and no missing regression inputs."""
        columns = [*(predictors or [const.SOLAR_RADIATION]), const.ELECTRICITY_PRODUCED]
        expected_days = (baseline_end - baseline_start).days + 1
        if len(baseline_frame) != expected_days:
            return False

        if baseline_frame[columns].isna().any().any():
            return False

        baseline_index = baseline_frame.index
        return (
            baseline_index[0].date() == baseline_start and baseline_index[-1].date() == baseline_end
        )

    def _reference_dataset(
        self,
        frame: pd.DataFrame,
        baseline_start: dt.date,
        baseline_end: dt.date,
        predictors: list[str] | None = None,
    ) -> pd.DataFrame:
        """Aggregate usable baseline days into month-like regression buckets."""
        predictors = predictors or [const.SOLAR_RADIATION]
        columns = [*predictors, const.ELECTRICITY_PRODUCED]
        frame_tz = frame.index.tz  # type: ignore
        assert frame_tz is not None

        start_ts = pd.Timestamp(baseline_start, tz=frame_tz)
        end_ts = pd.Timestamp(baseline_end, tz=frame_tz)
        baseline_frame = frame.loc[
            start_ts:end_ts,
            columns,
        ]

        # The common happy path is a full 12-month baseline with complete data.
        # In that case we can keep true calendar-month regression points, which
        # makes the reference easier to interpret and avoids unnecessary custom
        # bucketing. As soon as the baseline is irregular or any baseline day is
        # unusable, we fall back to the balanced month-like bucketing logic.
        if self._is_exact_year_period(
            baseline_start, baseline_end
        ) and self._has_complete_baseline_data(
            baseline_frame,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            predictors=predictors,
        ):
            return baseline_frame.resample("MS").sum()

        baseline_frame = baseline_frame.dropna(subset=columns)
        baseline_index = baseline_frame.index
        bucket_count = self._choose_bucket_count(len(baseline_index))
        bucket_sizes = self._bucket_sizes(len(baseline_index), bucket_count)

        bucket_labels: list[int] = []
        bucket_starts: list[pd.Timestamp] = []
        cursor = 0

        for bucket_id, bucket_size in enumerate(bucket_sizes):
            bucket_labels.extend([bucket_id] * bucket_size)
            bucket_starts.append(baseline_index[cursor])
            cursor += bucket_size
        bucket_start_map = pd.Series(bucket_starts, index=range(bucket_count))

        reference = (
            baseline_frame
            # Buckets are assigned after dropping invalid baseline days, so each
            # bucket represents a balanced amount of usable data rather than a
            # calendar slice that may be mostly outage.
            .assign(bucket=bucket_labels)
            .groupby("bucket", sort=True)[columns]
            .sum()
        )

        reference.index = pd.DatetimeIndex(bucket_start_map.loc[reference.index].to_list())
        return reference

    def _is_complete_calendar_year(self, yearly_frame: pd.DataFrame) -> bool:
        """Return whether a grouped yearly frame covers a full calendar year."""
        yearly_index = yearly_frame.index
        year = int(yearly_index[0].year)
        first_day = yearly_index[0].date()
        last_day = yearly_index[-1].date()
        expected_last_day = dt.date(year, 12, 31)

        if first_day != dt.date(year, 1, 1) or last_day != expected_last_day:
            return False

        expected_days = 366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365
        return len(yearly_frame) == expected_days

    def analyze(self, input_data: PVLongTermAnalysisInput) -> PVLongTermAnalysisOutput:
        """Run the analysis and return typed yearly metrics and diagnostics."""
        frame = self._prepare_daily_frame(input_data)
        predictors = self._predictor_columns(frame)
        # A day without irradiance cannot contribute to the regression predictors,
        # so we exclude the whole day from analysis even if production happened to
        # be present. With two predictors, a day needs both.
        frame = frame.dropna(subset=predictors).copy()
        if frame.empty:
            raise ValueError("Input must contain at least one day with solar_radiation data.")
        baseline_start, baseline_end = self._resolve_baseline_period(input_data, frame)

        reference = self._reference_dataset(
            frame,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            predictors=predictors,
        )
        # No intercept and non-negative weights: production is a non-negative mix
        # of the irradiance series. The predictors are strongly correlated, so an
        # unconstrained fit could return large weights of opposite sign that cancel
        # on the reference period and extrapolate badly to other years.
        regression = fit_linear_regression(
            reference,
            x_name=predictors,
            y_name=const.ELECTRICITY_PRODUCED,
            positive=True,
            fit_intercept=False,
        )

        _, intercept, r_squared = linear_regression_diagnostics(
            regression,
            reference,
            x_name=predictors,
            y_name=const.ELECTRICITY_PRODUCED,
        )
        coefficients = linear_regression_coefficients(regression, predictors)

        yearly_results: list[PVYearResult] = []
        year_index = frame.index.year  # type: ignore
        first_year = int(year_index[0])
        reference_year = (
            baseline_start.year
            if self._is_calendar_year_period(baseline_start, baseline_end)
            else None
        )

        for year, yearly_frame in frame.groupby(year_index):
            complete_year = self._is_complete_calendar_year(yearly_frame)
            if year == first_year and not complete_year:
                continue
            if not yearly_results and reference_year == int(year):
                continue

            actual_monthly = (
                yearly_frame[const.ELECTRICITY_PRODUCED].fillna(0.0).resample("MS").sum()
            )
            irradiance_monthly = (
                yearly_frame[predictors]
                .resample("MS")
                .sum()
                .reindex(actual_monthly.index, fill_value=0.0)
            )

            prediction_series = predict_linear_regression(
                regression,
                irradiance_monthly,
                x_name=predictors,
            )

            actual = float(actual_monthly.sum())
            prediction = float(prediction_series.sum())
            error = actual - prediction
            # Relative to the prediction, so a year with an outage bottoms out at -100%
            # instead of dividing by a near-zero actual. A non-positive prediction
            # (possible with a negative intercept) has no meaningful ratio.
            if prediction > 0:
                relative_error = error / prediction
            else:
                relative_error = 0.0 if actual == 0 else None

            yearly_results.append(
                PVYearResult(
                    year=int(year),
                    actual_production=actual,
                    predicted_production=prediction,
                    error=error,
                    relative_error=relative_error,
                    complete_year=complete_year,
                )
            )

        return PVLongTermAnalysisOutput(
            reference=input_data.reference,
            baseline_period=PVBaselinePeriod(start=baseline_start, end=baseline_end),
            yearly_results=yearly_results,
            regression_diagnostics=PVRegressionDiagnostics(
                coefficient=coefficients[const.SOLAR_RADIATION],
                coefficient_horizontal=coefficients.get(const.SOLAR_RADIATION_HORIZONTAL),
                intercept=intercept,
                r_squared=r_squared,
            ),
        )
