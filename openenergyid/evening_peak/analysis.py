"""Evening peak avoidance analysis.

Computes, per connection and per day, the two quantities the "Avondpiek mijden" analysis
is built on:

- the **evening peak**: the highest quarter-hour power inside a fixed evening window
  (16:00-21:00 by default), in kW;
- the **peak share**: the part of that day's net offtake which falls inside the same
  window, in percent.

Injection is clipped to zero before summation, so the share stays between 0 and 100% and
remains comparable between households with and without solar panels.

The two quantities answer different questions. The peak says how hard the connection
loads the grid at its worst moment; the share says how much of the day sits in the
evening window, which is what a participant can actually shift.
"""

import datetime as dt
from dataclasses import dataclass

import polars as pl

from .schemas import DailyEveningPeakSchema, WeekMedianSchema

#: Quarter-hourly energy in kWh multiplied by this gives average power in kW.
QUARTERS_PER_HOUR = 4

#: Length of one measurement interval, in minutes.
QUARTER_HOUR_MINUTES = 15

TIMESTAMP = "timestamp"
NET_OFFTAKE = "net_offtake_in_kilowatthour"
POWER = "power_in_kilowatt"
DAY = "day"
WEEK = "week"


@dataclass
class EveningPeakAnalysisResult:
    """Result of an evening peak analysis.

    Attributes
    ----------
    daily : pl.LazyFrame
        One row per local day, over a gapless run of days covering the measured range.
        A day with no measurements at all is present with null metrics and
        ``observed_quarters == 0``, so a break in the feed shows up in the series
        instead of having to be inferred from a jump in the index. Columns:

        - ``day``: start of the local day
        - ``evening_peak_in_kilowatt``: highest quarter-hour power inside the window;
          null when the window is not fully covered by measurements
        - ``evening_peak_share_in_percent``: share of net daily offtake inside the
          window; null when the day is incomplete or has no offtake at all
        - ``daily_offtake_in_kilowatthour`` / ``evening_offtake_in_kilowatthour``
        - ``observed_quarters`` / ``expected_quarters`` / ``observed_window_quarters``
        - ``has_full_window``: the evening window is fully measured
        - ``is_complete``: the window is full *and* the day meets the coverage minimum
        - ``is_below_threshold``: the share is strictly below the threshold; null when
          the share is null

    week_medians : pl.LazyFrame
        One row per ISO week (Monday-aligned), over complete days only, with columns
        ``week``, ``median_evening_peak_in_kilowatt`` and
        ``median_evening_peak_share_in_percent``. These are the calmer reference lines
        drawn over the daily series.
    """

    daily: pl.LazyFrame
    week_medians: pl.LazyFrame


@dataclass
class PeakMoment:
    """A single evening peak, with the day it happened on.

    Attributes
    ----------
    peak_time : dt.datetime
        Exact local timestamp of the highest quarter-hour inside the window.
    peak_value_in_kilowatt : float
        Power at that moment, in kW.
    day_curve : pl.DataFrame
        The whole local day at quarter-hour resolution, with columns ``timestamp`` and
        ``power_in_kilowatt``. The full day is returned on purpose: the sparkline shows
        where the peak sits relative to the rest of the day.
    """

    peak_time: dt.datetime
    peak_value_in_kilowatt: float
    day_curve: pl.DataFrame


class EveningPeakAnalyzer:
    """Analyses how much of a connection's consumption falls in the evening peak window.

    The analyzer works in two steps. ``prepare_net_offtake`` turns the two gross meter
    series into one non-negative net offtake series in the analysis timezone;
    ``analyze`` reduces that series to one row per day plus weekly medians.

    Parameters
    ----------
    timezone : str
        IANA timezone for the analysis, e.g. ``"Europe/Amsterdam"``. Day boundaries and
        the evening window are evaluated against the local wall clock, so this must be
        the connection's own timezone or the window lands on the wrong hours.

    window_start, window_end : dt.time
        The evening window, half-open as ``[window_start, window_end)``. The default
        16:00-21:00 covers the twenty quarter-hours 16:00 through 20:45.

    peak_share_threshold : float, default=0.37
        Share below which a day counts as a good day, as a fraction. The default 0.37
        is the peak share of an average Dutch household. One fixed value for all
        participants, configured centrally rather than per workspace.

    min_day_coverage : float, default=0.9
        Minimum fraction of a day's quarter-hours that must be present before its share
        is reported. Without this, the partial first and last day of any export produce
        a share computed against an incomplete denominator, which reads as a very good
        or very bad day when it is neither.

    Example
    -------
    >>> analyzer = EveningPeakAnalyzer(timezone="Europe/Amsterdam")
    >>> net = analyzer.prepare_net_offtake(offtake_lf, injection_lf)
    >>> result = analyzer.analyze(net)
    >>> moments = analyzer.peak_moments(net, num_peaks=10)
    """

    DEFAULT_WINDOW_START = dt.time(16, 0)
    DEFAULT_WINDOW_END = dt.time(21, 0)
    DEFAULT_PEAK_SHARE_THRESHOLD = 0.37
    DEFAULT_MIN_DAY_COVERAGE = 0.9

    def __init__(
        self,
        timezone: str,
        window_start: dt.time = DEFAULT_WINDOW_START,
        window_end: dt.time = DEFAULT_WINDOW_END,
        peak_share_threshold: float = DEFAULT_PEAK_SHARE_THRESHOLD,
        min_day_coverage: float = DEFAULT_MIN_DAY_COVERAGE,
    ):
        if window_start >= window_end:
            raise ValueError(
                f"window_start ({window_start}) must be strictly before window_end ({window_end})."
            )
        if not 0 <= peak_share_threshold <= 1:
            raise ValueError(
                f"peak_share_threshold must be a fraction between 0 and 1, "
                f"got {peak_share_threshold}."
            )
        if not 0 <= min_day_coverage <= 1:
            raise ValueError(
                f"min_day_coverage must be a fraction between 0 and 1, got {min_day_coverage}."
            )

        self.timezone = timezone
        self.window_start = window_start
        self.window_end = window_end
        self.peak_share_threshold = peak_share_threshold
        self.min_day_coverage = min_day_coverage

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _minute_of_day(time: dt.time) -> int:
        """Minutes since local midnight."""
        return time.hour * 60 + time.minute

    @property
    def expected_window_quarters(self) -> int:
        """Quarter-hours the evening window is expected to contain.

        Derived from the wall-clock window length. European DST transitions happen at
        02:00/03:00 local time and so never fall inside an evening window; a window that
        did straddle a transition would need this to be computed per day.
        """
        span = self._minute_of_day(self.window_end) - self._minute_of_day(self.window_start)
        return span // QUARTER_HOUR_MINUTES

    def _day_spine(self, observed: pl.LazyFrame) -> pl.LazyFrame:
        """A gapless run of local days covering the observed range.

        Days with no measurements at all are kept as rows with null metrics rather than
        dropped, so a gap in the feed is visible in the result instead of having to be
        inferred from a jump in the index. A chart drawn straight from the series then
        breaks where the data breaks.
        """
        bounds = observed.select(
            pl.col(DAY).min().alias("first"), pl.col(DAY).max().alias("last")
        ).collect()
        first_day, last_day = bounds.row(0)
        return pl.LazyFrame(
            {
                DAY: pl.datetime_range(
                    first_day,
                    last_day,
                    interval="1d",
                    time_zone=self.timezone,
                    eager=True,
                )
            }
        )

    def _in_window(self) -> pl.Expr:
        """Predicate selecting quarter-hours inside the evening window.

        Compares minute-of-day against the local wall clock, so it is unaffected by DST.
        ``dt.hour()`` and ``dt.minute()`` return Int8, which silently overflows when
        multiplied by 60 (16:00 would become -34), hence the cast.
        """
        minute_of_day = pl.col(TIMESTAMP).dt.hour().cast(pl.Int32) * 60 + pl.col(
            TIMESTAMP
        ).dt.minute().cast(pl.Int32)
        return minute_of_day.is_between(
            self._minute_of_day(self.window_start),
            self._minute_of_day(self.window_end),
            closed="left",
        )

    def _localize(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Put ``timestamp`` in the analysis timezone, whatever it arrives as."""
        schema = frame.collect_schema()
        dtype = schema[TIMESTAMP]
        naive = getattr(dtype, "time_zone", None) is None
        expression = pl.col(TIMESTAMP)
        if naive:
            expression = expression.dt.replace_time_zone("UTC")
        return frame.with_columns(expression.dt.convert_time_zone(self.timezone).alias(TIMESTAMP))

    @staticmethod
    def _single_value_column(frame: pl.LazyFrame, name: str) -> pl.LazyFrame:
        """Rename the one non-timestamp column to ``name``.

        Gross series arrive from ``TimeSeries.to_polars``, which names the value column
        after the series, so the column is identified by position rather than by name.
        """
        columns = frame.collect_schema().names()
        if TIMESTAMP not in columns:
            raise ValueError(f"Input frame must have a '{TIMESTAMP}' column, got {columns}.")
        value_columns = [column for column in columns if column != TIMESTAMP]
        if len(value_columns) != 1:
            raise ValueError(
                f"Input frame must have exactly one value column besides '{TIMESTAMP}', "
                f"got {value_columns}."
            )
        return frame.select(
            pl.col(TIMESTAMP), pl.col(value_columns[0]).cast(pl.Float64).alias(name)
        )

    # ------------------------------------------------------------------ step 1

    def prepare_net_offtake(
        self,
        gross_offtake_lf: pl.LazyFrame,
        gross_injection_lf: pl.LazyFrame | None = None,
    ) -> pl.LazyFrame:
        """Combine the two gross meter series into one net offtake series.

        Parameters
        ----------
        gross_offtake_lf : pl.LazyFrame
            ``timestamp`` plus one column of gross offtake in kWh per quarter-hour.
        gross_injection_lf : pl.LazyFrame | None
            The same for gross injection. Omit it for a connection without production;
            net offtake then equals gross offtake.

        Returns
        -------
        pl.LazyFrame
            ``timestamp`` in the analysis timezone and ``net_offtake_in_kilowatthour``,
            sorted, with nulls dropped and negatives clipped to zero.

        Notes
        -----
        Clipping happens per quarter-hour, before any summation. A quarter-hour in which
        the connection injects more than it takes counts as zero offtake rather than as
        negative offtake, which is what keeps the daily share inside 0-100%.
        """
        offtake = self._localize(self._single_value_column(gross_offtake_lf, "gross_offtake"))

        if gross_injection_lf is None:
            net = offtake.with_columns(pl.col("gross_offtake").alias(NET_OFFTAKE))
        else:
            injection = self._localize(
                self._single_value_column(gross_injection_lf, "gross_injection")
            )
            net = offtake.join(injection, on=TIMESTAMP, how="full", coalesce=True).with_columns(
                (
                    pl.col("gross_offtake").fill_null(0.0)
                    - pl.col("gross_injection").fill_null(0.0)
                ).alias(NET_OFFTAKE)
            )

        return (
            net.select(TIMESTAMP, NET_OFFTAKE)
            .filter(pl.col(TIMESTAMP).is_not_null() & pl.col(NET_OFFTAKE).is_not_null())
            .with_columns(pl.col(NET_OFFTAKE).clip(lower_bound=0.0))
            .sort(TIMESTAMP)
        )

    # ------------------------------------------------------------------ step 2

    def analyze(self, net_lf: pl.LazyFrame) -> EveningPeakAnalysisResult:
        """Reduce a net offtake series to daily metrics and weekly medians.

        Parameters
        ----------
        net_lf : pl.LazyFrame
            Output of :meth:`prepare_net_offtake`: ``timestamp`` and
            ``net_offtake_in_kilowatthour``.

        Returns
        -------
        EveningPeakAnalysisResult
        """
        net_lf = self._localize(net_lf.select(TIMESTAMP, NET_OFFTAKE)).sort(TIMESTAMP)

        if net_lf.select(pl.len()).collect().item() == 0:
            return self._empty_result()

        tagged = net_lf.with_columns(
            pl.col(TIMESTAMP).dt.truncate("1d").alias(DAY),
            (pl.col(NET_OFFTAKE) * QUARTERS_PER_HOUR).alias(POWER),
            self._in_window().alias("in_window"),
        )

        observed = (
            tagged.group_by(DAY)
            .agg(
                pl.col(NET_OFFTAKE).sum().alias("daily_offtake_in_kilowatthour"),
                pl.col(NET_OFFTAKE)
                .filter(pl.col("in_window"))
                .sum()
                .alias("evening_offtake_in_kilowatthour"),
                pl.col(POWER).filter(pl.col("in_window")).max().alias("raw_evening_peak"),
                pl.len().cast(pl.Int64).alias("observed_quarters"),
                pl.col("in_window").sum().cast(pl.Int64).alias("observed_window_quarters"),
            )
            .sort(DAY)
        )

        daily = (
            self._day_spine(observed)
            .join(observed, on=DAY, how="left")
            .with_columns(
                pl.col("daily_offtake_in_kilowatthour").fill_null(0.0),
                pl.col("evening_offtake_in_kilowatthour").fill_null(0.0),
                pl.col("observed_quarters").fill_null(0),
                pl.col("observed_window_quarters").fill_null(0),
            )
            # Day length is DST-aware: the October day has 100 quarter-hours and the
            # March day 92, so coverage cannot be measured against a constant 96.
            .with_columns(
                (
                    (pl.col(DAY).dt.offset_by("1d") - pl.col(DAY)).dt.total_minutes()
                    // QUARTER_HOUR_MINUTES
                )
                .cast(pl.Int64)
                .alias("expected_quarters")
            )
            .with_columns(
                (pl.col("observed_window_quarters") >= self.expected_window_quarters).alias(
                    "has_full_window"
                )
            )
            .with_columns(
                (
                    pl.col("has_full_window")
                    & (
                        pl.col("observed_quarters")
                        >= pl.col("expected_quarters") * self.min_day_coverage
                    )
                ).alias("is_complete")
            )
            .with_columns(
                pl.when(pl.col("has_full_window"))
                .then(pl.col("raw_evening_peak"))
                .otherwise(None)
                .alias("evening_peak_in_kilowatt"),
                pl.when(pl.col("is_complete") & (pl.col("daily_offtake_in_kilowatthour") > 0))
                .then(
                    pl.col("evening_offtake_in_kilowatthour")
                    / pl.col("daily_offtake_in_kilowatthour")
                    * 100
                )
                .otherwise(None)
                .alias("evening_peak_share_in_percent"),
            )
            .with_columns(
                pl.when(pl.col("evening_peak_share_in_percent").is_not_null())
                .then(pl.col("evening_peak_share_in_percent") < self.peak_share_threshold * 100)
                .otherwise(None)
                .alias("is_below_threshold")
            )
            .drop("raw_evening_peak")
            .select(
                DAY,
                "evening_peak_in_kilowatt",
                "evening_peak_share_in_percent",
                "daily_offtake_in_kilowatthour",
                "evening_offtake_in_kilowatthour",
                "observed_quarters",
                "expected_quarters",
                "observed_window_quarters",
                "has_full_window",
                "is_complete",
                "is_below_threshold",
            )
        )

        week_medians = (
            daily.filter(pl.col("is_complete"))
            .sort(DAY)
            .group_by_dynamic(DAY, every="1w", start_by="monday")
            .agg(
                pl.col("evening_peak_in_kilowatt")
                .median()
                .alias("median_evening_peak_in_kilowatt"),
                pl.col("evening_peak_share_in_percent")
                .median()
                .alias("median_evening_peak_share_in_percent"),
            )
            .rename({DAY: WEEK})
            .sort(WEEK)
        )

        # Validate eagerly: pandera only runs value checks on a collected frame, and
        # these are one row per day and per week, so collecting them costs nothing.
        daily = DailyEveningPeakSchema.validate(daily.collect()).lazy()
        week_medians = WeekMedianSchema.validate(week_medians.collect()).lazy()

        return EveningPeakAnalysisResult(daily=daily, week_medians=week_medians)

    # ------------------------------------------------------------------ step 3

    def peak_moments(self, net_lf: pl.LazyFrame, num_peaks: int = 10) -> list[PeakMoment]:
        """Return the highest evening peaks, each with the curve of its own day.

        One peak per day is taken, so — unlike the capacity analysis, which takes the
        largest quarter-hours across the whole series — no de-duplication of neighbouring
        quarter-hours is needed.

        Parameters
        ----------
        net_lf : pl.LazyFrame
            Output of :meth:`prepare_net_offtake`.
        num_peaks : int, default=10
            Maximum number of peaks to return, highest first.

        Returns
        -------
        list[PeakMoment]
            Days whose evening window is not fully measured are skipped.
        """
        if num_peaks <= 0:
            return []

        net_lf = self._localize(net_lf.select(TIMESTAMP, NET_OFFTAKE)).sort(TIMESTAMP)
        tagged = net_lf.with_columns(
            pl.col(TIMESTAMP).dt.truncate("1d").alias(DAY),
            (pl.col(NET_OFFTAKE) * QUARTERS_PER_HOUR).alias(POWER),
            self._in_window().alias("in_window"),
        ).collect()

        if tagged.height == 0:
            return []

        candidates = (
            tagged.filter(pl.col("in_window"))
            .group_by(DAY)
            .agg(
                pl.col(POWER).max().alias("peak_value"),
                pl.col(TIMESTAMP).sort_by(POWER, descending=True).first().alias("peak_time"),
                pl.len().alias("window_quarters"),
            )
            .filter(
                (pl.col("window_quarters") >= self.expected_window_quarters)
                & pl.col("peak_value").is_not_null()
            )
            .sort("peak_value", descending=True)
            .head(num_peaks)
        )

        moments: list[PeakMoment] = []
        for row in candidates.iter_rows(named=True):
            day_curve = tagged.filter(pl.col(DAY) == row[DAY]).select(TIMESTAMP, POWER)
            moments.append(
                PeakMoment(
                    peak_time=row["peak_time"],
                    peak_value_in_kilowatt=row["peak_value"],
                    day_curve=day_curve,
                )
            )
        return moments

    # ------------------------------------------------------------------ empty

    def _empty_result(self) -> EveningPeakAnalysisResult:
        """An explicitly typed empty result, so callers need no special casing."""
        day_dtype = pl.Datetime(time_zone=self.timezone)
        daily = pl.LazyFrame(
            schema={
                DAY: day_dtype,
                "evening_peak_in_kilowatt": pl.Float64,
                "evening_peak_share_in_percent": pl.Float64,
                "daily_offtake_in_kilowatthour": pl.Float64,
                "evening_offtake_in_kilowatthour": pl.Float64,
                "observed_quarters": pl.Int64,
                "expected_quarters": pl.Int64,
                "observed_window_quarters": pl.Int64,
                "has_full_window": pl.Boolean,
                "is_complete": pl.Boolean,
                "is_below_threshold": pl.Boolean,
            }
        )
        week_medians = pl.LazyFrame(
            schema={
                WEEK: day_dtype,
                "median_evening_peak_in_kilowatt": pl.Float64,
                "median_evening_peak_share_in_percent": pl.Float64,
            }
        )
        return EveningPeakAnalysisResult(daily=daily, week_medians=week_medians)


def summarize(
    result: EveningPeakAnalysisResult, peak_share_threshold: float
) -> dict[str, float | int | dt.date | None]:
    """Reduce a result to the key figures shown under the two cards.

    Parameters
    ----------
    result : EveningPeakAnalysisResult
    peak_share_threshold : float
        The threshold the day count was made against, as a fraction. Echoed back in the
        summary so a reader knows what "below threshold" meant.

    Returns
    -------
    dict
        Averages, extremes, the number of days below the threshold and the number of
        measured days. ``days_below_threshold`` and ``measured_days`` are the numerator
        and denominator of the "49 of 120 days" figure; both count complete days only.
    """
    daily = result.daily.collect()
    complete = daily.filter(pl.col("evening_peak_share_in_percent").is_not_null())
    peaks = daily.filter(pl.col("evening_peak_in_kilowatt").is_not_null())

    def _value(frame: pl.DataFrame, column: str, how: str) -> float | None:
        if frame.height == 0:
            return None
        return getattr(frame[column], how)()

    return {
        "average_peak_in_kilowatt": _value(peaks, "evening_peak_in_kilowatt", "mean"),
        "lowest_peak_in_kilowatt": _value(peaks, "evening_peak_in_kilowatt", "min"),
        "highest_peak_in_kilowatt": _value(peaks, "evening_peak_in_kilowatt", "max"),
        "average_share_in_percent": _value(complete, "evening_peak_share_in_percent", "mean"),
        "lowest_share_in_percent": _value(complete, "evening_peak_share_in_percent", "min"),
        "highest_share_in_percent": _value(complete, "evening_peak_share_in_percent", "max"),
        "days_below_threshold": int(complete["is_below_threshold"].sum()) if complete.height else 0,
        "measured_days": complete.height,
        "threshold_in_percent": peak_share_threshold * 100,
        "first_day": daily[DAY].min().date() if daily.height else None,
        "last_day": daily[DAY].max().date() if daily.height else None,
    }
