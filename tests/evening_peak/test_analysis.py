"""Tests for the evening peak avoidance analysis."""

import datetime as dt
import itertools

import polars as pl
import pytest

from openenergyid.evening_peak import (
    DailyEveningPeakSchema,
    EveningPeakAnalyzer,
    NetOfftakeSchema,
    summarize,
)

from .conftest import (
    TIMEZONE,
    day,
    flat_evening_profile,
    frame,
    in_evening,
    local_quarters,
)

# Baseload 0.05 kWh/quarter for 76 quarters, 0.5 kWh/quarter for the 20 evening quarters.
FLAT_DAILY_OFFTAKE = 76 * 0.05 + 20 * 0.5
FLAT_EVENING_OFFTAKE = 20 * 0.5
FLAT_SHARE = FLAT_EVENING_OFFTAKE / FLAT_DAILY_OFFTAKE * 100
FLAT_PEAK = 0.5 * 4


def analyze(offtake, injection=None, **kwargs):
    """Run a full analysis and return (analyzer, daily DataFrame, result)."""
    analyzer = EveningPeakAnalyzer(timezone=TIMEZONE, **kwargs)
    net = analyzer.prepare_net_offtake(offtake, injection)
    result = analyzer.analyze(net)
    return analyzer, result.daily.collect(), result


class TestDailyMetrics:
    """The two headline quantities."""

    @pytest.fixture
    def one_flat_day(self) -> pl.LazyFrame:
        index = local_quarters(day(2026, 11, 2), 96)
        return frame(index, flat_evening_profile)

    def test_peak_and_share_match_hand_computation(self, one_flat_day):
        _, daily, _ = analyze(one_flat_day)

        assert daily.height == 1
        row = daily.row(0, named=True)
        assert row["evening_peak_in_kilowatt"] == pytest.approx(FLAT_PEAK)
        assert row["evening_peak_share_in_percent"] == pytest.approx(FLAT_SHARE)
        assert row["daily_offtake_in_kilowatthour"] == pytest.approx(FLAT_DAILY_OFFTAKE)
        assert row["evening_offtake_in_kilowatthour"] == pytest.approx(FLAT_EVENING_OFFTAKE)
        assert row["observed_window_quarters"] == 20
        assert row["is_complete"] is True

    def test_peak_ignores_a_larger_spike_outside_the_window(self):
        """A midday spike must not become the evening peak, however large."""
        index = local_quarters(day(2026, 11, 2), 96)

        def profile(timestamp: dt.datetime) -> float:
            if timestamp.hour == 12 and timestamp.minute == 0:
                return 3.0  # 12 kW, far above the evening block
            return flat_evening_profile(timestamp)

        _, daily, _ = analyze(frame(index, profile))
        row = daily.row(0, named=True)

        assert row["evening_peak_in_kilowatt"] == pytest.approx(FLAT_PEAK)
        # The spike still counts towards the day's total, so it lowers the share.
        assert row["evening_peak_share_in_percent"] < FLAT_SHARE

    def test_share_is_null_on_a_day_without_offtake(self):
        """A day of zeroes must not raise, and has no meaningful share."""
        index = local_quarters(day(2026, 11, 2), 96)
        _, daily, _ = analyze(frame(index, lambda _: 0.0))
        row = daily.row(0, named=True)

        assert row["daily_offtake_in_kilowatthour"] == 0.0
        assert row["evening_peak_share_in_percent"] is None
        assert row["is_below_threshold"] is None
        # The peak is still defined: the window was measured, it was simply zero.
        assert row["evening_peak_in_kilowatt"] == 0.0

    def test_window_bounds_are_half_open(self):
        """21:00 belongs to the next window, 16:00 to this one."""
        index = local_quarters(day(2026, 11, 2), 96)

        def only_edges(timestamp: dt.datetime) -> float:
            if (timestamp.hour, timestamp.minute) == (16, 0):
                return 1.0
            if (timestamp.hour, timestamp.minute) == (21, 0):
                return 2.0
            return 0.0

        _, daily, _ = analyze(frame(index, only_edges))
        row = daily.row(0, named=True)

        assert row["evening_offtake_in_kilowatthour"] == pytest.approx(1.0)
        assert row["evening_peak_in_kilowatt"] == pytest.approx(4.0)


class TestInjection:
    """Injection is clipped to zero per quarter-hour, before any summation."""

    def test_share_stays_within_bounds_when_injection_dominates(self):
        """Midday injection above offtake must not push the share over 100%."""
        index = local_quarters(day(2026, 6, 15), 96)

        def injection(timestamp: dt.datetime) -> float:
            # A big solar midday, larger than the concurrent offtake.
            return 2.0 if 10 <= timestamp.hour < 16 else 0.0

        _, daily, _ = analyze(
            frame(index, flat_evening_profile),
            frame(index, injection, name="gross_injection"),
        )
        row = daily.row(0, named=True)

        assert 0 <= row["evening_peak_share_in_percent"] <= 100
        assert row["daily_offtake_in_kilowatthour"] >= 0

    def test_clipping_is_per_quarter_not_on_the_daily_total(self):
        """Surplus injection in one quarter must not offset offtake in another."""
        index = local_quarters(day(2026, 6, 15), 96)
        # 1 kWh of offtake at 18:00 only; 5 kWh of injection at 12:00 only.
        offtake = frame(
            index,
            lambda t: 1.0 if (t.hour, t.minute) == (18, 0) else 0.0,
        )
        injection = frame(
            index,
            lambda t: 5.0 if (t.hour, t.minute) == (12, 0) else 0.0,
            name="gross_injection",
        )
        _, daily, _ = analyze(offtake, injection)
        row = daily.row(0, named=True)

        # Had clipping happened after summation, the day would net to -4 kWh.
        assert row["daily_offtake_in_kilowatthour"] == pytest.approx(1.0)
        assert row["evening_peak_share_in_percent"] == pytest.approx(100.0)

    def test_injection_may_be_omitted(self):
        """A connection without production sends no injection series."""
        index = local_quarters(day(2026, 11, 2), 96)
        _, with_none, _ = analyze(frame(index, flat_evening_profile), None)
        _, with_zeros, _ = analyze(
            frame(index, flat_evening_profile),
            frame(index, lambda _: 0.0, name="gross_injection"),
        )

        assert with_none.row(0, named=True)["evening_peak_share_in_percent"] == pytest.approx(
            with_zeros.row(0, named=True)["evening_peak_share_in_percent"]
        )

    def test_injection_covering_a_shorter_range_is_treated_as_zero(self):
        """Missing injection quarter-hours must not drop offtake rows.

        This is the *intended* direction of the documented tolerance — injection may
        report fewer quarter-hours than offtake — and must keep passing unchanged as the
        regression anchor for it, distinct from the (disallowed) opposite direction below.
        """
        index = local_quarters(day(2026, 11, 2), 96)
        partial = index[:10]
        _, daily, _ = analyze(
            frame(index, flat_evening_profile),
            frame(partial, lambda _: 0.0, name="gross_injection"),
        )
        row = daily.row(0, named=True)

        assert row["observed_quarters"] == 96
        assert row["evening_peak_share_in_percent"] == pytest.approx(FLAT_SHARE)

    def test_injection_covering_a_wider_range_does_not_fabricate_offtake(self):
        """A quarter-hour present only in injection must not become a net-offtake row.

        Offtake is authoritative: the result contains exactly offtake's own timestamps,
        regardless of what injection additionally reports.
        """
        index = local_quarters(day(2026, 11, 2), 96)
        wider = local_quarters(day(2026, 11, 1), 4 * 96)  # four days, offtake is one
        _, daily, _ = analyze(
            frame(index, flat_evening_profile),
            frame(wider, lambda _: 0.0, name="gross_injection"),
        )

        assert daily.height == 1
        assert daily.row(0, named=True)["observed_quarters"] == 96

    def test_offtake_outage_stays_a_gap_even_with_zero_filled_injection(self):
        """A zero-filled injection register must not paper over a real offtake outage.

        A zero-filled injection register is routine for a connection without PV: it still
        reports zero every quarter-hour. Before the join-direction fix, that was enough to
        turn a genuine offtake outage into a suspiciously perfect 0 kW / null-share day
        rather than the incomplete day it actually is.
        """
        measured = [
            t for t in local_quarters(day(2026, 11, 1), 3 * 96) if t.date() != dt.date(2026, 11, 2)
        ]
        full_range = local_quarters(day(2026, 11, 1), 3 * 96)
        _, daily, _ = analyze(
            frame(measured, flat_evening_profile),
            frame(full_range, lambda _: 0.0, name="gross_injection"),
        )
        outage = daily.filter(pl.col("day").dt.date() == dt.date(2026, 11, 2)).row(0, named=True)

        assert outage["observed_quarters"] == 0
        assert outage["is_complete"] is False
        assert outage["evening_peak_in_kilowatt"] is None
        assert outage["evening_peak_share_in_percent"] is None


class TestCoverage:
    """Incompletely measured days must not produce confident numbers."""

    def test_evening_only_day_reports_a_peak_but_no_share(self):
        """The classic partial export: the window is there, the denominator is not."""
        index = [t for t in local_quarters(day(2026, 11, 2), 96) if in_evening(t)]
        _, daily, _ = analyze(frame(index, flat_evening_profile))
        row = daily.row(0, named=True)

        assert row["observed_quarters"] == 20
        assert row["has_full_window"] is True
        assert row["is_complete"] is False
        assert row["evening_peak_in_kilowatt"] == pytest.approx(FLAT_PEAK)
        assert row["evening_peak_share_in_percent"] is None

    def test_day_missing_part_of_the_window_reports_no_peak(self):
        index = [
            t for t in local_quarters(day(2026, 11, 2), 96) if not (t.hour == 18 and t.minute == 0)
        ]
        _, daily, _ = analyze(frame(index, flat_evening_profile))
        row = daily.row(0, named=True)

        assert row["observed_window_quarters"] == 19
        assert row["has_full_window"] is False
        assert row["evening_peak_in_kilowatt"] is None
        assert row["evening_peak_share_in_percent"] is None

    def test_incomplete_days_are_excluded_from_the_summary_counts(self):
        """measuredDays is the denominator of the "x of y days" figure."""
        complete = local_quarters(day(2026, 11, 2), 96)
        partial = [t for t in local_quarters(day(2026, 11, 3), 96) if in_evening(t)]
        _, _, result = analyze(frame(complete + partial, flat_evening_profile))
        stats = summarize(result, peak_share_threshold=0.37)

        assert stats["measured_days"] == 1


class TestThreshold:
    """The days-below-threshold count."""

    def test_count_is_strictly_below_the_threshold(self):
        """A day sitting exactly on the threshold does not count as below it."""
        index = local_quarters(day(2026, 11, 2), 96)
        # Constant offtake all day: the share equals the window's share of the day.
        _, daily, result = analyze(frame(index, lambda _: 0.25))
        share = daily.row(0, named=True)["evening_peak_share_in_percent"]
        assert share == pytest.approx(20 / 96 * 100)

        exact = EveningPeakAnalyzer(timezone=TIMEZONE, peak_share_threshold=share / 100)
        on_threshold = exact.analyze(exact.prepare_net_offtake(frame(index, lambda _: 0.25)))
        assert on_threshold.daily.collect().row(0, named=True)["is_below_threshold"] is False

        stats = summarize(result, peak_share_threshold=0.37)
        assert stats["days_below_threshold"] == 1
        assert stats["threshold_in_percent"] == pytest.approx(37.0)

    def test_threshold_is_configurable(self):
        index = local_quarters(day(2026, 11, 2), 96)
        _, daily, _ = analyze(frame(index, flat_evening_profile), peak_share_threshold=0.9)
        assert daily.row(0, named=True)["is_below_threshold"] is True


class TestDaylightSaving:
    """Local day boundaries must follow the wall clock, not a fixed 24 hours."""

    def test_long_and_short_days_are_each_one_row_with_correct_length(self):
        # 2026-10-25 falls back (25 hours); 2026-03-29 springs forward (23 hours).
        autumn = local_quarters(day(2026, 10, 24), 4 * 96)
        spring = local_quarters(day(2026, 3, 28), 4 * 96)

        for index, transition, quarters in (
            (autumn, dt.date(2026, 10, 25), 100),
            (spring, dt.date(2026, 3, 29), 92),
        ):
            _, daily, _ = analyze(frame(index, flat_evening_profile))
            days = {row["day"].date(): row for row in daily.iter_rows(named=True)}

            assert len(daily.filter(pl.col("day").dt.date() == transition)) == 1
            row = days[transition]
            assert row["observed_quarters"] == quarters
            assert row["expected_quarters"] == quarters
            # The DST transition happens at night, so the evening window is intact
            # and the day is complete despite not having 96 quarter-hours.
            assert row["observed_window_quarters"] == 20
            assert row["is_complete"] is True
            assert row["evening_peak_in_kilowatt"] == pytest.approx(FLAT_PEAK)

    def test_share_denominator_follows_the_real_day_length(self):
        """The 25-hour day has an extra hour of baseload, so a lower share."""
        autumn = local_quarters(day(2026, 10, 24), 4 * 96)
        _, daily, _ = analyze(frame(autumn, flat_evening_profile))
        days = {row["day"].date(): row for row in daily.iter_rows(named=True)}

        long_day = days[dt.date(2026, 10, 25)]["evening_peak_share_in_percent"]
        normal_day = days[dt.date(2026, 10, 26)]["evening_peak_share_in_percent"]
        assert long_day < normal_day
        assert long_day == pytest.approx(FLAT_EVENING_OFFTAKE / (80 * 0.05 + 10.0) * 100)


class TestWeekMedians:
    """The calmer reference line drawn over the daily series."""

    def test_medians_are_monday_aligned_and_one_row_per_week(self):
        # 2026-11-02 is a Monday; three weeks of data.
        index = local_quarters(day(2026, 11, 2), 21 * 96)
        _, _, result = analyze(frame(index, flat_evening_profile))
        weeks = result.week_medians.collect()

        assert weeks.height == 3
        assert [row["week"].date() for row in weeks.iter_rows(named=True)] == [
            dt.date(2026, 11, 2),
            dt.date(2026, 11, 9),
            dt.date(2026, 11, 16),
        ]
        assert all(row["week"].weekday() == 0 for row in weeks.iter_rows(named=True))

    def test_medians_start_on_the_monday_of_a_partial_first_week(self):
        # 2026-11-04 is a Wednesday.
        index = local_quarters(day(2026, 11, 4), 5 * 96)
        _, _, result = analyze(frame(index, flat_evening_profile))
        weeks = result.week_medians.collect()

        assert weeks.row(0, named=True)["week"].date() == dt.date(2026, 11, 2)

    def test_medians_ignore_incomplete_days(self):
        complete = local_quarters(day(2026, 11, 2), 96)
        partial = [t for t in local_quarters(day(2026, 11, 3), 96) if in_evening(t)]

        def profile(timestamp: dt.datetime) -> float:
            # Give the partial day a wildly different level.
            if timestamp.date() == dt.date(2026, 11, 3):
                return 5.0
            return flat_evening_profile(timestamp)

        _, _, result = analyze(frame(complete + partial, profile))
        weeks = result.week_medians.collect()

        assert weeks.height == 1
        assert weeks.row(0, named=True)["median_evening_peak_in_kilowatt"] == pytest.approx(
            FLAT_PEAK
        )

    def test_a_whole_missing_week_reads_as_a_null_row_not_an_absence(self):
        """Mirrors TestGaps' daily behaviour one resolution up: a week with no complete
        day in it must still appear, with null medians, so a step-line chart drawn from
        this series breaks at the outage instead of running flat across it."""
        # Two two-week blocks (days 1-14, 29-42) around a two-week outage (15-28), so the
        # ISO week 2026-11-16 has no data in it at all.
        days_present = list(range(1, 15)) + list(range(29, 43))
        index = [
            timestamp
            for offset in days_present
            for timestamp in local_quarters(day(2026, 11, 1) + dt.timedelta(days=offset - 1), 96)
        ]
        _, _, result = analyze(frame(index, flat_evening_profile))
        weeks = result.week_medians.collect()

        assert weeks.height == 7
        missing_week = weeks.filter(pl.col("week").dt.date() == dt.date(2026, 11, 16))
        assert missing_week.height == 1
        assert missing_week.row(0, named=True)["median_evening_peak_in_kilowatt"] is None
        assert missing_week.row(0, named=True)["median_evening_peak_share_in_percent"] is None

    def test_measured_but_never_complete_data_gives_all_null_weeks_without_crashing(self):
        """Every day only evening-only (has_full_window True, is_complete False): the
        spine must still build from the daily range even though no week ever computes
        a real median."""
        index = [
            timestamp
            for offset in range(10)
            for timestamp in local_quarters(day(2026, 11, 2, 16, 0) + dt.timedelta(days=offset), 20)
        ]
        _, daily, result = analyze(frame(index, flat_evening_profile))
        weeks = result.week_medians.collect()

        assert not daily["is_complete"].any()
        assert weeks.height == 2
        assert weeks["median_evening_peak_in_kilowatt"].null_count() == 2


class TestPeakMoments:
    """The Piekmomenten card."""

    @pytest.fixture
    def rising_peaks(self) -> pl.LazyFrame:
        """Ten days whose evening peak grows by 0.1 kWh/quarter each day."""
        index = local_quarters(day(2026, 11, 2), 10 * 96)
        first_day = index[0].date()

        def profile(timestamp: dt.datetime) -> float:
            if not in_evening(timestamp):
                return 0.05
            offset = (timestamp.date() - first_day).days
            return 0.5 + 0.1 * offset

        return frame(index, profile)

    def test_returns_highest_first_one_per_day_with_a_full_day_curve(self, rising_peaks):
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        net = analyzer.prepare_net_offtake(rising_peaks)
        moments = analyzer.peak_moments(net, num_peaks=4)

        assert len(moments) == 4
        values = [moment.peak_value_in_kilowatt for moment in moments]
        assert values == sorted(values, reverse=True)
        assert values[0] == pytest.approx((0.5 + 0.9) * 4)

        days = [moment.peak_time.date() for moment in moments]
        assert len(set(days)) == 4, "at most one peak per day"
        assert days[0] == dt.date(2026, 11, 11)

        for moment in moments:
            assert moment.day_curve.height == 96
            assert moment.peak_time.date() == moment.day_curve["timestamp"][0].date()
            assert 16 <= moment.peak_time.hour < 21

    def test_num_peaks_of_zero_returns_nothing(self, rising_peaks):
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        net = analyzer.prepare_net_offtake(rising_peaks)
        assert analyzer.peak_moments(net, num_peaks=0) == []

    def test_asking_for_more_peaks_than_there_are_days(self, rising_peaks):
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        net = analyzer.prepare_net_offtake(rising_peaks)
        assert len(analyzer.peak_moments(net, num_peaks=50)) == 10

    def test_days_without_a_full_window_are_skipped(self):
        full = local_quarters(day(2026, 11, 2), 96)
        # A later day with a much bigger evening, but a hole in the window.
        broken = [
            t for t in local_quarters(day(2026, 11, 3), 96) if not (t.hour == 19 and t.minute == 0)
        ]

        def profile(timestamp: dt.datetime) -> float:
            if timestamp.date() == dt.date(2026, 11, 3) and in_evening(timestamp):
                return 3.0
            return flat_evening_profile(timestamp)

        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        net = analyzer.prepare_net_offtake(frame(full + broken, profile))
        moments = analyzer.peak_moments(net, num_peaks=5)

        assert [moment.peak_time.date() for moment in moments] == [dt.date(2026, 11, 2)]


class TestEdgeCases:
    """Degenerate inputs return empty results rather than raising."""

    def test_empty_input(self):
        empty = frame([], flat_evening_profile)
        analyzer, daily, result = analyze(empty)

        assert daily.height == 0
        assert result.week_medians.collect().height == 0
        assert analyzer.peak_moments(analyzer.prepare_net_offtake(empty)) == []

    def test_summary_of_an_empty_result_has_no_invented_numbers(self):
        _, _, result = analyze(frame([], flat_evening_profile))
        stats = summarize(result, peak_share_threshold=0.37)

        assert stats["measured_days"] == 0
        assert stats["days_below_threshold"] == 0
        assert stats["average_peak_in_kilowatt"] is None
        assert stats["first_day"] is None

    def test_all_null_values_are_dropped(self):
        index = local_quarters(day(2026, 11, 2), 96)
        from openenergyid.models import TimeSeries

        nulls = TimeSeries(name="gross_offtake", index=index, data=[None] * 96).to_polars(
            timezone=TIMEZONE
        )
        _, daily, _ = analyze(nulls)
        assert daily.height == 0

    def test_a_single_quarter_hour(self):
        index = local_quarters(day(2026, 11, 2, 18, 0), 1)
        _, daily, _ = analyze(frame(index, flat_evening_profile))
        row = daily.row(0, named=True)

        assert row["observed_quarters"] == 1
        assert row["has_full_window"] is False
        assert row["evening_peak_share_in_percent"] is None


class TestConfigurationValidation:
    """Bad parameters fail loudly at construction."""

    @pytest.mark.parametrize(
        ("start", "end"),
        [(dt.time(21, 0), dt.time(16, 0)), (dt.time(16, 0), dt.time(16, 0))],
    )
    def test_window_must_be_a_forward_interval(self, start, end):
        with pytest.raises(ValueError, match="strictly before"):
            EveningPeakAnalyzer(timezone=TIMEZONE, window_start=start, window_end=end)

    @pytest.mark.parametrize("threshold", [-0.1, 1.5])
    def test_threshold_must_be_a_fraction(self, threshold):
        with pytest.raises(ValueError, match="fraction between 0 and 1"):
            EveningPeakAnalyzer(timezone=TIMEZONE, peak_share_threshold=threshold)

    def test_min_day_coverage_must_be_a_fraction(self):
        with pytest.raises(ValueError, match="fraction between 0 and 1"):
            EveningPeakAnalyzer(timezone=TIMEZONE, min_day_coverage=2.0)

    def test_window_must_span_a_whole_number_of_quarter_hours(self):
        """A 10-minute window would floor expected_window_quarters to 0, making every
        coverage check on it vacuously true."""
        with pytest.raises(ValueError, match="not a whole number"):
            EveningPeakAnalyzer(
                timezone=TIMEZONE, window_start=dt.time(16, 0), window_end=dt.time(16, 10)
            )

    def test_a_custom_window_changes_the_expected_quarter_count(self):
        analyzer = EveningPeakAnalyzer(
            timezone=TIMEZONE, window_start=dt.time(17, 0), window_end=dt.time(20, 0)
        )
        assert analyzer.expected_window_quarters == 12

        index = local_quarters(day(2026, 11, 2), 96)
        daily = analyzer.analyze(
            analyzer.prepare_net_offtake(frame(index, flat_evening_profile))
        ).daily.collect()
        assert daily.row(0, named=True)["observed_window_quarters"] == 12

    def test_frame_with_too_many_value_columns_is_rejected(self):
        bad = pl.LazyFrame({"timestamp": [day(2026, 11, 2)], "a": [1.0], "b": [2.0]})
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        with pytest.raises(ValueError, match="exactly one value column"):
            analyzer.prepare_net_offtake(bad)

    def test_frame_without_timestamp_is_rejected(self):
        bad = pl.LazyFrame({"when": [day(2026, 11, 2)], "a": [1.0]})
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        with pytest.raises(ValueError, match="must have a 'timestamp' column"):
            analyzer.prepare_net_offtake(bad)


class TestDuplicateTimestamps:
    """prepare_net_offtake is a public entry point in its own right, exercised directly
    throughout this file — it needs its own guard against repeated timestamps, not just
    EveningPeakInput's."""

    def test_duplicate_offtake_timestamp_is_rejected(self):
        index = local_quarters(day(2026, 11, 2), 8)
        duplicated = frame(index + index[:1], flat_evening_profile)
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)

        with pytest.raises(ValueError, match="grossOfftake has 1 repeated timestamp"):
            analyzer.prepare_net_offtake(duplicated)

    def test_duplicate_injection_timestamp_is_rejected(self):
        """Under the how="left" join a duplicate here would fan out the matching offtake
        row, not just shadow a value, so it needs checking independently of offtake."""
        index = local_quarters(day(2026, 11, 2), 8)
        offtake = frame(index, flat_evening_profile)
        duplicated_injection = frame(index + index[:1], lambda _: 0.0, name="gross_injection")
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)

        with pytest.raises(ValueError, match="grossInjection has 1 repeated timestamp"):
            analyzer.prepare_net_offtake(offtake, duplicated_injection)

    def test_a_repeated_timestamp_would_otherwise_double_count_energy(self):
        """Without the guard, sending the same window twice doubles the reported kWh and
        can satisfy the full-window coverage check on measurements that were never taken."""
        index = local_quarters(day(2026, 11, 2, 16, 0), 8)  # 16:00-18:00, 8 quarters
        duplicated = frame(index + index, lambda _: 0.5)
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)

        with pytest.raises(ValueError, match="repeated timestamp"):
            analyzer.prepare_net_offtake(duplicated)


class TestNonFiniteValues:
    """A bare NaN in the wire body must not poison the analysis."""

    def test_nan_is_dropped_not_summed(self):
        index = local_quarters(day(2026, 11, 2), 4)
        values = [0.1, float("nan"), 0.3, float("inf")]
        offtake = frame(index, lambda t: values[index.index(t)])
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)

        net = analyzer.prepare_net_offtake(offtake).collect()

        assert net.height == 2
        assert net["net_offtake_in_kilowatthour"].to_list() == pytest.approx([0.1, 0.3])


class TestTimezoneHandling:
    """The analysis timezone decides where day boundaries fall."""

    def test_naive_timestamps_are_read_as_utc(self):
        index = [day(2026, 11, 2, 15, 0) + dt.timedelta(minutes=15 * i) for i in range(8)]
        naive = pl.LazyFrame({"timestamp": index, "gross_offtake": [1.0] * 8})
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        net = analyzer.prepare_net_offtake(naive).collect()

        # 15:00 UTC is 16:00 in Amsterdam in November.
        assert net["timestamp"][0].hour == 16
        daily = analyzer.analyze(net.lazy()).daily.collect()
        assert daily.row(0, named=True)["observed_window_quarters"] == 8

    def test_the_same_instants_split_differently_in_another_timezone(self):
        """A day boundary in one zone is mid-evening in another."""
        index = local_quarters(day(2026, 11, 2), 96)
        offtake = frame(index, flat_evening_profile)

        amsterdam = EveningPeakAnalyzer(timezone="Europe/Amsterdam")
        lisbon = EveningPeakAnalyzer(timezone="Europe/Lisbon")

        ams = amsterdam.analyze(amsterdam.prepare_net_offtake(offtake)).daily.collect()
        lis = lisbon.analyze(lisbon.prepare_net_offtake(offtake)).daily.collect()

        assert ams.height == 1
        # Lisbon is an hour behind, so the Amsterdam day spans two Lisbon days.
        assert lis.height == 2


class TestResultSchema:
    """The result schema is a real guard, not decoration."""

    @staticmethod
    def _daily_frame(**overrides) -> pl.DataFrame:
        row = {
            "day": day(2026, 11, 2),
            "evening_peak_in_kilowatt": 2.0,
            "evening_peak_share_in_percent": 40.0,
            "daily_offtake_in_kilowatthour": 10.0,
            "evening_offtake_in_kilowatthour": 4.0,
            "observed_quarters": 96,
            "expected_quarters": 96,
            "observed_window_quarters": 20,
            "has_full_window": True,
            "is_complete": True,
            "is_below_threshold": False,
        }
        row.update(overrides)
        return pl.DataFrame({key: [value] for key, value in row.items()}).with_columns(
            pl.col("day").dt.replace_time_zone(TIMEZONE)
        )

    def test_a_share_above_one_hundred_percent_is_rejected(self):
        """This is the check that would catch a regression in injection clipping."""
        import pandera.errors

        with pytest.raises(pandera.errors.SchemaError):
            DailyEveningPeakSchema.validate(self._daily_frame(evening_peak_share_in_percent=140.0))

    def test_a_negative_offtake_is_rejected(self):
        import pandera.errors

        with pytest.raises(pandera.errors.SchemaError):
            DailyEveningPeakSchema.validate(self._daily_frame(daily_offtake_in_kilowatthour=-1.0))

    def test_validation_leaves_the_timezone_intact(self):
        """Coercion must not strip the zone the whole analysis depends on."""
        validated = DailyEveningPeakSchema.validate(self._daily_frame())
        assert validated.schema["day"].time_zone == TIMEZONE

    def test_analysis_output_keeps_its_timezone_through_validation(self):
        index = local_quarters(day(2026, 11, 2), 96)
        _, daily, result = analyze(frame(index, flat_evening_profile))

        assert daily.schema["day"].time_zone == TIMEZONE
        assert result.week_medians.collect().schema["week"].time_zone == TIMEZONE


class TestGaps:
    """A gap in the feed must be visible in the result, not inferable from it."""

    @pytest.fixture
    def week_with_a_hole(self) -> pl.LazyFrame:
        """Seven calendar days, with the middle three entirely missing."""
        index = [
            timestamp
            for timestamp in local_quarters(day(2026, 11, 2), 7 * 96)
            if timestamp.date()
            not in {dt.date(2026, 11, 4), dt.date(2026, 11, 5), dt.date(2026, 11, 6)}
        ]
        return frame(index, flat_evening_profile)

    def test_unmeasured_days_are_kept_as_rows(self, week_with_a_hole):
        _, daily, _ = analyze(week_with_a_hole)

        assert daily.height == 7, "the index spans the whole range, gap included"
        missing = daily.filter(pl.col("observed_quarters") == 0)
        assert missing.height == 3
        assert [row["day"].date() for row in missing.iter_rows(named=True)] == [
            dt.date(2026, 11, 4),
            dt.date(2026, 11, 5),
            dt.date(2026, 11, 6),
        ]

    def test_unmeasured_days_carry_nulls_not_zeroes(self, week_with_a_hole):
        """A day with no data has no peak; reporting 0 kW would be a lie."""
        _, daily, _ = analyze(week_with_a_hole)
        missing = daily.filter(pl.col("observed_quarters") == 0)

        assert missing["evening_peak_in_kilowatt"].null_count() == 3
        assert missing["evening_peak_share_in_percent"].null_count() == 3
        assert missing["is_below_threshold"].null_count() == 3
        assert not any(missing["is_complete"].to_list())

    def test_gaps_do_not_count_towards_the_measured_days(self, week_with_a_hole):
        _, _, result = analyze(week_with_a_hole)
        stats = summarize(result, peak_share_threshold=0.37)

        assert stats["measured_days"] == 4
        assert stats["first_day"] == dt.date(2026, 11, 2)
        assert stats["last_day"] == dt.date(2026, 11, 8)

    def test_the_daily_index_is_gapless_and_ordered(self, week_with_a_hole):
        _, daily, _ = analyze(week_with_a_hole)
        days = [row["day"] for row in daily.iter_rows(named=True)]

        assert days == sorted(days)
        gaps = {(later - earlier).days for earlier, later in itertools.pairwise(days)}
        assert gaps == {1}


class TestPreparedSeriesContract:
    """``prepare_net_offtake`` promises a frame that satisfies ``NetOfftakeSchema``.

    ``NetOfftakeSchema.validate()`` is now wired into ``prepare_net_offtake`` itself, on
    the production path — it stays lazy there, so it only checks columns and dtypes, not
    values (pandera runs no value checks on a ``LazyFrame``). This locks the columns/dtype
    contract in at the boundary; these tests add the value-level checks pandera can't do
    lazily, by validating the collected result directly.
    """

    def test_prepared_series_satisfies_the_schema(self):
        index = local_quarters(day(2026, 6, 15), 96)
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        prepared = analyzer.prepare_net_offtake(
            frame(index, flat_evening_profile),
            frame(index, lambda t: 3.0 if 10 <= t.hour < 16 else 0.0, name="gross_injection"),
        ).collect()

        NetOfftakeSchema.validate(prepared)
        assert prepared["net_offtake_in_kilowatthour"].min() >= 0
        assert prepared.schema["timestamp"].time_zone == TIMEZONE

    def test_prepare_net_offtake_returns_an_already_validated_lazyframe(self):
        """The schema wiring in prepare_net_offtake must not have been silently dropped,
        and must not force an eager collect — it should only ever check columns/dtypes."""
        index = local_quarters(day(2026, 6, 15), 96)
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        prepared = analyzer.prepare_net_offtake(frame(index, flat_evening_profile))

        assert isinstance(prepared, pl.LazyFrame)
        # A second pass must be a no-op if the columns/dtypes already match the schema.
        NetOfftakeSchema.validate(prepared).collect()

    def test_prepared_series_is_sorted_and_free_of_nulls(self):
        index = local_quarters(day(2026, 11, 2), 96)
        analyzer = EveningPeakAnalyzer(timezone=TIMEZONE)
        prepared = analyzer.prepare_net_offtake(frame(index, flat_evening_profile)).collect()

        timestamps = prepared["timestamp"].to_list()
        assert timestamps == sorted(timestamps)
        assert prepared["net_offtake_in_kilowatthour"].null_count() == 0
