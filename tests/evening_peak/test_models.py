"""Tests for the evening peak avoidance wire models."""

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from openenergyid.evening_peak import (
    EveningPeakAnalyzer,
    EveningPeakInput,
    EveningPeakOutput,
)

from .conftest import TIMEZONE, day, flat_evening_profile, local_quarters

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_SAMPLE_PATH = REPO_ROOT / "data" / "evening_peak" / "evening_peak_sample.json"


def payload(days: int = 3, *, with_injection: bool = True, **overrides) -> dict:
    """A request body covering whole local days."""
    index = local_quarters(day(2026, 11, 2), days * 96)
    stamps = [timestamp.isoformat() for timestamp in index]
    body = {
        "timeZone": TIMEZONE,
        "grossOfftake": {
            "index": stamps,
            "data": [flat_evening_profile(timestamp) for timestamp in index],
        },
        "reference": "EA-14214640",
    }
    if with_injection:
        body["grossInjection"] = {"index": stamps, "data": [0.0] * len(index)}
    body.update(overrides)
    return body


def run(model: EveningPeakInput) -> EveningPeakOutput:
    """Drive the analysis exactly as the endpoint does."""
    analyzer = EveningPeakAnalyzer(
        timezone=model.timezone,
        window_start=model.window_start,
        window_end=model.window_end,
        peak_share_threshold=model.peak_share_threshold,
        min_day_coverage=model.min_day_coverage,
    )
    offtake, injection = model.to_polars()
    net = analyzer.prepare_net_offtake(offtake, injection)
    return EveningPeakOutput.from_result(
        analyzer.analyze(net),
        analyzer.peak_moments(net, num_peaks=model.num_peak_moments),
        peak_share_threshold=model.peak_share_threshold,
        reference=model.reference,
    )


class TestInput:
    """Request parsing."""

    def test_documented_example_validates(self):
        example = EveningPeakInput.model_config["json_schema_extra"]["example"]
        model = EveningPeakInput.model_validate(example)

        assert model.window_start == dt.time(16, 0)
        assert model.window_end == dt.time(21, 0)
        assert model.reference == "EA-14214640"

    def test_defaults_match_the_offerte(self):
        model = EveningPeakInput.model_validate(payload(1))

        assert model.timezone == "Europe/Amsterdam"
        assert model.window_start == dt.time(16, 0)
        assert model.window_end == dt.time(21, 0)
        assert model.peak_share_threshold == 0.37
        assert model.num_peak_moments == 10

    def test_injection_is_optional(self):
        model = EveningPeakInput.model_validate(payload(1, with_injection=False))
        assert model.gross_injection is None
        assert model.to_polars()[1] is None

    def test_accepts_snake_case_field_names_too(self):
        body = payload(1)
        body["window_start"] = body.pop("windowStart", "17:00")
        model = EveningPeakInput.model_validate(body)
        assert model.window_start == dt.time(17, 0)

    @pytest.mark.parametrize("value", ["16:00", "16:00:00"])
    def test_window_accepts_both_time_spellings(self, value):
        model = EveningPeakInput.model_validate(payload(1, windowStart=value))
        assert model.window_start == dt.time(16, 0)

    def test_reversed_window_is_rejected(self):
        with pytest.raises(ValidationError, match="strictly before"):
            EveningPeakInput.model_validate(payload(1, windowStart="21:00", windowEnd="16:00"))

    def test_window_not_a_multiple_of_fifteen_minutes_is_rejected(self):
        """A 10-minute window would floor expected_window_quarters to 0, making every
        coverage check on it vacuously true."""
        with pytest.raises(ValidationError, match="not a whole number"):
            EveningPeakInput.model_validate(payload(1, windowStart="16:00", windowEnd="16:10"))

    def test_custom_quarter_hour_window_is_accepted(self):
        model = EveningPeakInput.model_validate(payload(1, windowStart="17:00", windowEnd="20:00"))
        assert model.window_start == dt.time(17, 0)
        assert model.window_end == dt.time(20, 0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("peakShareThreshold", 1.5),
            ("peakShareThreshold", -0.1),
            ("numPeakMoments", -1),
            ("numPeakMoments", 51),
            ("minDayCoverage", 1.2),
        ],
    )
    def test_out_of_range_parameters_are_rejected(self, field, value):
        with pytest.raises(ValidationError):
            EveningPeakInput.model_validate(payload(1, **{field: value}))

    def test_mismatched_index_and_data_length_is_rejected(self):
        body = payload(1)
        body["grossOfftake"]["data"] = body["grossOfftake"]["data"][:-1]
        with pytest.raises(ValidationError, match="timestamps but"):
            EveningPeakInput.model_validate(body)


def aware(naive: dt.datetime, timezone: str = TIMEZONE) -> dt.datetime:
    """Attach a zone to a naive local timestamp, for constructing test data at a
    granularity :func:`~tests.evening_peak.conftest.local_quarters` doesn't produce.

    Safe only away from a DST fold/gap — every use here stays within a single November
    day, so a plain `.replace` is unambiguous; anything DST-sensitive should go through
    `local_quarters` instead.
    """
    return naive.replace(tzinfo=ZoneInfo(timezone))


def series_at(index: list[dt.datetime], value_of=flat_evening_profile) -> dict:
    """A grossOfftake/grossInjection body from a list of timestamps."""
    return {
        "index": [timestamp.isoformat() for timestamp in index],
        "data": [value_of(timestamp) for timestamp in index],
    }


class TestMeasurementGrid:
    """EveningPeakInput enforces the "PT15M and nothing else" contract for real.

    Alignment alone (every timestamp on a quarter-hour boundary) is not sufficient —
    hourly data lands on :00 every time, which trivially satisfies alignment, so density
    (the minimum gap between consecutive present timestamps must be exactly 15 minutes)
    is checked as well. The two catch different things: data shifted off the true grid
    but still 15 minutes apart passes density and fails alignment; hourly/half-hourly data
    passes alignment and fails density.
    """

    def test_genuine_quarter_hourly_data_with_a_gap_passes(self):
        """A mid-series outage must not itself look like a granularity violation — the
        *minimum* gap is what matters, not every gap."""
        index = local_quarters(day(2026, 11, 2), 96)
        gapped = index[:20] + index[40:]
        model = EveningPeakInput.model_validate(
            {"timeZone": TIMEZONE, "grossOfftake": series_at(gapped)}
        )
        assert len(model.gross_offtake.index) == len(gapped)

    def test_the_committed_demo_fixture_satisfies_the_grid(self):
        """The real shipped sample, gap and all, must not false-positive."""
        if not DEMO_SAMPLE_PATH.exists():
            pytest.skip(f"{DEMO_SAMPLE_PATH} not present")
        EveningPeakInput.model_validate_json(DEMO_SAMPLE_PATH.read_text(encoding="utf-8"))

    def test_naive_timestamp_is_rejected(self):
        body = payload(1)
        body["grossOfftake"]["index"][0] = body["grossOfftake"]["index"][0][:-6]  # drop +01:00
        with pytest.raises(ValidationError, match="timezone-aware"):
            EveningPeakInput.model_validate(body)

    def test_misaligned_but_fifteen_minute_spaced_data_is_rejected(self):
        """Every point is 15 minutes from the next, but none of them is on the grid."""
        shifted = [
            timestamp + dt.timedelta(minutes=7) for timestamp in local_quarters(day(2026, 11, 2), 8)
        ]
        with pytest.raises(ValidationError, match="not aligned to a quarter-hour"):
            EveningPeakInput.model_validate(
                {"timeZone": TIMEZONE, "grossOfftake": series_at(shifted)}
            )

    def test_hourly_data_is_rejected(self):
        """Passes alignment (:00 is a valid boundary) but fails density."""
        hourly = [aware(day(2026, 11, 2, hour)) for hour in range(24)]
        with pytest.raises(ValidationError, match="minimum gap of 60"):
            EveningPeakInput.model_validate(
                {"timeZone": TIMEZONE, "grossOfftake": series_at(hourly)}
            )

    def test_half_hourly_data_is_rejected(self):
        half_hourly = [
            aware(day(2026, 11, 2, hour, minute)) for hour in range(24) for minute in (0, 30)
        ]
        with pytest.raises(ValidationError, match="minimum gap of 30"):
            EveningPeakInput.model_validate(
                {"timeZone": TIMEZONE, "grossOfftake": series_at(half_hourly)}
            )

    def test_five_minute_data_is_rejected(self):
        """Fails both alignment and density; either error message is acceptable."""
        five_minute = [aware(day(2026, 11, 2, 16)) + dt.timedelta(minutes=5 * i) for i in range(12)]
        with pytest.raises(ValidationError, match="not aligned|minimum gap"):
            EveningPeakInput.model_validate(
                {"timeZone": TIMEZONE, "grossOfftake": series_at(five_minute)}
            )

    def test_duplicate_timestamp_is_rejected(self):
        index = local_quarters(day(2026, 11, 2), 8)
        with pytest.raises(ValidationError, match="repeated timestamp"):
            EveningPeakInput.model_validate(
                {"timeZone": TIMEZONE, "grossOfftake": series_at(index + index[:1])}
            )

    def test_nan_value_is_rejected(self):
        body = payload(1)
        body["grossOfftake"]["data"][3] = float("nan")
        with pytest.raises(ValidationError, match="non-finite"):
            EveningPeakInput.model_validate(body)

    def test_infinite_value_is_rejected(self):
        body = payload(1)
        body["grossOfftake"]["data"][3] = float("inf")
        with pytest.raises(ValidationError, match="non-finite"):
            EveningPeakInput.model_validate(body)

    def test_a_none_value_is_not_a_violation(self):
        """None represents a legitimate gap in the reading, not bad data."""
        body = payload(1)
        body["grossOfftake"]["data"][3] = None
        EveningPeakInput.model_validate(body)

    def test_multiple_violations_are_reported_together(self):
        """A caller with more than one problem should see all of them in one round trip,
        not just the first one found.

        The naive violation is put on index[0] and the duplicate on index[1], deliberately
        on different timestamps: stripping the offset from a duplicated entry would change
        its type and stop it comparing equal to its own copy, silently hiding the
        duplicate rather than testing the two violations together.
        """
        index = local_quarters(day(2026, 11, 2), 8)
        naive_and_duplicated = series_at(index + index[1:2])
        naive_and_duplicated["index"][0] = naive_and_duplicated["index"][0][:-6]
        with pytest.raises(ValidationError, match=r"timezone-aware.*repeated timestamp"):
            EveningPeakInput.model_validate(
                {"timeZone": TIMEZONE, "grossOfftake": naive_and_duplicated}
            )

    def test_length_mismatch_is_reported_as_a_length_error_not_a_grid_error(self):
        """validate_measurement_grid is declared after validate_series_lengths, so a
        length-mismatched payload must fail with the length message even when it would
        also trip the grid validator (a naive timestamp here)."""
        body = payload(1)
        body["grossOfftake"]["index"][0] = body["grossOfftake"]["index"][0][:-6]
        body["grossOfftake"]["data"] = body["grossOfftake"]["data"][:-1]
        with pytest.raises(ValidationError, match="timestamps but") as excinfo:
            EveningPeakInput.model_validate(body)
        assert "timezone-aware" not in str(excinfo.value)

    def test_injection_is_checked_independently_of_offtake(self):
        body = payload(1)
        body["grossInjection"]["index"][0] = body["grossInjection"]["index"][0][:-6]
        with pytest.raises(ValidationError, match="grossInjection.*timezone-aware"):
            EveningPeakInput.model_validate(body)

    def test_omitted_injection_is_not_checked(self):
        EveningPeakInput.model_validate(payload(1, with_injection=False))

    def test_empty_series_has_nothing_to_violate(self):
        EveningPeakInput.model_validate({"timeZone": TIMEZONE, "grossOfftake": series_at([])})

    def test_single_timestamp_has_no_gap_to_measure(self):
        index = local_quarters(day(2026, 11, 2), 1)
        EveningPeakInput.model_validate({"timeZone": TIMEZONE, "grossOfftake": series_at(index)})


class TestOutput:
    """Response shape — this is the contract the front end builds against."""

    @pytest.fixture
    def response(self) -> dict:
        model = EveningPeakInput.model_validate(payload(14))
        return json.loads(run(model).model_dump_json(by_alias=True))

    def test_top_level_keys_are_camel_case(self, response):
        assert set(response) == {
            "dailyPeak",
            "dailyShare",
            "weekMedianPeak",
            "weekMedianShare",
            "peakMoments",
            "summary",
            "reference",
        }

    def test_summary_keys_are_camel_case(self, response):
        assert set(response["summary"]) == {
            "averagePeak",
            "lowestPeak",
            "highestPeak",
            "averageShare",
            "lowestShare",
            "highestShare",
            "daysBelowThreshold",
            "measuredDays",
            "thresholdInPercent",
            "firstDay",
            "lastDay",
        }

    def test_daily_series_are_aligned_and_one_point_per_day(self, response):
        assert len(response["dailyPeak"]["data"]) == 14
        assert response["dailyPeak"]["index"] == response["dailyShare"]["index"]

    def test_week_medians_are_shorter_than_the_daily_series(self, response):
        assert len(response["weekMedianPeak"]["data"]) == 2
        assert response["weekMedianPeak"]["index"] == response["weekMedianShare"]["index"]

    def test_peak_moments_carry_a_full_day_curve(self, response):
        moments = response["peakMoments"]
        assert len(moments) == 10
        assert set(moments[0]) == {"peakTime", "peakValue", "dayCurve"}
        assert len(moments[0]["dayCurve"]["data"]) == 96

    def test_reference_is_echoed_back(self, response):
        assert response["reference"] == "EA-14214640"

    def test_summary_fraction_is_consistent_with_the_daily_series(self, response):
        summary = response["summary"]
        measured = [value for value in response["dailyShare"]["data"] if value is not None]

        assert summary["measuredDays"] == len(measured)
        assert summary["daysBelowThreshold"] <= summary["measuredDays"]
        assert summary["highestShare"] == pytest.approx(max(measured))
        assert summary["lowestShare"] == pytest.approx(min(measured))

    def test_output_survives_a_json_round_trip(self, response):
        assert EveningPeakOutput.model_validate(response).reference == "EA-14214640"

    def test_empty_series_produce_an_empty_but_valid_response(self):
        model = EveningPeakInput.model_validate(
            {
                "timeZone": TIMEZONE,
                "grossOfftake": {"index": [], "data": []},
            }
        )
        response = json.loads(run(model).model_dump_json(by_alias=True))

        assert response["dailyPeak"]["data"] == []
        assert response["peakMoments"] == []
        assert response["summary"]["measuredDays"] == 0
        assert response["summary"]["firstDay"] is None


class TestTimezoneValidation:
    """An unknown timezone is a bad request, not a runtime failure."""

    def test_unknown_timezone_is_rejected(self):
        with pytest.raises(ValidationError, match="not a known IANA time zone"):
            EveningPeakInput.model_validate(payload(1, timeZone="Mars/Olympus"))

    @pytest.mark.parametrize("zone", ["Europe/Amsterdam", "Europe/Brussels", "UTC"])
    def test_real_timezones_are_accepted(self, zone):
        assert EveningPeakInput.model_validate(payload(1, timeZone=zone)).timezone == zone
