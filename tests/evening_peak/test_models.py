"""Tests for the evening peak avoidance wire models."""

import datetime as dt
import json

import pytest
from pydantic import ValidationError

from openenergyid.evening_peak import (
    EveningPeakAnalyzer,
    EveningPeakInput,
    EveningPeakOutput,
)

from .conftest import TIMEZONE, day, flat_evening_profile, local_quarters


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
