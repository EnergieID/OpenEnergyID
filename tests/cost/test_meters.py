"""Tests for mapping simulation frames onto energy_cost meters."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from openenergyid import const
from openenergyid.cost import CostWarningCode, detect_frame_resolution, frame_to_meters
from openenergyid.cost.meters import KWH_PER_MWH

from .conftest import TZ, make_frame

QUARTER_HOUR = dt.timedelta(minutes=15)


class TestDetectFrameResolution:
    def test_uses_index_freq_when_set(self) -> None:
        index = pd.date_range("2024-01-01", periods=100, freq="15min", tz=TZ)
        assert detect_frame_resolution(index) == QUARTER_HOUR

    def test_infers_when_freq_is_absent(self) -> None:
        index = pd.date_range("2024-01-01", periods=100, freq="15min", tz=TZ)
        stripped = pd.DatetimeIndex(index.to_list())
        assert stripped.freq is None
        assert detect_frame_resolution(stripped) == QUARTER_HOUR

    def test_modal_fallback_survives_a_gap(self) -> None:
        index = pd.date_range("2024-01-01", periods=400, freq="15min", tz=TZ)
        gapped = pd.DatetimeIndex(index[:100].to_list() + index[300:].to_list())
        assert pd.infer_freq(gapped) is None
        assert detect_frame_resolution(gapped) == QUARTER_HOUR

    @pytest.mark.parametrize(
        ("label", "start"),
        [("spring-forward", "2024-03-30"), ("fall-back", "2024-10-26")],
    )
    def test_survives_dst_transitions(self, label: str, start: str) -> None:
        """A tz-aware index keeps constant deltas across a DST change."""
        index = pd.date_range(start, periods=96 * 3, freq="15min", tz=TZ)
        assert detect_frame_resolution(pd.DatetimeIndex(index.to_list())) == QUARTER_HOUR

    def test_hourly_data(self) -> None:
        index = pd.date_range("2024-01-01", periods=100, freq="h", tz=TZ)
        assert detect_frame_resolution(index) == dt.timedelta(hours=1)

    def test_rejects_too_few_timestamps(self) -> None:
        index = pd.date_range("2024-01-01", periods=1, freq="15min", tz=TZ)
        with pytest.raises(ValueError, match="fewer than 2"):
            detect_frame_resolution(index)

    def test_rejects_non_monotonic_index(self) -> None:
        index = pd.date_range("2024-01-01", periods=10, freq="15min", tz=TZ)
        with pytest.raises(ValueError, match="non-monotonic"):
            detect_frame_resolution(index[::-1])


class TestFrameToMeters:
    def test_converts_kwh_to_mwh(self) -> None:
        frame = make_frame(periods=96, delivered=0.5, exported=0.25)
        consumption, injection, warnings = frame_to_meters(frame)

        assert warnings == []
        assert injection is not None
        expected = frame[const.ELECTRICITY_DELIVERED].sum() / KWH_PER_MWH
        assert consumption.measurements["value"].sum() == pytest.approx(expected)

    def test_injection_volumes_stay_positive(self) -> None:
        """The sign of feed-in lives in the tariff rate, never in the volume."""
        frame = make_frame(periods=96, exported=0.25)
        _, injection, _ = frame_to_meters(frame)

        assert injection is not None
        assert (injection.measurements["value"] >= 0).all()

    def test_timestamps_are_utc(self) -> None:
        """energy_cost merges against frames it builds in UTC."""
        frame = make_frame(periods=96)
        consumption, _, _ = frame_to_meters(frame)

        assert str(consumption.measurements["timestamp"].dt.tz) == "UTC"

    def test_nan_energy_is_filled_and_warned(self) -> None:
        """energy_cost voids a whole billing period containing a NaN."""
        frame = make_frame(periods=96)
        frame.iloc[3, frame.columns.get_loc(const.ELECTRICITY_DELIVERED)] = np.nan
        frame.iloc[7, frame.columns.get_loc(const.ELECTRICITY_DELIVERED)] = np.nan

        consumption, _, warnings = frame_to_meters(frame)

        codes = [w.code for w in warnings]
        assert CostWarningCode.ENERGY_DATA_GAPS_FILLED in codes
        filled = next(w for w in warnings if w.code == CostWarningCode.ENERGY_DATA_GAPS_FILLED)
        assert filled.context["filled"] == 2
        assert not consumption.measurements["value"].isna().any()

    def test_missing_export_column_yields_no_injection_meter(self) -> None:
        frame = make_frame(periods=96, exported=None)
        _, injection, warnings = frame_to_meters(frame)

        assert injection is None
        assert CostWarningCode.NO_INJECTION_DATA in [w.code for w in warnings]

    def test_all_nan_export_column_yields_no_injection_meter(self) -> None:
        frame = make_frame(periods=96)
        frame[const.ELECTRICITY_EXPORTED] = np.nan
        _, injection, warnings = frame_to_meters(frame)

        assert injection is None
        assert CostWarningCode.NO_INJECTION_DATA in [w.code for w in warnings]

    def test_rejects_naive_index(self) -> None:
        frame = make_frame(periods=96)
        frame.index = frame.index.tz_localize(None)
        with pytest.raises(ValueError, match="timezone-aware"):
            frame_to_meters(frame)

    def test_rejects_non_datetime_index(self) -> None:
        frame = make_frame(periods=96).reset_index(drop=True)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            frame_to_meters(frame)

    def test_requires_delivered_column(self) -> None:
        frame = make_frame(periods=96).drop(columns=[const.ELECTRICITY_DELIVERED])
        with pytest.raises(ValueError, match=const.ELECTRICITY_DELIVERED):
            frame_to_meters(frame)

    def test_does_not_mutate_the_input_frame(self) -> None:
        frame = make_frame(periods=96)
        frame.iloc[3, frame.columns.get_loc(const.ELECTRICITY_DELIVERED)] = np.nan
        before = frame.copy()

        frame_to_meters(frame)

        pd.testing.assert_frame_equal(frame, before)
