"""Tests for the Elia PV load factor helpers across the ods032 -> ods209/ods210 switch."""

import asyncio
import datetime as dt

import pandas as pd
import pytest

from openenergyid import elia
from openenergyid.elia import pv as pv_module
from openenergyid.elia.const import PV_REGION_MEMBERS, Region


class FakeElia:
    """Stands in for get_dataset: answers per dataset and records every call."""

    def __init__(self, responses: dict[str, list[dict]]):
        self.responses = responses
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.get(kwargs["dataset"], [])


def _run(coro):
    return asyncio.run(coro)


def _rows(timestamp: str, regions, production_field: str, production, capacity):
    return [
        {
            "datetime": timestamp,
            "region": r,
            production_field: production,
            "monitoredcapacity": capacity,
        }
        for r in regions
    ]


@pytest.fixture
def fake(monkeypatch):
    def install(responses):
        fake_elia = FakeElia(responses)
        monkeypatch.setattr(pv_module, "get_dataset", fake_elia)
        return fake_elia

    return install


def test_range_before_cutover_reads_only_ods032(fake):
    fake_elia = fake({"ods032": [{"datetime": "2026-10-20T12:00:00+02:00", "loadfactor": 31.7}]})

    result = _run(
        elia.get_pv_load_factors(
            dt.date(2026, 10, 20), dt.date(2026, 10, 21), Region.Flanders, None
        )
    )

    assert [c["dataset"] for c in fake_elia.calls] == ["ods032"]
    assert fake_elia.calls[0]["region"] is Region.Flanders
    assert result.tolist() == [31.7]


def test_range_after_cutover_reads_only_ods209_total_rows(fake):
    fake_elia = fake(
        {
            "ods209": _rows(
                "2026-11-02T12:00:00+01:00", ["Belgium"], "measured_upscaled", 1211.0, 12110.0
            )
        }
    )

    result = _run(
        elia.get_pv_load_factors(dt.date(2026, 11, 1), dt.date(2026, 11, 2), Region.Belgium, None)
    )

    (call,) = fake_elia.calls
    assert call["dataset"] == "ods209"
    assert list(call["regions"]) == ["Belgium"]
    assert call["extra_where"] == "gridconnectiontype='Total'"
    assert result.tolist() == pytest.approx([10.0])


def test_range_across_cutover_splits_without_overlap(fake):
    fake_elia = fake(
        {
            "ods032": [{"datetime": "2026-10-25T23:45:00+01:00", "loadfactor": 0.0}],
            "ods209": _rows(
                "2026-10-26T00:00:00+01:00", ["Brussel"], "measured_upscaled", 0.0, 365.0
            ),
        }
    )

    result = _run(
        elia.get_pv_load_factors(dt.date(2026, 10, 20), dt.date(2026, 11, 3), Region.Brussels, None)
    )

    legacy, new = fake_elia.calls
    assert (legacy["dataset"], legacy["start"], legacy["end"]) == (
        "ods032",
        dt.date(2026, 10, 20),
        dt.date(2026, 10, 25),
    )
    assert (new["dataset"], new["start"], new["end"]) == (
        "ods209",
        dt.date(2026, 10, 26),
        dt.date(2026, 11, 3),
    )
    assert list(new["regions"]) == ["Brussel"]
    assert len(result) == 2
    assert result.index.is_monotonic_increasing


def test_flanders_sums_its_provinces(fake):
    provinces = PV_REGION_MEMBERS[Region.Flanders]
    fake(
        {
            "ods209": _rows(
                "2026-11-02T12:00:00+01:00", provinces, "measured_upscaled", 100.0, 1000.0
            )
        }
    )

    result = _run(
        elia.get_pv_load_factors(dt.date(2026, 11, 2), dt.date(2026, 11, 2), Region.Flanders, None)
    )

    assert len(provinces) == 5
    assert result.tolist() == pytest.approx([10.0])


def test_timestamp_missing_a_province_is_dropped(fake):
    provinces = PV_REGION_MEMBERS[Region.Wallonia]
    rows = _rows("2026-11-02T12:00:00+01:00", provinces, "measured_upscaled", 50.0, 500.0)
    rows += _rows("2026-11-02T12:15:00+01:00", provinces[:-1], "measured_upscaled", 50.0, 500.0)
    fake({"ods209": rows})

    result = _run(
        elia.get_pv_load_factors(dt.date(2026, 11, 2), dt.date(2026, 11, 2), Region.Wallonia, None)
    )

    assert result.index.tolist() == [pd.Timestamp("2026-11-02T12:00:00+01:00")]


def test_text_with_decimal_comma_is_parsed(fake):
    fake(
        {
            "ods209": _rows(
                "2026-11-02T12:00:00+01:00", ["Namur"], "measured_upscaled", "41,5", "415,0"
            )
        }
    )

    result = _run(
        elia.get_pv_load_factors(dt.date(2026, 11, 2), dt.date(2026, 11, 2), Region.Namur, None)
    )

    assert result.tolist() == pytest.approx([10.0])


def test_forecast_reads_ods210(fake):
    fake_elia = fake(
        {"ods210": _rows("2026-11-05T12:00:00+01:00", ["Liege"], "mostrecentforecast", 96.0, 960.0)}
    )

    result = _run(
        elia.get_pv_load_factor_forecast(
            dt.date(2026, 11, 5), dt.date(2026, 11, 12), Region.Liège, None
        )
    )

    (call,) = fake_elia.calls
    assert call["dataset"] == "ods210"
    assert list(call["regions"]) == ["Liege"]
    assert result.tolist() == pytest.approx([10.0])


def test_error_response_raises(fake):
    fake({"ods209": {"error_code": "ODSQLError", "message": "boom"}})

    with pytest.raises(TypeError, match="ods209"):
        _run(
            elia.get_pv_load_factors(
                dt.date(2026, 11, 2), dt.date(2026, 11, 2), Region.Belgium, None
            )
        )


def test_every_region_has_members():
    assert set(PV_REGION_MEMBERS) == set(Region)
