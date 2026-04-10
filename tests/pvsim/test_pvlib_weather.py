import asyncio
import datetime as dt
import threading
import time
import typing
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from openenergyid.pvsim.pvlib import main as pvlib_main_module
from openenergyid.pvsim.pvlib import weather as weather_module
from openenergyid.pvsim.pvlib.main import PVLibSimulationInput, PVLibSimulator


def _make_pvlib_input(**overrides) -> PVLibSimulationInput:
    payload = {
        "type": "pvlibsimulation",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "timeout": 12,
        "retry_count": 3,
        "retry_backoff_seconds": 0.5,
        "modelchain": {
            "type": "quickscan",
            "system": {
                "surface_tilt": 35,
                "surface_azimuth": 180,
                "p_inverter": 2000,
                "modules_per_string": 6,
                "strings_per_inverter": 1,
            },
            "location": {
                "latitude": 51.2,
                "longitude": 4.4,
                "tz": "UTC",
            },
        },
    }
    payload.update(overrides)
    return PVLibSimulationInput.model_validate(payload)


def _make_weather_frame() -> pd.DataFrame:
    index = pd.date_range("1990-01-01", periods=25, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "temp_air": pd.Series(range(25), index=index, dtype=float),
            "ghi": pd.Series(range(100, 125), index=index, dtype=float),
        },
        index=index,
    )


async def _awaitable(value):
    return value


@pytest.fixture(autouse=True)
def clear_weather_cache() -> typing.Iterator[None]:
    weather_module.clear_weather_cache()
    yield
    weather_module.clear_weather_cache()


def _make_dummy_modelchain(tz: str = "UTC") -> SimpleNamespace:
    location = SimpleNamespace(latitude=51.2, longitude=4.4, tz=tz)
    return SimpleNamespace(location=location)


def test_pvlib_input_accepts_and_serializes_request_controls() -> None:
    input_model = _make_pvlib_input()
    dumped = input_model.model_dump()
    schema = PVLibSimulationInput.model_json_schema()

    assert input_model.timeout == 12
    assert input_model.retry_count == 3
    assert input_model.retry_backoff_seconds == pytest.approx(0.5)
    assert dumped["timeout"] == 12
    assert dumped["retry_count"] == 3
    assert dumped["retry_backoff_seconds"] == pytest.approx(0.5)
    assert "timeout" in schema["properties"]
    assert "retry_count" in schema["properties"]
    assert "retry_backoff_seconds" in schema["properties"]


def test_from_pydantic_stores_timeout_and_retry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    input_model = _make_pvlib_input()
    dummy_modelchain = _make_dummy_modelchain()

    monkeypatch.setattr(pvlib_main_module, "to_pv", lambda _: dummy_modelchain)

    simulator = PVLibSimulator.from_pydantic(input_model)

    assert simulator.modelchain is dummy_modelchain
    assert simulator.timeout == 12
    assert simulator.retry_count == 3
    assert simulator.retry_backoff_seconds == pytest.approx(0.5)


def test_load_resources_awaits_weather_loader_and_stores_weather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame({"ghi": [1.0]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
    simulator = PVLibSimulator(
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        modelchain=_make_dummy_modelchain(),
        timeout=9,
        retry_count=4,
        retry_backoff_seconds=0.25,
    )
    calls: list[dict[str, object]] = []

    async def fake_get_weather(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(pvlib_main_module, "get_weather", fake_get_weather)

    asyncio.run(simulator.load_resources())

    assert simulator.weather is expected
    assert calls == [
        {
            "latitude": 51.2,
            "longitude": 4.4,
            "start": dt.date(2024, 1, 1),
            "end": dt.date(2024, 1, 2),
            "tz": "UTC",
            "timeout": 9,
            "retry_count": 4,
            "retry_backoff_seconds": 0.25,
        }
    ]


def test_get_weather_uses_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = pd.DataFrame({"ghi": [1.0]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
    to_thread_calls: list[tuple[object, tuple[object, ...]]] = []

    def fake_get_weather_sync(*args, **kwargs):
        raise AssertionError("sync helper should be invoked through asyncio.to_thread")

    async def fake_to_thread(func, *args):
        to_thread_calls.append((func, args))
        return expected

    monkeypatch.setattr(weather_module, "_get_weather_sync", fake_get_weather_sync)
    monkeypatch.setattr(weather_module.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(
        weather_module.get_weather(
            latitude=51.2,
            longitude=4.4,
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 1, 2),
            tz="UTC",
            timeout=7,
        )
    )

    assert result is expected
    assert to_thread_calls == [
        (
            fake_get_weather_sync,
            (51.2, 4.4, dt.date(2024, 1, 1), dt.date(2024, 1, 2), "UTC", 7, 2, 1.0),
        )
    ]


def test_download_normal_year_weather_sync_forwards_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    weather_frame = _make_weather_frame()

    def fake_get_pvgis_tmy(**kwargs):
        calls.append(kwargs)
        return weather_frame.copy(), {"meta": "ignored"}

    monkeypatch.setattr(weather_module.pvlib.iotools, "get_pvgis_tmy", fake_get_pvgis_tmy)

    result = weather_module._download_normal_year_weather_sync(
        latitude=51.2,
        longitude=4.4,
        tz="UTC",
        timeout=17,
    )

    assert result.equals(weather_frame)
    assert calls == [
        {
            "latitude": 51.2,
            "longitude": 4.4,
            "timeout": 17,
            "roll_utc_offset": 0,
        }
    ]


def test_same_site_reuses_cached_normal_year_across_simulator_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    normal_year_weather = _make_weather_frame()

    def fake_download_normal_year_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        return normal_year_weather

    monkeypatch.setattr(
        weather_module,
        "_download_normal_year_weather_sync",
        fake_download_normal_year_weather_sync,
    )
    monkeypatch.setattr(
        weather_module.asyncio, "to_thread", lambda func, *args: _awaitable(func(*args))
    )

    simulator_one = PVLibSimulator(
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        modelchain=_make_dummy_modelchain(),
    )
    simulator_two = PVLibSimulator(
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        modelchain=_make_dummy_modelchain(),
    )

    asyncio.run(simulator_one.load_resources())
    asyncio.run(simulator_two.load_resources())

    assert attempts["count"] == 1
    assert simulator_one.weather is not None
    assert simulator_two.weather is not None
    assert simulator_one.weather.equals(simulator_two.weather)


def test_different_date_ranges_reuse_cached_normal_year(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    normal_year_weather = _make_weather_frame()

    def fake_download_normal_year_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        return normal_year_weather

    monkeypatch.setattr(
        weather_module,
        "_download_normal_year_weather_sync",
        fake_download_normal_year_weather_sync,
    )

    first = weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="UTC",
        timeout=30,
        retry_count=2,
        retry_backoff_seconds=1.0,
    )
    second = weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 2),
        end=dt.date(2024, 1, 3),
        tz="UTC",
        timeout=30,
        retry_count=2,
        retry_backoff_seconds=1.0,
    )

    assert attempts["count"] == 1
    assert len(first) == 96
    assert len(second) == 96
    assert first.index[0] == pd.Timestamp("2024-01-01 00:00:00+0000", tz="UTC")
    assert second.index[0] == pd.Timestamp("2024-01-02 00:00:00+0000", tz="UTC")


def test_different_location_or_timezone_uses_separate_cache_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    normal_year_weather = _make_weather_frame()

    def fake_download_normal_year_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        return normal_year_weather

    monkeypatch.setattr(
        weather_module,
        "_download_normal_year_weather_sync",
        fake_download_normal_year_weather_sync,
    )

    weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="UTC",
        timeout=30,
        retry_count=2,
        retry_backoff_seconds=1.0,
    )
    weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="Europe/Brussels",
        timeout=30,
        retry_count=2,
        retry_backoff_seconds=1.0,
    )
    weather_module._get_weather_sync(
        latitude=51.3,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="UTC",
        timeout=30,
        retry_count=2,
        retry_backoff_seconds=1.0,
    )

    assert attempts["count"] == 3


def test_cached_hit_ignores_later_transport_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    normal_year_weather = _make_weather_frame()

    def fake_download_normal_year_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        return normal_year_weather

    monkeypatch.setattr(
        weather_module,
        "_download_normal_year_weather_sync",
        fake_download_normal_year_weather_sync,
    )

    weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="UTC",
        timeout=10,
        retry_count=0,
        retry_backoff_seconds=0.1,
    )
    weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 2, 1),
        end=dt.date(2024, 2, 2),
        tz="UTC",
        timeout=90,
        retry_count=5,
        retry_backoff_seconds=5.0,
    )

    assert attempts["count"] == 1


def test_get_weather_retries_transient_failures_until_success_on_cold_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []
    normal_year_weather = _make_weather_frame()

    def fake_download_normal_year_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise requests.Timeout("temporary timeout")
        return normal_year_weather

    monkeypatch.setattr(
        weather_module,
        "_download_normal_year_weather_sync",
        fake_download_normal_year_weather_sync,
    )
    monkeypatch.setattr(weather_module.time, "sleep", sleep_calls.append)

    result = weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="UTC",
        timeout=7,
        retry_count=2,
        retry_backoff_seconds=0.5,
    )

    assert len(result) == 96
    assert attempts["count"] == 3
    assert sleep_calls == [0.5, 1.0]


def test_failed_download_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    normal_year_weather = _make_weather_frame()

    def fake_download_normal_year_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise requests.ConnectionError("network down")
        return normal_year_weather

    monkeypatch.setattr(
        weather_module,
        "_download_normal_year_weather_sync",
        fake_download_normal_year_weather_sync,
    )

    with pytest.raises(requests.ConnectionError, match="network down"):
        weather_module._get_weather_sync(
            latitude=51.2,
            longitude=4.4,
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 1, 2),
            tz="UTC",
            timeout=30,
            retry_count=0,
            retry_backoff_seconds=1.0,
        )

    result = weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="UTC",
        timeout=30,
        retry_count=0,
        retry_backoff_seconds=1.0,
    )

    assert attempts["count"] == 2
    assert len(result) == 96


def test_get_weather_does_not_retry_non_retryable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def fake_download_normal_year_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        raise requests.HTTPError("bad request")

    monkeypatch.setattr(
        weather_module,
        "_download_normal_year_weather_sync",
        fake_download_normal_year_weather_sync,
    )
    monkeypatch.setattr(weather_module.time, "sleep", sleep_calls.append)

    with pytest.raises(requests.HTTPError, match="bad request"):
        weather_module._get_weather_sync(
            latitude=51.2,
            longitude=4.4,
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 1, 2),
            tz="UTC",
            retry_count=2,
            retry_backoff_seconds=0.25,
            timeout=30,
        )

    assert attempts["count"] == 1
    assert sleep_calls == []


def test_parallel_same_key_cold_misses_share_one_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    normal_year_weather = _make_weather_frame()
    started = threading.Event()
    release = threading.Event()
    results: list[pd.DataFrame] = []

    def fake_download_normal_year_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        started.set()
        if not release.wait(timeout=1):
            raise TimeoutError("parallel test did not release in time")
        return normal_year_weather

    monkeypatch.setattr(
        weather_module,
        "_download_normal_year_weather_sync",
        fake_download_normal_year_weather_sync,
    )

    def run_request() -> None:
        result = weather_module._get_weather_sync(
            latitude=51.2,
            longitude=4.4,
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 1, 2),
            tz="UTC",
            timeout=30,
            retry_count=2,
            retry_backoff_seconds=1.0,
        )
        results.append(result)

    thread_one = threading.Thread(target=run_request)
    thread_two = threading.Thread(target=run_request)
    thread_one.start()

    try:
        assert started.wait(timeout=1)

        thread_two.start()
        time.sleep(0.05)
    finally:
        release.set()
        thread_one.join(timeout=1)
        thread_two.join(timeout=1)

    assert attempts["count"] == 1
    assert not thread_one.is_alive()
    assert not thread_two.is_alive()
    assert len(results) == 2
    assert results[0].equals(results[1])


def test_clear_weather_cache_forces_refetch(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    normal_year_weather = _make_weather_frame()

    def fake_download_normal_year_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        return normal_year_weather

    monkeypatch.setattr(
        weather_module,
        "_download_normal_year_weather_sync",
        fake_download_normal_year_weather_sync,
    )

    weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="UTC",
        timeout=30,
        retry_count=2,
        retry_backoff_seconds=1.0,
    )
    weather_module.clear_weather_cache()
    weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="UTC",
        timeout=30,
        retry_count=2,
        retry_backoff_seconds=1.0,
    )

    assert attempts["count"] == 2
