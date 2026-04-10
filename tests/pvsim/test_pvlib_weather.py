import asyncio
import datetime as dt
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
            (51.2, 4.4, dt.date(2024, 1, 1), dt.date(2024, 1, 2), "UTC", 7),
        )
    ]


def test_get_weather_sync_forwards_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    weather_frame = _make_weather_frame()

    def fake_get_pvgis_tmy(**kwargs):
        calls.append(kwargs)
        return weather_frame.copy(), {"meta": "ignored"}

    monkeypatch.setattr(weather_module.pvlib.iotools, "get_pvgis_tmy", fake_get_pvgis_tmy)

    result = weather_module._get_weather_sync(
        latitude=51.2,
        longitude=4.4,
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 2),
        tz="UTC",
        timeout=17,
    )

    assert len(result) == 96
    assert calls == [
        {
            "latitude": 51.2,
            "longitude": 4.4,
            "timeout": 17,
            "roll_utc_offset": 0,
        }
    ]


def test_get_weather_retries_transient_failures_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []
    expected = pd.DataFrame({"ghi": [1.0]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))

    def fake_get_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise requests.Timeout("temporary timeout")
        return expected

    async def fake_to_thread(func, *args):
        return func(*args)

    async def fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(weather_module, "_get_weather_sync", fake_get_weather_sync)
    monkeypatch.setattr(weather_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(weather_module.asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        weather_module.get_weather(
            latitude=51.2,
            longitude=4.4,
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 1, 2),
            tz="UTC",
            timeout=7,
            retry_count=2,
            retry_backoff_seconds=0.5,
        )
    )

    assert result is expected
    assert attempts["count"] == 3
    assert sleep_calls == [0.5, 1.0]


def test_get_weather_raises_after_retry_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def fake_get_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        raise requests.ConnectionError("network down")

    async def fake_to_thread(func, *args):
        return func(*args)

    async def fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(weather_module, "_get_weather_sync", fake_get_weather_sync)
    monkeypatch.setattr(weather_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(weather_module.asyncio, "sleep", fake_sleep)

    with pytest.raises(requests.ConnectionError, match="network down"):
        asyncio.run(
            weather_module.get_weather(
                latitude=51.2,
                longitude=4.4,
                start=dt.date(2024, 1, 1),
                end=dt.date(2024, 1, 2),
                tz="UTC",
                retry_count=2,
                retry_backoff_seconds=0.25,
            )
        )

    assert attempts["count"] == 3
    assert sleep_calls == [0.25, 0.5]


def test_get_weather_does_not_retry_non_retryable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def fake_get_weather_sync(*args, **kwargs):
        attempts["count"] += 1
        raise requests.HTTPError("bad request")

    async def fake_to_thread(func, *args):
        return func(*args)

    async def fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(weather_module, "_get_weather_sync", fake_get_weather_sync)
    monkeypatch.setattr(weather_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(weather_module.asyncio, "sleep", fake_sleep)

    with pytest.raises(requests.HTTPError, match="bad request"):
        asyncio.run(
            weather_module.get_weather(
                latitude=51.2,
                longitude=4.4,
                start=dt.date(2024, 1, 1),
                end=dt.date(2024, 1, 2),
                tz="UTC",
                retry_count=2,
                retry_backoff_seconds=0.25,
            )
        )

    assert attempts["count"] == 1
    assert sleep_calls == []
