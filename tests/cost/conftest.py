"""Shared fixtures for contract-based cost calculation tests."""

import zoneinfo

import numpy as np
import pandas as pd
import pytest
from energy_cost import Contract
from energy_cost.index import DataFrameIndex, Index

from openenergyid import const

TZ = zoneinfo.ZoneInfo("Europe/Brussels")
MOCK_INDEX = "MockSpot"


@pytest.fixture(autouse=True)
def isolate_index_registry():
    """Snapshot and restore the global index registry around every test.

    ``Index`` is a process-global per-subclass registry, so a test that registers or
    drops an index would otherwise leak into every test after it.
    """
    saved = dict(Index.items())
    yield
    Index.clear()
    for name, index in saved.items():
        Index.register(name, index)


def make_index(
    name: str = MOCK_INDEX,
    start: str = "2023-12-01",
    end: str = "2025-02-01",
    constant: float | None = None,
) -> DataFrameIndex:
    """Register a synthetic quarter-hourly EUR/MWh index and return it."""
    stamps = pd.date_range(start, end, freq="15min", tz=TZ)
    if constant is None:
        # A daily sine, so time-shifting a battery has something to exploit.
        values = 90 + 60 * np.sin(np.arange(len(stamps)) / 96 * 2 * np.pi)
    else:
        values = np.full(len(stamps), constant)
    index = DataFrameIndex(
        pd.DataFrame({"timestamp": stamps, "value": values}),
        resolution=pd.Timedelta("15min"),
    )
    Index.register(name, index)
    return index


@pytest.fixture
def mock_index() -> DataFrameIndex:
    return make_index()


@pytest.fixture
def flat_index() -> DataFrameIndex:
    """A flat 100 EUR/MWh index, for arithmetic that is easy to verify by hand."""
    return make_index(constant=100.0)


def make_contract(
    supplier_consumption: float = 100.0,
    supplier_injection: float | None = -50.0,
    index_name: str = MOCK_INDEX,
    with_index: bool = False,
) -> Contract:
    """A full Flanders residential contract with an inline supplier tariff.

    Injection rates are negative by convention: ``energy_cost`` sums every cost group
    into the bill total, so a feed-in payment has to be a negative cost.
    """
    consumption: dict = {"energy": {"constant_cost": supplier_consumption}}
    if with_index:
        consumption["energy"]["variable_costs"] = [{"index": index_name, "scalar": 1.0}]

    version: dict = {
        "start": "2023-01-01T00:00:00+01:00",
        "consumption": consumption,
        "fixed": {"subscription": {"period": "P1M", "constant_cost": 5.0}},
    }
    if supplier_injection is not None:
        version["injection"] = {"energy": {"constant_cost": supplier_injection}}

    return Contract.model_validate(
        {
            "start": "2023-01-01T00:00:00+01:00",
            "region": "be_flanders",
            "connection_type": "electricity",
            "customer_type": "residential",
            "distributor_key": "fluvius_imewo",
            "supplier": [version],
        }
    )


@pytest.fixture
def simple_contract() -> Contract:
    return make_contract()


def make_frame(
    start: str = "2024-01-01",
    periods: int = 96 * 366,
    delivered: float = 0.05,
    exported: float | None = 0.02,
    produced: float | None = 0.03,
    freq: str = "15min",
) -> pd.DataFrame:
    """A tz-aware simulation frame in kWh per interval."""
    index = pd.date_range(start, periods=periods, freq=freq, tz=TZ)
    data = {const.ELECTRICITY_DELIVERED: np.full(periods, delivered)}
    if exported is not None:
        data[const.ELECTRICITY_EXPORTED] = np.full(periods, exported)
    if produced is not None:
        data[const.ELECTRICITY_PRODUCED] = np.full(periods, produced)
    return pd.DataFrame(data, index=index)


@pytest.fixture
def year_frame() -> pd.DataFrame:
    """A full leap year of quarter-hourly data, so no window warnings fire."""
    return make_frame()


@pytest.fixture
def month_frame() -> pd.DataFrame:
    return make_frame(periods=96 * 31)
