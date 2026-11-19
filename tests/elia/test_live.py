"""
Live checks against the Elia open data API, so a renamed region or field shows up here
before it shows up as missing production.

They need network access and are skipped unless ELIA_LIVE=1:

    ELIA_LIVE=1 uv run pytest tests/elia/test_live.py
"""

import asyncio
import datetime as dt
import os

import aiohttp
import pytest

from openenergyid import elia
from openenergyid.elia.const import Region

pytestmark = pytest.mark.skipif(
    os.environ.get("ELIA_LIVE") != "1", reason="calls the Elia API; set ELIA_LIVE=1"
)


async def _with_session(fetch):
    async with aiohttp.ClientSession() as session:
        return await fetch(session)


@pytest.mark.parametrize("region", list(Region), ids=lambda r: r.name)
def test_measured_load_factors_for_every_region(region):
    day = dt.date.today() - dt.timedelta(days=7)

    result = asyncio.run(
        _with_session(lambda session: elia.get_pv_load_factors(day, day, region, session))
    )

    assert len(result) >= 92  # 96 quarter hours, 92 or 100 on a DST day
    assert result.between(0, 100).all()


def test_forecast_for_flanders():
    today = dt.date.today()

    result = asyncio.run(
        _with_session(
            lambda session: elia.get_pv_load_factor_forecast(
                today, today + dt.timedelta(days=1), Region.Flanders, session
            )
        )
    )

    assert not result.empty
    assert result.between(0, 100).all()
