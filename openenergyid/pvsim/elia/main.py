"""
This module contains the LoadFactorPVSimulator class which
simulates the power output of a PV system based on load factors.
"""

import datetime as dt
from typing import Literal

import aiohttp
import pandas as pd
from pydantic import Field

from openenergyid import elia
from openenergyid.pvsim.abstract import PVSimulationInputAbstract, PVSimulator


class EliaPVSimulationInput(PVSimulationInputAbstract):
    """Input parameters for the Elia PV simulation."""

    type: Literal["eliapvsimulation"] = Field("eliapvsimulation", frozen=True)  # tag
    region: elia.Region
    panel_power: float = Field(..., gt=0, description="Installed panel power in W")
    inverter_power: float = Field(..., gt=0, description="Installed inverter power in W")


class EliaPVSimulator(PVSimulator):
    """
    A PV simulator that simulates the power output of a PV system based on load factors.
    """

    def __init__(
        self,
        start: dt.date,
        end: dt.date,
        panel_power: float,
        inverter_power: float,
        region: elia.Region,
        load_factors: pd.Series | None = None,
        **kwargs,
    ):
        self.start = start
        self.end = end
        self.panel_power = panel_power / 1000  # convert to kW
        self.inverter_power = inverter_power / 1000  # convert to kW
        self.region = region
        self.load_factors = load_factors if load_factors is not None else pd.Series(dtype=float)

        super().__init__(**kwargs)

    def simulate(self, **kwargs) -> pd.Series:
        """Run the simulation."""
        result = self.load_factors * self.panel_power * 0.01
        result.clip(upper=self.inverter_power, inplace=True)
        result = result * 0.25  # To Energy
        result.name = None
        return result

    @staticmethod
    async def download_load_factors(
        start: dt.date,
        end: dt.date,
        region: elia.Region,
        session: aiohttp.ClientSession,
    ) -> pd.Series:
        """Download load factors from the API."""
        pv_load_factor_json = await elia.get_dataset(
            dataset="ods032",
            start=start,
            end=end,
            region=region,
            select={"loadfactor"},
            session=session,
        )
        pv_load_factors = elia.parse_response(
            data=pv_load_factor_json, index="datetime", columns=["loadfactor"]
        )["loadfactor"]
        pv_load_factors = pv_load_factors.truncate(
            after=pd.Timestamp(end, tz="Europe/Brussels") - pd.Timedelta(minutes=15)
        )
        return pv_load_factors

    async def load_resources(self, session: aiohttp.ClientSession) -> None:
        """Load resources required for the simulation."""
        self.load_factors = await self.download_load_factors(
            start=self.start,
            end=self.end,
            region=self.region,
            session=session,
        )
