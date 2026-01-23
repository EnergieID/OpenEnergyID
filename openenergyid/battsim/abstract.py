"""Abstract base classes for battery simulators."""

from abc import ABC
from typing import cast

import pandas as pd
from aiohttp import ClientSession
from pydantic import Field

from openenergyid.models import TimeDataFrame

from .. import const
from ..abstractsim import SimulationInputAbstract, Simulator


class BatterySimulationInputAbstract(SimulationInputAbstract):
    """
    Input parameters for the battery simulation.
    """

    result_resolution: str = Field(
        "15min",
        description="Resolution of the simulation results",
        examples=["15min", "1h", "D", "MS"],
    )


class BatterySimulator(Simulator, ABC):
    """
    An abstract base class for battery simulators.
    """

    def __init__(self, result_resolution: str = "15min", **kwargs) -> None:
        self._simulation_results: pd.DataFrame | None = None
        self.result_resolution = result_resolution

    async def load_resources(self, session: ClientSession | None = None) -> None:
        return

    def result_as_frame(self) -> pd.DataFrame:
        return self.simulation_results

    @property
    def simulation_results(self) -> pd.DataFrame:
        """The results of the simulation."""
        if self._simulation_results is None:
            results = self.simulate()
            self._simulation_results = cast(pd.DataFrame, results)
        return self._simulation_results

    def result_to_timedataframe(self) -> TimeDataFrame:
        """
        Convert the simulation results to a TimeDataFrame object.
        """
        result = self.simulation_results.resample(self.result_resolution).agg(
            {
                const.ELECTRICITY_DELIVERED: "sum",
                const.ELECTRICITY_EXPORTED: "sum",
                const.STATE_OF_ENERGY: "mean",
                const.BATTERY_CYCLES: "sum",
                const.ELECTRICITY_CHARGED: "sum",
                const.ELECTRICITY_DISCHARGED: "sum",
            }
        )
        return TimeDataFrame.from_pandas(result)
