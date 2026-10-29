"""
This module contains the abstract base class for PVSimulator.
"""

import datetime as dt
from abc import ABC
from typing import cast

import pandas as pd
from pydantic import Field

from openenergyid.models import TimeSeries

from ..abstractsim import SimulationInputAbstract, Simulator
from ..const import ELECTRICITY_PRODUCED


class PVSimulationInputAbstract(SimulationInputAbstract):
    """
    Input parameters for the PV simulation.
    """

    start: dt.date
    end: dt.date
    result_resolution: str = Field(
        "15min",
        description="Resolution of the simulation results",
        examples=["15min", "1h", "D", "MS"],
    )
    degradation_rate: float = Field(
        0.0,
        ge=0,
        lt=1,
        description=(
            "Yearly loss of production as a fraction (0.005 is 0.5 % per year). Only "
            "used to project the savings over the years (`financial`); the simulated "
            "period itself is year 1, without degradation."
        ),
        examples=[0.0, 0.005],
    )


class PVSimulator(Simulator, ABC):
    """
    An abstract base class for PV simulators.
    """

    def __init__(self, result_resolution: str = "15min", **kwargs) -> None:
        self._simulation_results: pd.Series | None = None
        self.result_resolution = result_resolution

    @property
    def simulation_results(self) -> pd.Series:
        """The results of the simulation."""
        if self._simulation_results is None:
            results = self.simulate()
            self._simulation_results = cast(pd.Series, results)
        return self._simulation_results

    def result_to_timeseries(self):
        """
        Convert the simulation results to a TimeSeries object.
        """
        result = self.simulation_results.resample(self.result_resolution).sum()
        return TimeSeries.from_pandas(result)

    def result_as_frame(self) -> pd.DataFrame:
        """
        Convert the simulation results to a DataFrame.
        """
        return self.simulation_results.rename(ELECTRICITY_PRODUCED).to_frame()
