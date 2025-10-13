"""
PVLib-based simulator implementation.

Defines a PVSimulator subclass that uses PVLib's ModelChain and weather data
to simulate PV system performance.
"""

import datetime as dt
from typing import Annotated, Literal, Union

import pandas as pd
import pvlib
from aiohttp import ClientSession
from pvlib.modelchain import ModelChain
from pydantic import Field

from openenergyid.pvsim.abstract import PVSimulationInputAbstract, PVSimulator

from .models import ModelChainModel, to_pv
from .quickscan import (
    QuickScanModelChainModel,  # pyright: ignore[reportAttributeAccessIssue]
)
from .weather import get_weather

ModelChainUnion = Annotated[
    Union[ModelChainModel, QuickScanModelChainModel], Field(discriminator="type")
]


class PVLibSimulationInput(PVSimulationInputAbstract):
    """
    Input parameters for the PVLibSimulator.
    """

    type: Literal["pvlibsimulation"] = Field("pvlibsimulation", frozen=True)  # tag
    modelchain: ModelChainUnion


class PVLibSimulator(PVSimulator):
    """
    Simulator for PV systems using PVLib's ModelChain and weather data.
    """

    def __init__(
        self,
        start: dt.date,
        end: dt.date,
        modelchain: pvlib.modelchain.ModelChain,
        weather: pd.DataFrame | None = None,
        **kwargs,
    ):
        """
        Initialize the simulator with a ModelChain and weather DataFrame.

        Args:
            modelchain: An instance of pvlib.modelchain.ModelChain.
            weather: Weather data as a pandas DataFrame.
        """
        super().__init__(**kwargs)

        self.start = start
        self.end = end
        self.modelchain = modelchain
        self.weather = weather

    def simulate(self, **kwargs) -> pd.Series:
        """
        Run the simulation and return the resulting AC energy series.

        Returns:
            pd.Series: Simulated AC energy in kWh for each timestep.

        Raises:
            ValueError: If the AC power result is None.
        """
        # Run model
        self.modelchain.run_model(self.weather)

        results = self.modelchain.results
        ac = results.ac

        if ac is None:
            raise ValueError("AC power is None")

        # Convert W to kWh
        energy = ac * 0.25 / 1000

        energy.name = None

        return energy

    @classmethod
    def from_pydantic(cls, input_: PVLibSimulationInput) -> "PVLibSimulator":
        """
        Create a PVLibSimulator instance from a PVLibSimulationInput model.

        Args:
            input_: A PVLibSimulationInput instance.

        Returns:
            PVLibSimulator: An initialized simulator.
        """
        mc: ModelChain = to_pv(input_.modelchain)

        return cls(modelchain=mc, **input_.model_dump(exclude={"modelchain"}))

    async def load_resources(self, session: ClientSession | None = None) -> None:
        weather = get_weather(
            latitude=self.modelchain.location.latitude,
            longitude=self.modelchain.location.longitude,
            start=self.start,
            end=self.end,
            tz=self.modelchain.location.tz,
        )
        self.weather = weather
