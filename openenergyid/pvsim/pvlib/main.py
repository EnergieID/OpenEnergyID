"""
PVLib-based simulator implementation.

Defines a PVSimulator subclass that uses PVLib's ModelChain and weather data
to simulate PV system performance.
"""

import datetime as dt

import pandas as pd
import pvlib
from pvlib.modelchain import ModelChain
from pydantic import BaseModel

from openenergyid.pvsim.abstract import PVSimulator

from .models import ModelChainModel, to_pv
from .weather import get_weather


class PVLibSimulationInput(BaseModel):
    """
    Input parameters for the PVLibSimulator.
    """

    modelchain: ModelChainModel  # type: ignore
    start: dt.date
    end: dt.date


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
    ):
        """
        Initialize the simulator with a ModelChain and weather DataFrame.

        Args:
            modelchain: An instance of pvlib.modelchain.ModelChain.
            weather: Weather data as a pandas DataFrame.
        """
        super().__init__()

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

        return cls(start=input_.start, end=input_.end, modelchain=mc)

    def load_weather(self):
        """
        Load weather data for the simulation period.
        """
        weather = get_weather(
            latitude=self.modelchain.location.latitude,
            longitude=self.modelchain.location.longitude,
            start=self.start,
            end=self.end,
            tz=self.modelchain.location.tz,
        )
        self.weather = weather
