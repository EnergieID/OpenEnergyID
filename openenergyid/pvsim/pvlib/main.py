"""
PVLib-based simulator implementation.

Defines a PVSimulator subclass that uses PVLib's ModelChain and weather data
to simulate PV system performance.
"""

import datetime as dt

import pandas as pd
import pvlib
from pydantic import BaseModel, ValidationError, field_validator

from openenergyid.pvsim.abstract import PVSimulator

from .models import PVLibModelChain, PVLibPVWattsModelChain


class PVLibSimulationInput(BaseModel):
    """
    Input parameters for the PVLibSimulator.
    """

    modelchain: PVLibModelChain | PVLibPVWattsModelChain
    start: dt.date
    end: dt.date

    @field_validator("modelchain", mode="before")
    @classmethod
    def validate_modelchain(cls, v):
        # Try to parse the input into a regular PVLibModelChain
        try:
            mc = PVLibModelChain.model_validate(v)
            # Try exporting to PVLib object
            mc.create_modelchain()
            return mc
        except (ValidationError, ValueError):
            # If that fails, try parsing as a PVLibPVWattsModelChain
            try:
                mc = PVLibPVWattsModelChain.model_validate(v)
                mc.create_modelchain()
                return mc
            except (ValidationError, ValueError) as e:
                raise ValueError("Invalid modelchain input") from e


class PVLibSimulator(PVSimulator):
    """
    Simulator for PV systems using PVLib's ModelChain and weather data.
    """

    def __init__(
        self,
        start: dt.date,
        end: dt.date,
        modelchain: pvlib.modelchain.ModelChain,
        weather: pd.DataFrame,
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
        mc = input_.modelchain.create_modelchain()
        weather = input_.modelchain.get_weather(start=input_.start, end=input_.end)

        return cls(start=input_.start, end=input_.end, modelchain=mc, weather=weather)
