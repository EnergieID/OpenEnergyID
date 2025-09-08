"""
PVLib-based simulator implementation.

Defines a PVSimulator subclass that uses PVLib's ModelChain and weather data
to simulate PV system performance.
"""

import pandas as pd
import pvlib

from openenergyid.pvsim.abstract import PVSimulator

from .models import PVLibModelChain


class PVLibSimulator(PVSimulator):
    """
    Simulator for PV systems using PVLib's ModelChain and weather data.
    """

    def __init__(self, modelchain: pvlib.modelchain.ModelChain, weather: pd.DataFrame):
        """
        Initialize the simulator with a ModelChain and weather DataFrame.

        Args:
            modelchain: An instance of pvlib.modelchain.ModelChain.
            weather: Weather data as a pandas DataFrame.
        """
        super().__init__()

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
    def from_pydantic(cls, modelchain: PVLibModelChain) -> "PVLibSimulator":
        """
        Create a PVLibSimulator instance from a PVLibModelChain model.

        Args:
            modelchain: A PVLibModelChain instance.

        Returns:
            PVLibSimulator: An initialized simulator.
        """
        mc = modelchain.create_modelchain()
        weather = modelchain.get_weather()

        return cls(modelchain=mc, weather=weather)
