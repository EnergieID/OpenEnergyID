"""
This module contains the LoadFactorPVSimulator class which
simulates the power output of a PV system based on load factors.
"""

import pandas as pd
from .abstract import PVSimulator


class LoadFactorPVSimulator(PVSimulator):
    """
    A PV simulator that simulates the power output of a PV system based on load factors.
    """

    def __init__(self, load_factors: pd.Series, panel_power: float, inverter_power: float):
        self.load_factors = load_factors
        self.panel_power = panel_power
        self.inverter_power = inverter_power

        super().__init__()

    def simulate(self, **kwargs) -> pd.Series:
        """Run the simulation."""
        result = self.load_factors * self.panel_power * 0.01
        result.clip(upper=self.inverter_power, inplace=True)
        result.rename("power", inplace=True)
        self._simulation_results = result
        return result
