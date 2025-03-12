from typing import cast
import pandas as pd
from .abstract import PVSimulator


class LoadFactorPVSimulator(PVSimulator):
    def __init__(
        self, load_factors: pd.Series, panel_power: float, inverter_power: float, **kwargs
    ):
        self.load_factors = load_factors
        self.panel_power = panel_power
        self.inverter_power = inverter_power

        self._simulation_results: pd.Series | None = None

    @property
    def simulation_results(self) -> pd.Series:
        if self._simulation_results is None:
            self.simulate()
            self._simulation_results = cast(pd.Series, self._simulation_results)
        return self._simulation_results

    def simulate(self, **kwargs) -> pd.Series:
        result = self.load_factors * self.panel_power * 0.01
        result.clip(upper=self.inverter_power, inplace=True)
        result.rename("power", inplace=True)
        self._simulation_results = result
        return result
