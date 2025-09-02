"""
This module contains the abstract base class for PVSimulator.
"""

from abc import ABC, abstractmethod
from typing import cast

import pandas as pd

from openenergyid.models import TimeSeries


class PVSimulator(ABC):
    """
    An abstract base class for PV simulators.
    """

    def __init__(self):
        self._simulation_results: pd.Series | None = None

    @property
    def simulation_results(self) -> pd.Series:
        """The results of the simulation."""
        if self._simulation_results is None:
            results = self.simulate()
            self._simulation_results = cast(pd.Series, results)
        return self._simulation_results

    @abstractmethod
    def simulate(self, **kwargs) -> pd.Series:
        """
        Run the simulation and return the results as a Series.
        """
        raise NotImplementedError()

    def result_to_timeseries(self):
        """
        Convert the simulation results to a TimeSeries object.
        """
        return TimeSeries.from_pandas(self.simulation_results)
