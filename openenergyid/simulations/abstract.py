from abc import ABC, abstractmethod
from typing import Self

import pandas as pd
from aiohttp import ClientSession
from pydantic import BaseModel


class SimulationInputAbstract(BaseModel):
    """Abstract input parameters for any Simulation"""

    type: str


class Simulator(ABC):
    """
    An abstract base class simulators.
    """

    @property
    @abstractmethod
    def simulation_results(self):
        """The results of the simulation."""
        raise NotImplementedError()

    @abstractmethod
    def simulate(self, **kwargs):
        """
        Run the simulation and return the results.
        """
        raise NotImplementedError()

    @abstractmethod
    def result_as_frame(self) -> pd.DataFrame:
        """
        Convert the simulation results to a DataFrame.
        """
        raise NotImplementedError()

    @classmethod
    def from_pydantic(cls, input_: SimulationInputAbstract) -> Self:
        """
        Create an instance of the simulator from Pydantic input data.
        """
        return cls(**input_.model_dump())

    @abstractmethod
    async def load_resources(self, session: ClientSession) -> None:
        """
        Asynchronously load any required resources using the provided session.
        """
        raise NotImplementedError()
