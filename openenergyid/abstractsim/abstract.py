from abc import ABC, abstractmethod
from typing import Annotated, Self

import pandas as pd
from aiohttp import ClientSession
from pydantic import BaseModel, Field


class SimulationInputAbstract(BaseModel):
    """Abstract input parameters for any Simulation"""

    type: str


class SimulationSummary(BaseModel):
    """Summary of a simulation including ex-ante, simulation results, ex-post, and comparisons."""

    ex_ante: Annotated[dict, Field(description="Ex-ante evaluation results.")]
    simulation_result: Annotated[dict | list[dict], Field(description="Simulation results.")]
    ex_post: Annotated[dict, Field(description="Ex-post evaluation results.")]
    comparison: Annotated[
        dict, Field(description="Comparison between ex-ante and ex-post results.")
    ]


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
    def from_pydantic(cls, input_: SimulationInputAbstract, **kwargs) -> Self:
        """
        Create an instance of the simulator from Pydantic input data.
        """
        return cls(**input_.model_dump(), **kwargs)

    @abstractmethod
    async def load_resources(self, session: ClientSession | None = None) -> None:
        """
        Asynchronously load any required resources using the provided session.
        """
        raise NotImplementedError()
