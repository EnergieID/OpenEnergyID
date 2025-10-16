from abc import ABC, abstractmethod
from typing import Annotated, Self

import pandas as pd
from aiohttp import ClientSession
from pydantic import BaseModel, Field

from ..simeval.models import ComparisonPayload, EvalPayload, EvaluationOutput


class SimulationInputAbstract(BaseModel):
    """Abstract input parameters for any Simulation"""

    type: str


class SimulationSummary(BaseModel):
    """Summary of a simulation including ex-ante, simulation results, ex-post, and comparisons."""

    ex_ante: Annotated[EvalPayload, Field(description="Ex-ante evaluation results.")]
    simulation_result: Annotated[EvalPayload, Field(description="Simulation results.")]
    ex_post: Annotated[EvalPayload, Field(description="Ex-post evaluation results.")]
    comparison: Annotated[
        ComparisonPayload, Field(description="Comparison between ex-ante and ex-post results.")
    ]

    @classmethod
    def from_simulation(
        cls,
        ex_ante: dict[str, pd.DataFrame | pd.Series],
        simulation_result: dict[str, pd.DataFrame | pd.Series],
        ex_post: dict[str, pd.DataFrame | pd.Series],
        comparison: dict[str, dict[str, pd.DataFrame | pd.Series]],
    ) -> Self:
        """Create a SimulationSummary from simulation data."""
        ea = {
            k: EvaluationOutput.from_pandas(v) if isinstance(v, pd.DataFrame) else v.to_dict()
            for k, v in ex_ante.items()
        }
        sr = {
            k: EvaluationOutput.from_pandas(v) if isinstance(v, pd.DataFrame) else v.to_dict()
            for k, v in simulation_result.items()
        }
        ep = {
            k: EvaluationOutput.from_pandas(v) if isinstance(v, pd.DataFrame) else v.to_dict()
            for k, v in ex_post.items()
        }
        c = {
            k: {
                kk: EvaluationOutput.from_pandas(vv)
                if isinstance(vv, pd.DataFrame)
                else vv.to_dict()
                for kk, vv in v.items()
            }
            for k, v in comparison.items()
        }
        return cls(
            ex_ante=ea,  # type: ignore
            simulation_result=sr,  # type: ignore
            ex_post=ep,  # type: ignore
            comparison=c,  # type: ignore
        )


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
