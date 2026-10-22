import datetime as dt
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Self

import pandas as pd
from aiohttp import ClientSession
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..cost.models import CostSummary


class SimulationInputAbstract(BaseModel):
    """Abstract input parameters for any Simulation"""

    type: str


class SimulationWarningCode(StrEnum):
    """Machine-readable reasons why a simulation result may be unreliable."""

    NO_INJECTION_DATA = "no_injection_data"
    ENERGY_DATA_INCOMPLETE = "energy_data_incomplete"


class SimulationWarning(BaseModel):
    """A caveat attached to a simulation result, about the ex-ante data it was based on."""

    code: SimulationWarningCode
    message: str
    context: dict = Field(default_factory=dict)


class DataCoverage(BaseModel):
    """How much measured data the simulation was based on."""

    start: dt.datetime | None = Field(None, description="First ex-ante timestamp.")
    end: dt.datetime | None = Field(None, description="Last ex-ante timestamp.")
    days_with_delivered: int = Field(
        description="Days (in the simulation timezone) with at least one offtake value."
    )
    days_with_exported: int = Field(
        description="Days (in the simulation timezone) with at least one injection value."
    )
    intervals: int = Field(description="Number of ex-ante intervals.")
    intervals_incomplete: int = Field(
        description=(
            "Intervals missing offtake, or missing injection while injection data "
            "exists for other intervals."
        )
    )


class SimulationSummary(BaseModel):
    """Summary of a simulation including ex-ante, simulation results, ex-post, and comparisons."""

    ex_ante: Annotated[dict, Field(description="Ex-ante evaluation results.")]
    simulation_result: Annotated[dict | list[dict], Field(description="Simulation results.")]
    ex_post: Annotated[dict, Field(description="Ex-post evaluation results.")]
    comparison: Annotated[
        dict, Field(description="Comparison between ex-ante and ex-post results.")
    ]
    cost: Annotated[
        "CostSummary | None",
        Field(
            default=None,
            description=(
                "Contract-based cost of the simulation before and after, with the "
                "savings breakdown. Present only when a tariff contract was supplied."
            ),
        ),
    ] = None
    coverage: Annotated[
        DataCoverage | None,
        Field(default=None, description="How much measured data the simulation was based on."),
    ] = None
    warnings: Annotated[
        list[SimulationWarning],
        Field(
            default_factory=list,
            description="Caveats about the ex-ante data. Cost caveats are in `cost.warnings`.",
        ),
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


# Resolve the forward reference on SimulationSummary.cost. The cost models live in
# openenergyid.cost, which does not import abstractsim, so this is safe to do eagerly.
from ..cost.models import (  # noqa: E402  pylint: disable=wrong-import-position
    CostSummary,
)

SimulationSummary.model_rebuild()
