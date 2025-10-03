from typing import Annotated, Union

from pydantic import Field

from .elia import EliaPVSimulationInput, EliaPVSimulator
from .pvlib import PVLibSimulationInput, PVLibSimulator

PVSimulationInput = Annotated[
    Union[PVLibSimulationInput, EliaPVSimulationInput], Field(discriminator="type")
]


def get_simulator(input_: PVSimulationInput) -> PVLibSimulator | EliaPVSimulator:
    """Get an instance of the simulator based on the input data."""
    if isinstance(input_, PVLibSimulationInput):
        return PVLibSimulator.from_pydantic(input_)
    if isinstance(input_, EliaPVSimulationInput):
        return EliaPVSimulator.from_pydantic(input_)
    raise ValueError(f"Unknown simulator type: {input_.type}")
