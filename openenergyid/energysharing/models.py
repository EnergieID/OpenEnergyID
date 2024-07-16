"""Data models for energy sharing."""

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, confloat
import pandas as pd

from openenergyid import TimeDataFrame
from .data_formatting import create_multi_index_input_frame
from .const import NET_INJECTION, NET_OFFTAKE, SHARED_ENERGY


class CalculationMethod(Enum):
    """Calculation method for energy sharing."""

    FIXED = "Fixed"
    RELATIVE = "Relative"
    OPTIMAL = "Optimal"


class KeyInput(TimeDataFrame):
    """Energy Sharing Keys."""

    data: Annotated[
        list[list[confloat(ge=0.0, le=1.0)]],  # type: ignore
        Field(
            description="Key data, column per participant. "
            "Must be between 0 and 1. "
            "Each row must sum to 1."
        ),
    ]

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        for row in self.data:
            if round(sum(row), 6) != 1:
                raise ValueError("Each row must sum to 1.")
        return super().model_post_init(__context)


class EnergySharingInput(BaseModel):
    """Input data for energy sharing."""

    gross_injection: Annotated[
        TimeDataFrame,
        Field(alias="grossInjection", description="Gross injection data, column per participant"),
    ]
    gross_offtake: Annotated[
        TimeDataFrame,
        Field(alias="grossOfftake", description="Gross offtake data, column per participant"),
    ]
    key: KeyInput

    def to_pandas(self) -> pd.DataFrame:
        """Return the data as a combined DataFrame"""
        df = create_multi_index_input_frame(
            gross_injection=self.gross_injection.to_pandas(),
            gross_offtake=self.gross_offtake.to_pandas(),
            key=self.key.to_pandas(),
        )
        return df


class EnergySharingOutput(BaseModel):
    """Output data for energy sharing."""

    net_injection: TimeDataFrame = Field(alias="netInjection")
    net_offtake: TimeDataFrame = Field(alias="netOfftake")
    shared_energy: TimeDataFrame = Field(alias="sharedEnergy")

    @classmethod
    def from_calculation_result(cls, result: pd.DataFrame) -> "EnergySharingOutput":
        """Create an output model from a calculation result."""
        return cls.model_construct(
            net_injection=TimeDataFrame.from_pandas(result[NET_INJECTION]),
            net_offtake=TimeDataFrame.from_pandas(result[NET_OFFTAKE]),
            shared_energy=TimeDataFrame.from_pandas(result[SHARED_ENERGY]),
        )
