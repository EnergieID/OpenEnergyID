"""Data models for energy sharing."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field
import pandas as pd

from openenergyid import TimeDataFrame
from .data_formatting import create_multi_index_input_frame
from .const import NET_INJECTION, NET_OFFTAKE, SHARED_ENERGY


class CalculationMethod(Enum):
    """Calculation method for energy sharing."""

    FIXED = "Fixed"
    RELATIVE = "Relative"
    OPTIMAL = "Optimal"


class EnergySharingInput(BaseModel):
    """Input data for energy sharing."""

    gross_injection: Annotated[TimeDataFrame, Field(alias="grossInjection")]
    gross_offtake: Annotated[TimeDataFrame, Field(alias="grossOfftake")]
    key: Annotated[TimeDataFrame, Field(alias="key")]
    timezone: str = Field(alias="timeZone", default="Europe/Brussels")

    def data_frame(self) -> pd.DataFrame:
        """Return the data as a combined DataFrame"""
        df = create_multi_index_input_frame(
            gross_injection=self.gross_injection.to_pandas(),
            gross_offtake=self.gross_offtake.to_pandas(),
            key=self.key.to_pandas(),
        )
        df = df.tz_convert(self.timezone)
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
