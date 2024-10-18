"""Models for dynamic tariff analysis."""

from typing import Literal
from pydantic import Field, conlist, confloat, BaseModel

from openenergyid.models import TimeDataFrame
from .const import Register


RequiredColumns = Literal[
    "electricity_delivered",
    "electricity_exported",
    "price_electricity_delivered",
    "price_electricity_exported",
    "RLP",
    "SPP",
]

OutputColumns = Literal[
    "electricity_delivered_smr3",
    "electricity_exported_smr3",
    "price_electricity_delivered",
    "price_electricity_exported",
    "RLP",
    "SPP",
    "electricity_delivered_smr2",
    "electricity_exported_smr2",
    "cost_electricity_delivered_smr2",
    "cost_electricity_exported_smr2",
    "cost_electricity_delivered_smr3",
    "cost_electricity_exported_smr3",
    "rlp_weighted_price_delivered",
    "spp_weighted_price_exported",
    "heatmap_delivered",
    "heatmap_exported",
    "heatmap_total",
    "heatmap_delivered_description",
    "heatmap_exported_description",
    "heatmap_total_description",
]


class DynamicTariffAnalysisInput(TimeDataFrame):
    """Input frame for dynamic tariff analysis."""

    columns: list[RequiredColumns] = Field(
        min_length=3,
        max_length=len(RequiredColumns.__args__),
        examples=[RequiredColumns.__args__],
    )
    data: list[
        conlist(
            item_type=confloat(allow_inf_nan=True),
            min_length=3,
            max_length=len(RequiredColumns.__args__),
        )  # type: ignore
    ] = Field(examples=[[0.0] * len(RequiredColumns.__args__)])

    @property
    def registers(self) -> list[Register]:
        """Check which registers are present in the input data."""
        registers = []
        columns = list(self.columns)
        # if "electricity_delivered", "price_electricity_delivered" and "RLP" are present
        if all(
            column in columns
            for column in [
                "electricity_delivered",
                "price_electricity_delivered",
                "RLP",
            ]
        ):
            registers.append(Register.DELIVERY)
        # if "electricity_exported", "price_electricity_exported" and "SPP" are present
        if all(
            column in columns
            for column in ["electricity_exported", "price_electricity_exported", "SPP"]
        ):
            registers.append(Register.EXPORT)
        return registers


class DynamicTariffAnalysisOutputSummary(BaseModel):
    """Summary of the dynamic tariff analysis output."""

    cost_electricity_delivered_smr2: float | None = None
    cost_electricity_delivered_smr3: float | None = None
    cost_electricity_exported_smr2: float | None = None
    cost_electricity_exported_smr3: float | None = None
    cost_electricity_total_smr2: float | None = None
    cost_electricity_total_smr3: float | None = None
    ratio: float | None = None


class DynamicTariffAnalysisOutput(TimeDataFrame):
    """Output frame for dynamic tariff analysis."""

    columns: list[OutputColumns] = Field(
        min_length=1,
        max_length=len(OutputColumns.__args__),
        examples=[OutputColumns.__args__],
    )
    data: list[
        conlist(
            item_type=confloat(allow_inf_nan=True),
            min_length=1,
            max_length=len(OutputColumns.__args__),
        )  # type: ignore
    ] = Field(examples=[[0.0] * len(OutputColumns.__args__)])
    summary: DynamicTariffAnalysisOutputSummary | None = None
