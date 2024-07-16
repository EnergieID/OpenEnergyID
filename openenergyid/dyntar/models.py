"""Models for dynamic tariff analysis."""

from typing import Literal
from pydantic import Field, conlist

from openenergyid.models import TimeDataFrame


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
]


class DynamicTariffAnalysisInput(TimeDataFrame):
    """Input frame for dynamic tariff analysis."""

    columns: list[RequiredColumns] = Field(
        min_length=len(RequiredColumns.__args__),
        max_length=len(RequiredColumns.__args__),
        examples=[RequiredColumns.__args__],
    )
    data: list[
        conlist(
            item_type=float,
            min_length=len(RequiredColumns.__args__),
            max_length=len(RequiredColumns.__args__),
        )  # type: ignore
    ] = Field(examples=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


class DynamicTariffAnalysisOutput(TimeDataFrame):
    """Output frame for dynamic tariff analysis."""

    columns: list[OutputColumns] = Field(
        min_length=1,
        max_length=len(OutputColumns.__args__),
        examples=[OutputColumns.__args__],
    )
    data: list[
        conlist(item_type=float, min_length=1, max_length=len(OutputColumns.__args__))  # type: ignore
    ] = Field(examples=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
