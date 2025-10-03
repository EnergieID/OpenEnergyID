"""Models for basic energy system evaluation."""

from typing import Literal

from pydantic import Field, confloat, conlist

from ..models import TimeDataFrame


class EvaluationInput(TimeDataFrame):
    """Input frame for basic energy system evaluation."""

    columns: list[
        Literal[
            "electricity_delivered",
            "electricity_exported",
            "electricity_produced",
            "price_electricity_delivered",
            "price_electricity_exported",
        ]
    ] = Field(min_length=1, max_length=5)
    data: list[conlist(item_type=confloat(allow_inf_nan=True), min_length=1, max_length=5)]  # type: ignore
