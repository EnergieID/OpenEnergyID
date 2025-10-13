"""Models for basic energy system evaluation."""

from typing import Annotated, Literal, Union

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


class EvaluationOutput(TimeDataFrame):
    """Output frame for basic energy system evaluation."""

    columns: list[
        Literal[
            "electricity_delivered",
            "electricity_exported",
            "electricity_produced",
            "electricity_consumed",
            "electricity_self_consumed",
            "cost_electricity_delivered",
            "earnings_electricity_exported",
            "cost_electricity_net",
            "ratio_self_consumption",
            "ratio_self_sufficiency",
        ]
    ] = Field(min_length=1, max_length=10)
    data: list[conlist(item_type=confloat(allow_inf_nan=True), min_length=1, max_length=10)]  # type: ignore


FrequencyStr = Annotated[
    str,
    Field(
        description="Pandas freqstr. that was used in the request.",
        examples=["MS", "W-MON"],
    ),
]

Metric = Annotated[
    str,
    Field(
        description="The metric that was calculated.",
        examples=[
            "electricity_delivered",
            "electricity_exported",
            "electricity_produced",
            "electricity_consumed",
            "electricity_self_consumed",
            "cost_electricity_delivered",
            "earnings_electricity_exported",
            "cost_electricity_net",
            "ratio_self_consumption",
            "ratio_self_sufficiency",
        ],
    ),
]

MetricSummary = Annotated[dict[Metric, float], Field(description="Total values for each metric.")]

EvalPayload = Union[dict[Literal["total"], MetricSummary], dict[FrequencyStr, EvaluationOutput]]

CompBase = Annotated[
    Union[Literal["diff"], Literal["ratio_diff"]],
    Field(
        description="The type of comparison: absolute difference (`diff`) or relative difference (`ratio_diff`).",
        examples=["diff", "ratio_diff"],
    ),
]

ComparisonPayload = Annotated[
    Union[
        dict[Literal["total"], dict[CompBase, MetricSummary]],
        dict[FrequencyStr, dict[CompBase, EvaluationOutput]],
    ],
    Field(description="Comparison results."),
]
