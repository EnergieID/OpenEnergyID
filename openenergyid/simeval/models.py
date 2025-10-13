"""Models for basic energy system evaluation."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, RootModel, confloat, conlist

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


# ---------- Reusable bits ----------

FrequencyKey = Annotated[
    str,
    Field(
        description="Pandas-style frequency string used in the request.",
        examples=["15min", "H", "MS", "W-MON"],
    ),
]

MetricSummary = Annotated[
    dict[str, float],
    Field(
        description=(
            "Total/aggregate values per metric (e.g. { 'electricity_delivered': 123.4 }). "
            "Keys are metric identifiers from the evaluation domain."
        ),
        examples=[{"electricity_delivered": 123.4, "cost_electricity_net": 42.0}],
    ),
]

# ---------- Eval payloads ----------
# We model “either { 'total': MetricSummary } OR { '<freq>': EvaluationOutput, ... }”
# as two schemas and present them via oneOf.


class EvalTotals(BaseModel):
    """Totals per metric (no time series)."""

    total: MetricSummary = Field(
        ..., description="Aggregate totals per metric over the full evaluation window."
    )


class EvalByFrequency(RootModel[dict[FrequencyKey, EvaluationOutput]]):
    """Per-frequency time series results."""

    root: dict[FrequencyKey, EvaluationOutput]


EvalPayload = Annotated[
    Union[EvalTotals, EvalByFrequency],
    Field(description="Either totals-only, or per-frequency time series results."),
]

# ---------- Comparison payloads ----------
# Shape:
#   EITHER  { "total": { "diff": MetricSummary } } or { "total": { "ratio_diff": MetricSummary } }
#   OR      { "<freq>": { "diff": EvaluationOutput } } or { "<freq>": { "ratio_diff": EvaluationOutput } }


class DiffTotal(BaseModel):
    """Totals-level absolute differences."""

    diff: MetricSummary = Field(..., description="Absolute differences of totals.")


class RatioDiffTotal(BaseModel):
    """Totals-level relative (ratio) differences."""

    ratio_diff: MetricSummary = Field(..., description="Relative (ratio) differences of totals.")


class DiffTS(BaseModel):
    """Time series absolute differences at a given frequency."""

    diff: EvaluationOutput = Field(
        ..., description="Absolute differences as a time series at this frequency."
    )


class RatioDiffTS(BaseModel):
    """Time series relative (ratio) differences at a given frequency."""

    ratio_diff: EvaluationOutput = Field(
        ..., description="Relative (ratio) differences as a time series at this frequency."
    )


# Enforce exclusivity if you decide to combine diff/ratio into one model; here we keep them separate for clean docs.


class ComparisonTotals(BaseModel):
    """Totals comparison; pick exactly one of diff or ratio_diff by choosing the model."""

    total: DiffTotal | RatioDiffTotal = Field(..., description="Totals-level comparison.")


class ComparisonByFrequency(RootModel[dict[FrequencyKey, Union[DiffTS, RatioDiffTS]]]):
    """Per-frequency comparison; for each freq, pick diff or ratio_diff."""

    root: dict[FrequencyKey, DiffTS | RatioDiffTS]


ComparisonPayload = Annotated[
    Union[ComparisonTotals, ComparisonByFrequency],
    Field(description="Comparison results at totals level or per-frequency."),
]
