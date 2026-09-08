"""Models for contract-based energy cost calculation."""

import datetime as dt
from enum import StrEnum

import isodate
from energy_cost import Contract, ContractHistory, MeterType
from energy_cost.resolution import Resolution
from pydantic import BaseModel, ConfigDict, Field

from ..models import TimeDataFrame


class CostWarningCode(StrEnum):
    """Machine-readable reasons why a cost result may be unreliable."""

    CAPACITY_WINDOW_INCOMPLETE = "capacity_window_incomplete"
    PARTIAL_BILLING_PERIOD = "partial_billing_period"
    PERIOD_SHORTER_THAN_BILLING_PERIOD = "period_shorter_than_billing_period"
    INDEX_DATA_INCOMPLETE = "index_data_incomplete"
    ENERGY_DATA_GAPS_FILLED = "energy_data_gaps_filled"
    COST_INCOMPLETE = "cost_incomplete"
    NO_INJECTION_DATA = "no_injection_data"


class CostWarning(BaseModel):
    """A caveat attached to a cost result.

    Cost calculation is best-effort: a result can be arithmetically valid but still
    misleading (a capacity tariff averaged over an incomplete window, a price index
    that silently forward-filled past its last data point). Rather than failing, those
    conditions are reported here so the caller can decide.
    """

    code: CostWarningCode
    message: str
    context: dict = Field(default_factory=dict)


class ContractCostSettings(BaseModel):
    """Settings for contract-based cost calculation of a simulation.

    Note that ``output_resolution`` (an ISO 8601 duration, e.g. ``P1M``) is deliberately
    separate from ``FullSimulationInput.return_frequencies`` (pandas offset aliases, e.g.
    ``MS``). Energy may be reported at any frequency the caller finds useful, but cost
    periodicity is a contractual fact: fixed fees, the capacity tariff and its rolling
    window are only defined at billing granularity.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    contract: Contract | ContractHistory = Field(
        description="The tariff contract to cost the simulation against."
    )
    meter_type: MeterType = MeterType.SINGLE_RATE
    output_resolution: Resolution = Field(
        default_factory=lambda: isodate.Duration(months=1),
        examples=["P1M", "P1Y"],
        description="Billing resolution for the cost breakdown (ISO 8601 duration).",
    )
    include_periods: bool = Field(
        default=True,
        description="Include the per-period cost series, not just the period total.",
    )
    include_breakdown: bool = Field(
        default=True,
        description="Include the (category, cost_group, cost_type) breakdown.",
    )
    per_stage: bool = Field(
        default=False,
        description=(
            "Also cost the cumulative frame after each individual simulation stage, "
            "giving the marginal value of adding each asset. Costs one extra contract "
            "evaluation per stage."
        ),
    )
    start: dt.datetime | None = Field(
        default=None, description="Override the costing window start. Defaults to the data."
    )
    end: dt.datetime | None = Field(
        default=None, description="Override the costing window end. Defaults to the data."
    )


# breakdown[category][cost_group][cost_type] -> EUR, summed over the whole period
CostBreakdown = dict[str, dict[str, dict[str, float | None]]]


class CostResult(BaseModel):
    """Cost of one frame under one contract."""

    total: float | None = Field(description="Grand total in EUR over the whole period.")
    breakdown: CostBreakdown = Field(
        default_factory=dict,
        description="Cost per (tariff category, cost group, cost type), summed over the period.",
    )
    periods: TimeDataFrame | None = Field(
        default=None,
        description=(
            "Per-period cost series. Columns are flattened as "
            "'category.cost_group.cost_type'. Timestamps mark billing period starts in "
            "the contract's timezone, so read them back with "
            "`to_pandas(timezone=...)`; the default UTC would shift a monthly boundary "
            "into the previous month."
        ),
    )
    start: dt.datetime | None = None
    end: dt.datetime | None = None


class CostComparison(BaseModel):
    """Difference between two cost results.

    Mirrors the ``diff`` / ``ratio_diff`` idiom of ``openenergyid.simeval.compare_results``:
    ``diff = after - before``, so **savings are negative**. Keys are flattened breakdown
    paths (``"total"``, ``"distributor.capacity.total"``, ...).
    """

    diff: dict[str, float | None]
    ratio_diff: dict[str, float | None]


class CostSummary(BaseModel):
    """Full cost section of a simulation summary."""

    ex_ante: CostResult
    ex_post: CostResult
    stages: list[CostResult] | None = None
    comparison: CostComparison
    stage_comparisons: list[CostComparison] | None = None
    warnings: list[CostWarning] = Field(default_factory=list)
