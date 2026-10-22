"""Models for contract-based energy cost calculation."""

import datetime as dt
from enum import StrEnum
from typing import Literal

import isodate
from energy_cost import Contract, ContractHistory, MeterType
from energy_cost.resolution import Resolution
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

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


class MonthPeak(BaseModel):
    """The highest quarter-hour of one calendar month."""

    timestamp: AwareDatetime = Field(
        description=(
            "Start of the quarter-hour in which the month's peak occurred, with the "
            "local UTC offset."
        ),
        examples=["2023-02-17T18:15:00+01:00"],
    )
    value: float = Field(
        ge=0,
        description="Average offtake power over that quarter-hour, in W.",
        examples=[4210.0],
    )


class MonthPeaks(BaseModel):
    """Measured monthly peaks of a grid connection, used as capacity history."""

    resolution: Literal["P1M"] = Field(
        default="P1M", description="Always `P1M`: one peak per calendar month."
    )
    values: list[MonthPeak] = Field(
        description=(
            "One peak per calendar month, in W. Provide the costed period plus the 11 "
            "months before it (the capacity tariff averages 12 monthly peaks). Only the "
            "months before the first billed month are used; peaks inside the period are "
            "recalculated from the frame being costed."
        )
    )

    @model_validator(mode="after")
    def _one_peak_per_month(self) -> "MonthPeaks":
        # The month is the one in the peak's own local offset, which is how it was stored.
        months = [(v.timestamp.year, v.timestamp.month) for v in self.values]
        duplicated = sorted({f"{y}-{m:02d}" for y, m in months if months.count((y, m)) > 1})
        if duplicated:
            raise ValueError(
                "`month_peaks` must hold one value per calendar month; more than one "
                f"for {', '.join(duplicated)}."
            )
        return self


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
    month_peaks: MonthPeaks | None = Field(
        default=None,
        description=(
            "Measured month peaks of the connection (W, one per calendar month, "
            "resolution `P1M`) used as capacity history. The capacity tariff averages "
            "the last 12 monthly peaks, so a costed period needs the 11 months before "
            "it. Supply the costed period plus those 11 months: the months before the "
            "first billed month then fill the window, for every frame costed (ex-ante, "
            "ex-post and each stage alike), while the months inside the period are "
            "always recalculated from the frame being costed, so provided in-period "
            "peaks are ignored. Optional: without it, or when any of those 11 months is "
            "missing, the capacity is derived from the frame alone and "
            "`capacity_window_incomplete` is reported as before."
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
