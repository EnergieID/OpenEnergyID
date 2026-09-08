"""Contract-based cost calculation for simulation results."""

import logging

import pandas as pd

from ..models import TimeDataFrame
from .diagnostics import CostCalculationError, check_indexes, check_period
from .meters import detect_frame_resolution, frame_to_meters
from .models import (
    ContractCostSettings,
    CostBreakdown,
    CostComparison,
    CostResult,
    CostSummary,
    CostWarning,
    CostWarningCode,
)

logger = logging.getLogger(__name__)

TOTAL_COLUMN = ("total", "total", "total")
TIMESTAMP_COLUMN = "timestamp"


def _column_path(column) -> tuple[str, ...]:
    """Normalise a result column into a tuple of non-empty string parts."""
    parts = column if isinstance(column, tuple) else (column,)
    return tuple(str(part) for part in parts if part != "")


def _build_breakdown(frame: pd.DataFrame) -> CostBreakdown:
    """Sum each cost column over the period and nest it by category/group/type."""
    breakdown: CostBreakdown = {}
    for column in frame.columns:
        parts = _column_path(column)
        if parts == (TIMESTAMP_COLUMN,) or parts == TOTAL_COLUMN:
            continue
        if len(parts) != 3:
            continue
        category, cost_group, cost_type = parts
        total = frame[column].sum(skipna=False)
        breakdown.setdefault(category, {}).setdefault(cost_group, {})[cost_type] = (
            None if pd.isna(total) else float(total)
        )
    return breakdown


def _build_periods(frame: pd.DataFrame) -> TimeDataFrame:
    """Flatten the MultiIndex columns into a TimeDataFrame keyed by timestamp."""
    flat = frame.copy()
    timestamps = flat[[c for c in flat.columns if _column_path(c) == (TIMESTAMP_COLUMN,)][0]]
    values = flat[[c for c in flat.columns if _column_path(c) != (TIMESTAMP_COLUMN,)]]
    values.columns = [".".join(_column_path(c)) for c in values.columns]
    values.index = pd.DatetimeIndex(timestamps)
    return TimeDataFrame.from_pandas(values)


def _flatten_totals(result: CostResult) -> dict[str, float | None]:
    """Flatten a cost result into comparable 'category.group.type' keys."""
    flat: dict[str, float | None] = {"total": result.total}
    for category, groups in result.breakdown.items():
        for cost_group, cost_types in groups.items():
            for cost_type, value in cost_types.items():
                flat[f"{category}.{cost_group}.{cost_type}"] = value
    return flat


def compute_cost(
    data: pd.DataFrame,
    settings: ContractCostSettings,
    *,
    preflight: bool = True,
) -> tuple[CostResult, list[CostWarning]]:
    """Cost one simulation frame under one contract."""
    resolution = detect_frame_resolution(data.index)
    consumption, injection, warnings = frame_to_meters(
        data, meter_type=settings.meter_type, resolution=resolution
    )

    start = settings.start or data.index[0].to_pydatetime()
    # energy_cost treats the window as half-open, so the last interval needs its width.
    end = settings.end or (data.index[-1].to_pydatetime() + resolution)

    if preflight:
        warnings.extend(check_indexes(settings.contract, end))
        warnings.extend(check_period(settings.contract, start, end, settings.output_resolution))

    try:
        frame = settings.contract.apply(
            consumption=consumption,
            injection=injection,
            start=start,
            end=end,
            output_resolution=settings.output_resolution,
        )
    except KeyError as exc:  # an index name that slipped past the pre-flight
        raise CostCalculationError(
            f"Cost calculation referenced an unregistered price index: {exc}"
        ) from exc

    if frame is None or frame.empty:
        raise CostCalculationError(
            "No cost could be calculated for the requested period. Check that the "
            "contract's versions cover it."
        )

    total_columns = [c for c in frame.columns if _column_path(c) == TOTAL_COLUMN]
    if not total_columns:
        raise CostCalculationError("Cost result carries no grand total column.")
    total_series = frame[total_columns[0]]
    total = total_series.sum(skipna=False)

    if pd.isna(total):
        warnings.append(
            CostWarning(
                code=CostWarningCode.COST_INCOMPLETE,
                message=(
                    "The cost total is undefined for at least one billing period, most "
                    "likely because a price index has no data covering it."
                ),
                context={"periods": int(total_series.isna().sum())},
            )
        )

    result = CostResult(
        total=None if pd.isna(total) else float(total),
        breakdown=_build_breakdown(frame) if settings.include_breakdown else {},
        periods=_build_periods(frame) if settings.include_periods else None,
        start=start,
        end=end,
    )
    return result, warnings


def compare_costs(before: CostResult, after: CostResult) -> CostComparison:
    """Compare two cost results.

    ``diff = after - before``, so savings are negative -- the same sign convention as
    ``openenergyid.simeval.compare_results``. Unlike ``simeval.compare_results``,
    ``ratio_diff`` is deliberately ``None`` wherever the baseline is zero or missing,
    never ``inf`` or ``nan``, because the result has to survive JSON serialization.
    """
    flat_before = _flatten_totals(before)
    flat_after = _flatten_totals(after)

    diff: dict[str, float | None] = {}
    ratio_diff: dict[str, float | None] = {}

    for key in sorted(set(flat_before) | set(flat_after)):
        base = flat_before.get(key)
        new = flat_after.get(key)
        if base is None and new is None:
            diff[key] = None
            ratio_diff[key] = None
            continue
        base_value = 0.0 if base is None else base
        new_value = 0.0 if new is None else new
        delta = new_value - base_value
        diff[key] = delta
        ratio_diff[key] = None if not base_value else delta / base_value

    return CostComparison(diff=diff, ratio_diff=ratio_diff)


def _dedupe(warnings: list[CostWarning]) -> list[CostWarning]:
    """Drop warnings that repeat the same code and context."""
    seen: set[tuple] = set()
    unique: list[CostWarning] = []
    for warning in warnings:
        key = (warning.code, tuple(sorted(warning.context.items(), key=lambda kv: kv[0])))
        if key in seen:
            continue
        seen.add(key)
        unique.append(warning)
    return unique


def cost_simulation(frames, settings: ContractCostSettings) -> CostSummary:
    """Cost a simulation's ex-ante frame, ex-post frame and optionally each stage.

    ``frames`` is a ``openenergyid.sim.SimulationFrames``. Stage costs use the
    *cumulative* frames, so each one answers "what would the bill be with assets 1..k
    installed"; the per-simulator result frames are not billable on their own.

    The pre-flight runs once: the contract and window are identical for every frame.
    """
    warnings: list[CostWarning] = []

    ex_ante, ex_ante_warnings = compute_cost(frames.ex_ante, settings, preflight=True)
    warnings.extend(ex_ante_warnings)

    stages: list[CostResult] | None = None
    stage_comparisons: list[CostComparison] | None = None
    if settings.per_stage:
        stages = []
        stage_comparisons = []
        previous = ex_ante
        for stage_frame in frames.stages:
            stage_result, stage_warnings = compute_cost(stage_frame, settings, preflight=False)
            warnings.extend(stage_warnings)
            stages.append(stage_result)
            stage_comparisons.append(compare_costs(previous, stage_result))
            previous = stage_result
        ex_post = stages[-1] if stages else ex_ante
    else:
        ex_post, ex_post_warnings = compute_cost(frames.ex_post, settings, preflight=False)
        warnings.extend(ex_post_warnings)

    return CostSummary(
        ex_ante=ex_ante,
        ex_post=ex_post,
        stages=stages,
        comparison=compare_costs(ex_ante, ex_post),
        stage_comparisons=stage_comparisons,
        warnings=_dedupe(warnings),
    )
