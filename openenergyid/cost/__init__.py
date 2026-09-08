"""Contract-based energy cost calculation.

Where ``openenergyid.simeval`` monetises a simulation with per-interval prices supplied
by the caller, this module costs it against a full tariff contract (``energy_cost``):
supplier, distributor, fees and taxes, including the capacity tariff and fixed charges.
The two mechanisms are mutually exclusive.
"""

from .diagnostics import CostCalculationError, UnknownIndexError, iter_index_names
from .main import compare_costs, compute_cost, cost_simulation
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
from .resolution import pandas_freq_to_resolution

__all__ = [
    "ContractCostSettings",
    "CostBreakdown",
    "CostCalculationError",
    "CostComparison",
    "CostResult",
    "CostSummary",
    "CostWarning",
    "CostWarningCode",
    "UnknownIndexError",
    "compare_costs",
    "compute_cost",
    "cost_simulation",
    "detect_frame_resolution",
    "frame_to_meters",
    "iter_index_names",
    "pandas_freq_to_resolution",
]
