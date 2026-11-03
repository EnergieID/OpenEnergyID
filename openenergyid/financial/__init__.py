"""Financial projection of a simulation: payback time, NPV and IRR.

The yearly savings come from contract costing (``openenergyid.cost``) of the simulation,
re-simulated for every year with the assets degraded (``openenergyid.sim.years``). This
module holds the models and the arithmetic on the resulting cash flows.
"""

from .main import (
    annualisation_factor,
    check_period,
    discount,
    irr,
    npv,
    payback_years,
    summarise,
)
from .models import (
    SIMULATED_YEARS,
    FinancialSettings,
    FinancialSummary,
    FinancialWarning,
    FinancialWarningCode,
    YearFinancials,
)

__all__ = [
    "SIMULATED_YEARS",
    "FinancialSettings",
    "FinancialSummary",
    "FinancialWarning",
    "FinancialWarningCode",
    "YearFinancials",
    "annualisation_factor",
    "check_period",
    "discount",
    "irr",
    "npv",
    "payback_years",
    "summarise",
]
