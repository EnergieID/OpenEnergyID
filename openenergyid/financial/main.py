"""Payback, NPV and IRR from a series of yearly cash flows."""

import datetime as dt
from collections.abc import Sequence

from .models import (
    FinancialSettings,
    FinancialSummary,
    FinancialWarning,
    FinancialWarningCode,
    YearFinancials,
)

DAYS_PER_YEAR = 365.25

# The IRR is searched between these rates. Below -99 % the discount factors explode, and
# an investment earning more than 1000 % a year is not a realistic answer.
IRR_LOWER = -0.99
IRR_UPPER = 10.0
IRR_TOLERANCE = 1e-10


def annualisation_factor(start: dt.datetime, end: dt.datetime) -> float:
    """The factor that turns a total over ``[start, end)`` into a yearly figure."""
    days = (end - start).total_seconds() / 86400
    if days <= 0:
        raise ValueError(f"Cannot annualise an empty period: {start} to {end}.")
    return DAYS_PER_YEAR / days


def check_period(start: dt.datetime, end: dt.datetime) -> list[FinancialWarning]:
    """Flag a period shorter than a year: annualising it ignores the seasons it misses."""
    days = (end - start).total_seconds() / 86400
    if days >= 365:
        return []
    return [
        FinancialWarning(
            code=FinancialWarningCode.SHORT_PERIOD_ANNUALISED,
            message=(
                f"The yearly savings are extrapolated from {days:.0f} days. A period "
                "shorter than a year leaves out seasons with other consumption and solar "
                "production, so the yearly figures and the payback are less reliable."
            ),
            context={"start": start.isoformat(), "end": end.isoformat(), "days": days},
        )
    ]


def discount(cash_flows: Sequence[float], rate: float) -> list[float]:
    """Discount the cash flow of year n (starting at 1) by ``(1 + rate) ** n``."""
    return [cash_flow / (1 + rate) ** year for year, cash_flow in enumerate(cash_flows, start=1)]


def payback_years(investment: float, cash_flows: Sequence[float]) -> float | None:
    """Years until the cumulative cash flow reaches the investment.

    Within the year it is reached, the cash flow is assumed to come in evenly, so the
    result is interpolated. None when the horizon ends first.
    """
    if investment <= 0:
        return 0.0
    cumulative = 0.0
    for year, cash_flow in enumerate(cash_flows, start=1):
        if cash_flow > 0 and cumulative + cash_flow >= investment:
            return year - 1 + (investment - cumulative) / cash_flow
        cumulative += cash_flow
    return None


def npv(investment: float, cash_flows: Sequence[float], rate: float) -> float:
    """Net present value: the discounted cash flows minus the investment at year 0."""
    return sum(discount(cash_flows, rate)) - investment


def irr(investment: float, cash_flows: Sequence[float]) -> float | None:
    """The rate at which the NPV is zero, found by bisection.

    None when the NPV does not change sign between ``IRR_LOWER`` and ``IRR_UPPER``: no
    investment, or cash flows that never pay it back. With cash flows that change sign
    more than once there can be several such rates; this returns one of them.
    """
    if investment <= 0:
        return None
    low, high = IRR_LOWER, IRR_UPPER
    npv_low = npv(investment, cash_flows, low)
    npv_high = npv(investment, cash_flows, high)
    if npv_low == 0:
        return low
    if npv_low * npv_high > 0:
        return None
    while high - low > IRR_TOLERANCE:
        middle = (low + high) / 2
        npv_middle = npv(investment, cash_flows, middle)
        if npv_middle == 0:
            return middle
        if (npv_middle > 0) == (npv_low > 0):
            low, npv_low = middle, npv_middle
        else:
            high = middle
    return (low + high) / 2


def summarise(
    settings: FinancialSettings,
    costs_ex_ante: Sequence[float],
    costs_ex_post: Sequence[float],
    warnings: list[FinancialWarning] | None = None,
) -> FinancialSummary:
    """Project the yearly costs into cash flows, payback, NPV and IRR.

    ``costs_ex_ante`` and ``costs_ex_post`` hold one annualised cost per year of the
    horizon, year 1 first.
    """
    savings = [before - after for before, after in zip(costs_ex_ante, costs_ex_post)]
    cash_flows = [saving - settings.yearly_maintenance for saving in savings]
    rate = settings.discount_rate
    discounted = discount(cash_flows, rate) if rate is not None else None

    yearly: list[YearFinancials] = []
    cumulative = 0.0
    discounted_cumulative = 0.0
    for index, cash_flow in enumerate(cash_flows):
        cumulative += cash_flow
        discounted_cash_flow = None
        if discounted is not None:
            discounted_cash_flow = discounted[index]
            discounted_cumulative += discounted_cash_flow
        yearly.append(
            YearFinancials(
                year=index + 1,
                cost_ex_ante=costs_ex_ante[index],
                cost_ex_post=costs_ex_post[index],
                savings=savings[index],
                cash_flow=cash_flow,
                cumulative=cumulative,
                discounted_cash_flow=discounted_cash_flow,
                discounted_cumulative=discounted_cumulative if discounted is not None else None,
            )
        )

    return FinancialSummary(
        settings=settings,
        yearly=yearly,
        payback_years=payback_years(settings.investment, cash_flows),
        discounted_payback_years=(
            payback_years(settings.investment, discounted) if discounted is not None else None
        ),
        npv=npv(settings.investment, cash_flows, rate) if rate is not None else None,
        irr=irr(settings.investment, cash_flows),
        warnings=warnings or [],
    )
