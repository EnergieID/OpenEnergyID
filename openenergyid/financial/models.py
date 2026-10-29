"""Models for the financial projection of a simulation over its lifetime."""

from enum import StrEnum

from pydantic import BaseModel, Field


class FinancialWarningCode(StrEnum):
    """Machine-readable reasons why a financial projection may be unreliable."""

    SHORT_PERIOD_ANNUALISED = "short_period_annualised"
    SAVINGS_UNDEFINED = "savings_undefined"


class FinancialWarning(BaseModel):
    """A caveat attached to a financial projection."""

    code: FinancialWarningCode
    message: str
    context: dict = Field(default_factory=dict)


class FinancialSettings(BaseModel):
    """The investment in a simulated situation, to project its savings over the years.

    The investment covers the simulation as a whole (every stage together). The yearly
    savings come from contract costing, so this needs ``cost`` on the simulation input.
    How much each asset degrades is set on its own simulation parameters
    (``degradation_rate``); tariffs are kept at the simulated period's level every year.
    """

    investment: float = Field(
        ge=0,
        description="One-off investment (CAPEX) in EUR, paid at the start of year 1.",
        examples=[6000.0],
    )
    yearly_maintenance: float = Field(
        default=0.0,
        ge=0,
        description="Yearly running cost (OPEX) in EUR, subtracted from every year's savings.",
        examples=[0.0, 100.0],
    )
    discount_rate: float | None = Field(
        default=None,
        gt=-1,
        lt=1,
        description=(
            "Yearly discount rate as a fraction (0.03 is 3 %): the return the investment "
            "could have earned elsewhere. Tariffs are not escalated, so use a real rate "
            "(without inflation). Gives a discounted payback and NPV; without it only "
            "the simple payback and IRR are reported."
        ),
        examples=[0.03],
    )
    horizon_years: int = Field(
        default=25,
        ge=1,
        le=50,
        description="Number of years to project, typically the expected lifetime.",
        examples=[25, 10],
    )


class YearFinancials(BaseModel):
    """The projection of one year. Costs and savings are annualised, in EUR."""

    year: int = Field(description="Year since the investment, starting at 1.")
    simulated: bool = Field(
        default=True,
        description=(
            "Whether this year was simulated and costed. Otherwise its ex-post cost is "
            "interpolated between the simulated years around it."
        ),
    )
    cost_ex_ante: float = Field(description="Yearly cost without the simulated assets.")
    cost_ex_post: float = Field(description="Yearly cost with the assets, degraded to this year.")
    savings: float = Field(description="`cost_ex_ante - cost_ex_post`.")
    cash_flow: float = Field(description="`savings - yearly_maintenance`.")
    cumulative: float = Field(description="Sum of the cash flows up to and including this year.")
    discounted_cash_flow: float | None = Field(
        default=None, description="`cash_flow / (1 + discount_rate) ** year`."
    )
    discounted_cumulative: float | None = Field(
        default=None, description="Sum of the discounted cash flows up to this year."
    )


class FinancialSummary(BaseModel):
    """Payback, NPV and IRR of a simulation, from its savings projected year by year."""

    settings: FinancialSettings
    yearly: list[YearFinancials] = Field(default_factory=list)
    payback_years: float | None = Field(
        default=None,
        description=(
            "Years until the cumulative cash flow equals the investment, interpolated "
            "within the year. None when that does not happen within the horizon."
        ),
    )
    discounted_payback_years: float | None = Field(
        default=None,
        description="The same with discounted cash flows. Only with a discount rate.",
    )
    npv: float | None = Field(
        default=None,
        description=(
            "Net present value over the horizon: the discounted cash flows minus the "
            "investment. Only with a discount rate."
        ),
    )
    irr: float | None = Field(
        default=None,
        description=(
            "Internal rate of return over the horizon, as a fraction: the discount rate at "
            "which the NPV is zero. None when there is no such rate."
        ),
    )
    warnings: list[FinancialWarning] = Field(default_factory=list)
