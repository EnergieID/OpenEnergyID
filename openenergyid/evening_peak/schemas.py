"""Pandera schemas for the evening peak avoidance analysis.

Only the value columns are declared. The ``day`` / ``week`` / ``timestamp`` columns are
deliberately left out: pandera's Polars ``DateTime`` type cannot express "a datetime in
any time zone", so declaring it would either reject the timezone-aware frames the
analysis produces or, with coercion enabled, silently strip the time zone. Correct
local-time behaviour is asserted by the tests instead, which is where it belongs.

Note that pandera only runs value checks on an eager ``DataFrame``; validating a
``LazyFrame`` checks columns and dtypes alone. The analysis therefore validates its
small per-day and per-week results eagerly, and the large quarter-hourly frame lazily.
"""

import pandera.polars as pa


class NetOfftakeSchema(pa.DataFrameModel):
    """Validates the net offtake series produced by ``prepare_net_offtake``.

    Net offtake is ``offtake - injection`` clipped at zero per quarter-hour, so it is
    non-negative by construction. Values are energy per quarter-hour, in kWh.
    """

    net_offtake_in_kilowatthour: float = pa.Field(ge=0)

    class Config:
        """Allow the undeclared timestamp column through untouched."""

        strict = False
        coerce = True


class DailyEveningPeakSchema(pa.DataFrameModel):
    """Validates the per-day results of the evening peak analysis.

    The two headline metrics are nullable: a day whose evening window is not fully
    covered by measurements has no meaningful peak, and a day without offtake has no
    meaningful share. The ``le=100`` bound on the share is the check that would catch a
    regression in the injection clipping.
    """

    evening_peak_in_kilowatt: float | None = pa.Field(ge=0, nullable=True)
    evening_peak_share_in_percent: float | None = pa.Field(ge=0, le=100, nullable=True)
    daily_offtake_in_kilowatthour: float = pa.Field(ge=0)
    evening_offtake_in_kilowatthour: float = pa.Field(ge=0)
    observed_quarters: int = pa.Field(ge=0)
    expected_quarters: int = pa.Field(ge=0)
    observed_window_quarters: int = pa.Field(ge=0)
    has_full_window: bool = pa.Field()
    is_complete: bool = pa.Field()
    is_below_threshold: bool | None = pa.Field(nullable=True)

    class Config:
        """Allow the undeclared day column through untouched."""

        strict = False
        coerce = True


class WeekMedianSchema(pa.DataFrameModel):
    """Validates the weekly median reference lines drawn over the daily series."""

    median_evening_peak_in_kilowatt: float | None = pa.Field(ge=0, nullable=True)
    median_evening_peak_share_in_percent: float | None = pa.Field(ge=0, le=100, nullable=True)

    class Config:
        """Allow the undeclared week column through untouched."""

        strict = False
        coerce = True
