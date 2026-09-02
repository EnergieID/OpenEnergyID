"""Wire models for the evening peak avoidance analysis.

These are the request and response bodies of the Data Analytics Engine endpoint, and
their field descriptions are what the front-end developers read in the generated
OpenAPI documentation. Keep them descriptive.
"""

import datetime as dt
from typing import Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openenergyid.models import TimeSeries

from .analysis import DAY, WEEK, EveningPeakAnalysisResult, PeakMoment, summarize


class EveningPeakInput(BaseModel):
    """Input for the evening peak avoidance analysis of a single connection.

    Both series are **gross meter registers in kWh per quarter-hour**, as delivered by
    the smart meter's P4 port. Net offtake is derived as
    ``max(grossOfftake - grossInjection, 0)`` per quarter-hour, before any summation, so
    that the resulting peak share stays between 0 and 100% and is comparable between
    households with and without solar panels.

    Timestamps label the **start** of each quarter-hour. The evening window is therefore
    half-open: with the defaults, it covers the twenty quarter-hours 16:00 through 20:45.
    """

    gross_offtake: TimeSeries = Field(
        alias="grossOfftake",
        description=(
            "Gross offtake from the grid, in kWh per quarter-hour, non-negative. "
            "Timestamps label the start of each interval and must carry a UTC offset."
        ),
    )
    gross_injection: TimeSeries | None = Field(
        default=None,
        alias="grossInjection",
        description=(
            "Gross injection into the grid, in kWh per quarter-hour, non-negative. "
            "Omit for a connection without production; net offtake then equals gross "
            "offtake. Timestamps need not cover exactly the same range as grossOfftake: "
            "missing quarter-hours are treated as zero injection."
        ),
    )
    timezone: str = Field(
        default="Europe/Amsterdam",
        alias="timeZone",
        description=(
            "IANA timezone of the connection, e.g. 'Europe/Amsterdam'. Day boundaries "
            "and the evening window are evaluated against this local wall clock, so an "
            "incorrect value shifts the window onto the wrong hours."
        ),
    )
    window_start: dt.time = Field(
        default=dt.time(16, 0),
        alias="windowStart",
        description="Local start of the evening window, inclusive. Default 16:00.",
    )
    window_end: dt.time = Field(
        default=dt.time(21, 0),
        alias="windowEnd",
        description="Local end of the evening window, exclusive. Default 21:00.",
    )
    peak_share_threshold: float = Field(
        default=0.37,
        ge=0,
        le=1,
        alias="peakShareThreshold",
        description=(
            "Peak share below which a day counts as a good day, as a fraction. Default "
            "0.37, the peak share of an average Dutch household. One fixed value for "
            "all participants, configured centrally rather than per workspace."
        ),
    )
    num_peak_moments: int = Field(
        default=10,
        ge=0,
        le=50,
        alias="numPeakMoments",
        description="How many of the highest evening peaks to return, highest first.",
    )
    min_day_coverage: float = Field(
        default=0.9,
        ge=0,
        le=1,
        alias="minDayCoverage",
        description=(
            "Minimum fraction of a day's quarter-hours that must be present before that "
            "day's share is reported. Guards against the partial first and last day of "
            "an export producing a share against an incomplete denominator."
        ),
    )
    reference: str | None = Field(
        default=None,
        description="Optional caller-supplied identifier, echoed back in the response.",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "timeZone": "Europe/Amsterdam",
                "grossOfftake": {
                    "index": [
                        "2026-11-01T16:00:00+01:00",
                        "2026-11-01T16:15:00+01:00",
                        "2026-11-01T16:30:00+01:00",
                    ],
                    "data": [0.412, 0.688, 1.104],
                },
                "grossInjection": {
                    "index": [
                        "2026-11-01T16:00:00+01:00",
                        "2026-11-01T16:15:00+01:00",
                        "2026-11-01T16:30:00+01:00",
                    ],
                    "data": [0.0, 0.0, 0.0],
                },
                "windowStart": "16:00",
                "windowEnd": "21:00",
                "peakShareThreshold": 0.37,
                "numPeakMoments": 10,
                "reference": "EA-14214640",
            }
        },
    )

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """The window must be a non-empty, forward interval."""
        if self.window_start >= self.window_end:
            raise ValueError(
                f"windowStart ({self.window_start}) must be strictly before "
                f"windowEnd ({self.window_end})."
            )
        return self

    @model_validator(mode="after")
    def validate_timezone(self) -> Self:
        """The timezone must be a real IANA zone.

        Checked here so an unknown zone is a rejected request with a clear message,
        rather than an error raised deep in the dataframe layer once the analysis is
        already running.
        """
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"timeZone {self.timezone!r} is not a known IANA time zone.") from exc
        return self

    @model_validator(mode="after")
    def validate_series_lengths(self) -> Self:
        """Each series must carry one value per timestamp.

        ``TimeSeries`` itself does not enforce this, and a mismatch would otherwise
        surface much later as a shape error from the dataframe layer rather than as a
        rejected request.
        """
        for alias, series in (
            ("grossOfftake", self.gross_offtake),
            ("grossInjection", self.gross_injection),
        ):
            if series is None:
                continue
            if len(series.index) != len(series.data):
                raise ValueError(
                    f"{alias} has {len(series.index)} timestamps but "
                    f"{len(series.data)} values; they must match."
                )
        return self

    def to_polars(self) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
        """Convert the two gross series to Polars frames in the analysis timezone."""
        offtake = self.gross_offtake.to_polars(timezone=self.timezone)
        injection = (
            self.gross_injection.to_polars(timezone=self.timezone)
            if self.gross_injection is not None
            else None
        )
        return offtake, injection


class EveningPeakSummary(BaseModel):
    """The key figures shown under the Avondpieken and Piekaandeel cards.

    All statistics are computed over measured days only. ``daysBelowThreshold`` and
    ``measuredDays`` are the numerator and denominator of the "49 of 120 days" figure.
    """

    average_peak_in_kilowatt: float | None = Field(
        default=None, alias="averagePeak", description="Mean evening peak, in kW."
    )
    lowest_peak_in_kilowatt: float | None = Field(
        default=None, alias="lowestPeak", description="Lowest evening peak, in kW."
    )
    highest_peak_in_kilowatt: float | None = Field(
        default=None, alias="highestPeak", description="Highest evening peak, in kW."
    )
    average_share_in_percent: float | None = Field(
        default=None, alias="averageShare", description="Mean peak share, in percent."
    )
    lowest_share_in_percent: float | None = Field(
        default=None, alias="lowestShare", description="Lowest peak share, in percent."
    )
    highest_share_in_percent: float | None = Field(
        default=None, alias="highestShare", description="Highest peak share, in percent."
    )
    days_below_threshold: int = Field(
        alias="daysBelowThreshold",
        description="Measured days whose peak share was strictly below the threshold.",
    )
    measured_days: int = Field(
        alias="measuredDays",
        description=(
            "Days with enough measurements to report a peak share. Days that are "
            "incompletely measured are excluded from this count."
        ),
    )
    threshold_in_percent: float = Field(
        alias="thresholdInPercent",
        description="The threshold the day count was made against, in percent.",
    )
    first_day: dt.date | None = Field(
        default=None,
        alias="firstDay",
        description="First local day present in the data.",
    )
    last_day: dt.date | None = Field(
        default=None,
        alias="lastDay",
        description="Last local day present in the data.",
    )

    model_config = ConfigDict(populate_by_name=True)


class PeakMomentOutput(BaseModel):
    """One of the highest evening peaks, with the curve of the day it happened on."""

    peak_time: dt.datetime = Field(
        alias="peakTime",
        description="Local timestamp of the quarter-hour in which the peak occurred.",
    )
    peak_value_in_kilowatt: float = Field(
        alias="peakValue", description="Power during that quarter-hour, in kW."
    )
    day_curve: TimeSeries = Field(
        alias="dayCurve",
        description=(
            "The whole local day at quarter-hour resolution, in kW, for the sparkline. "
            "The full day is returned rather than only the evening, so the peak can be "
            "shown in the context of the rest of the day."
        ),
    )

    model_config = ConfigDict(populate_by_name=True)


class EveningPeakOutput(BaseModel):
    """Result of the evening peak avoidance analysis.

    Covers three of the four cards on the analysis page. The Trends heatmap is not
    included: the caller already holds the quarter-hourly data it sent, and building the
    heatmap from it front-end avoids returning the same values twice.
    """

    daily_peak: TimeSeries = Field(
        alias="dailyPeak",
        description=(
            "Highest evening peak per day, in kW, indexed by the start of the local "
            "day. The index is a gapless run of days covering the measured range, so it "
            "can be plotted directly; the value is null on days whose evening window "
            "was not fully measured and on days with no measurements at all, which "
            "breaks the line where the data breaks."
        ),
    )
    daily_share: TimeSeries = Field(
        alias="dailyShare",
        description=(
            "Peak share per day, in percent, on the same gapless daily index as "
            "dailyPeak. Null on incompletely measured days, on days with no "
            "measurements, and on days without any offtake."
        ),
    )
    week_median_peak: TimeSeries = Field(
        alias="weekMedianPeak",
        description=(
            "Median evening peak per ISO week, in kW, indexed by the Monday of each "
            "week. This is the calmer reference line drawn over the daily series; "
            "render it as a step line."
        ),
    )
    week_median_share: TimeSeries = Field(
        alias="weekMedianShare",
        description=(
            "Median peak share per ISO week, in percent, indexed by the Monday of each "
            "week. Render as a step line."
        ),
    )
    peak_moments: list[PeakMomentOutput] = Field(
        alias="peakMoments",
        description="The highest evening peaks, highest first.",
    )
    summary: EveningPeakSummary = Field(description="Key figures shown under the cards.")
    reference: str | None = Field(
        default=None, description="The identifier supplied in the request, if any."
    )

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_result(
        cls,
        result: EveningPeakAnalysisResult,
        moments: list[PeakMoment],
        *,
        peak_share_threshold: float,
        reference: str | None = None,
    ) -> Self:
        """Build the response from an analyzer result."""
        daily = result.daily.collect()
        weeks = result.week_medians.collect()

        def series(frame: pl.DataFrame, index_column: str, value_column: str) -> TimeSeries:
            return TimeSeries(
                name=value_column,
                index=frame[index_column].to_list(),
                data=frame[value_column].to_list(),
            )

        return cls(
            daily_peak=series(daily, DAY, "evening_peak_in_kilowatt"),
            daily_share=series(daily, DAY, "evening_peak_share_in_percent"),
            week_median_peak=series(weeks, WEEK, "median_evening_peak_in_kilowatt"),
            week_median_share=series(weeks, WEEK, "median_evening_peak_share_in_percent"),
            peak_moments=[
                PeakMomentOutput(
                    peak_time=moment.peak_time,
                    peak_value_in_kilowatt=moment.peak_value_in_kilowatt,
                    day_curve=TimeSeries(
                        name="power_in_kilowatt",
                        index=moment.day_curve["timestamp"].to_list(),
                        data=moment.day_curve["power_in_kilowatt"].to_list(),
                    ),
                )
                for moment in moments
            ],
            summary=EveningPeakSummary.model_validate(
                summarize(result, peak_share_threshold=peak_share_threshold)
            ),
            reference=reference,
        )
