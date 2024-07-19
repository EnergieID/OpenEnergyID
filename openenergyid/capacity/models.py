"""Model for Capacity Analysis."""

from pydantic import BaseModel, Field
from openenergyid.models import TimeSeries


class CapacityInput(BaseModel):
    """Model for capacity input"""

    timezone: str = Field(alias="timeZone")
    series: TimeSeries


class PeakDetail(BaseModel):
    """Model for peak detail"""

    peak_time: str = Field(alias="peakTime")
    peak_value: float = Field(alias="peakValue")
    surrounding_data: TimeSeries = Field(alias="surroundingData")


class CapacityOutput(BaseModel):
    """Model for capacity output"""

    peaks: TimeSeries
    peak_details: list[PeakDetail] = Field(alias="peakDetails")
