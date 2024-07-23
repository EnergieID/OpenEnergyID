"""Model for Capacity Analysis."""

import datetime as dt
from pydantic import BaseModel, ConfigDict, Field
from openenergyid.models import TimeSeries


class CapacityInput(BaseModel):
    """Model for capacity input"""

    timezone: str = Field(alias="timeZone")
    series: TimeSeries


class PeakDetail(BaseModel):
    """Model for peak detail"""

    peak_time: dt.datetime = Field(alias="peakTime")
    peak_value: float = Field(alias="peakValue")
    surrounding_data: TimeSeries = Field(alias="surroundingData")
    model_config = ConfigDict(populate_by_name=True)


class CapacityOutput(BaseModel):
    """Model for capacity output"""

    peaks: TimeSeries
    peak_details: list[PeakDetail] = Field(alias="peakDetails")
    model_config = ConfigDict(populate_by_name=True)
