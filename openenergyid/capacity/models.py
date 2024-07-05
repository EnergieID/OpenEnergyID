import datetime
import pandas as pd
from pydantic import BaseModel, Field

from openenergyid.models import TimeSeries


class CapacityInput(BaseModel):
    """Model for capacity input"""

    timezone: str = Field(alias="timeZone")
    series: TimeSeries
    # fromDate: datetime.datetime
    # toDate: datetime.datetime

    class Config:
        populate_by_name = True

    @classmethod
    def from_pandas(cls, series: pd.Series, timezone: str = "UTC"):
        return cls(timeZone=timezone, series=TimeSeries.from_pandas(series))

    def get_series(self) -> pd.Series:
        """Return the pandas series ready for analysis."""
        return self.series.to_pandas(timezone=self.timezone)
