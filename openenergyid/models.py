"""Data models for the Open Energy ID."""

import datetime as dt

import pandas as pd
from pydantic import BaseModel


class TimeSeries(BaseModel):
    """Time series data."""

    columns: list[str]
    index: list[dt.datetime]
    data: list[list[float]]

    @classmethod
    def from_pandas(cls, data: pd.DataFrame) -> "TimeSeries":
        """Create a MultiVariableRegressionInputFrame from a pandas DataFrame."""
        return cls.model_validate(data.to_dict(orient="split"))

    def to_pandas(self, timezone: str = "UTC") -> pd.DataFrame:
        """Convert the MultiVariableRegressionInputFrame to a pandas DataFrame."""
        frame = pd.DataFrame(self.data, columns=self.columns, index=self.index)
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame.tz_convert(timezone)
