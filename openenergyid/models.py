"""Data models for the Open Energy ID."""

import datetime as dt
from typing import Optional, overload

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

    @overload
    def to_json(self, path: None = None, **kwargs) -> str:
        ...

    @overload
    def to_json(self, path: str, **kwargs) -> None:
        ...

    def to_json(self, path: Optional[str] = None, **kwargs) -> Optional[str]:
        """Save the TimeSeries to a JSON file or return as string."""
        if path is None:
            return self.model_dump_json(**kwargs)
        else:
            encoding = kwargs.pop("encoding", "UTF-8")
            with open(path, "w", encoding=encoding) as file:
                file.write(self.model_dump_json(**kwargs))

    @overload
    @classmethod
    def from_json(cls, string: str, **kwargs) -> "TimeSeries":
        ...

    @overload
    @classmethod
    def from_json(cls, path: str, **kwargs) -> "TimeSeries":
        ...

    @classmethod
    def from_json(
        cls, string: Optional[str] = None, path: Optional[str] = None, **kwargs
    ) -> "TimeSeries":
        """Load the TimeSeries from a JSON file or string."""
        if string:
            return cls.model_validate_json(string, **kwargs)
        elif path:
            encoding = kwargs.pop("encoding", "UTF-8")
            with open(path, "r", encoding=encoding) as file:
                return cls.model_validate_json(file.read(), **kwargs)
        else:
            raise ValueError("Either string or path must be provided.")
