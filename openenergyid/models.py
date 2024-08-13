"""Data models for the Open Energy ID."""

import datetime as dt
from typing import overload

from typing import Self

import pandas as pd
from pydantic import BaseModel, field_validator


class TimeSeriesBase(BaseModel):
    """Pydantic base model for time series data."""

    index: list[dt.datetime]

    @classmethod
    def from_pandas(cls, data: pd.Series | pd.DataFrame) -> Self:
        """Create from a Pandas Object."""
        raise NotImplementedError

    def to_pandas(self, timezone: str = "UTC") -> pd.Series | pd.DataFrame:
        """Convert to a Pandas Object."""
        raise NotImplementedError

    @overload
    def to_json(self, path: None = None, **kwargs) -> str:
        """Dump to a JSON string."""

    @overload
    def to_json(self, path: str, **kwargs) -> None:
        """Dump to a JSON file."""

    def to_json(self, path: str | None = None, **kwargs) -> str | None:
        """Dump to a JSON string or file."""
        if path is None:
            return self.model_dump_json(**kwargs)
        encoding = kwargs.pop("encoding", "UTF-8")
        with open(path, "w", encoding=encoding) as file:
            file.write(self.model_dump_json(**kwargs))
        return None

    @overload
    @classmethod
    def from_json(cls, string: str, **kwargs) -> Self:
        """Load from a JSON string."""

    @overload
    @classmethod
    def from_json(cls, *, path: str, **kwargs) -> Self:
        """Load from a JSON file."""

    @classmethod
    def from_json(cls, string: str | None = None, path: str | None = None, **kwargs) -> Self:
        """Load from a JSON file or string."""
        if string:
            return cls.model_validate_json(string, **kwargs)
        if path:
            encoding = kwargs.pop("encoding", "UTF-8")
            with open(path, encoding=encoding) as file:
                return cls.model_validate_json(file.read(), **kwargs)
        raise ValueError("Either string or path must be provided.")


class TimeSeries(TimeSeriesBase):
    """
    Represents a time series data.
    Attributes:
        name (str | None): The name of the time series.
        data (list[float | None]): The data points of the time series.
    Methods:
        replace_nan_with_none(cls, data: list[float]) -> list[float | None]:
            Replace NaN values with None.
        from_pandas(cls, data: pd.Series) -> Self:
            Create a TimeSeries object from a Pandas Series.
        to_pandas(self, timezone: str = "UTC") -> pd.Series:
            Convert the TimeSeries object to a Pandas Series.
    """

    name: str | None = None
    data: list[float | None]

    @field_validator("data")
    @classmethod
    def replace_nan_with_none(cls, data: list[float]) -> list[float | None]:
        """Replace NaN values with None."""
        return [None if pd.isna(value) else value for value in data]

    @classmethod
    def from_pandas(cls, data: pd.Series) -> Self:
        """Create from a Pandas Series."""
        return cls(name=str(data.name), data=data.tolist(), index=data.index.tolist())

    def to_pandas(self, timezone: str = "UTC") -> pd.Series:
        """Convert to a Pandas Series."""
        series = pd.Series(self.data, name=self.name, index=self.index)
        series.index = pd.to_datetime(series.index, utc=True)
        return series.tz_convert(timezone)


class TimeDataFrame(TimeSeriesBase):
    """Time series data with multiple columns."""

    columns: list[str]
    data: list[list[float | None]]

    @field_validator("data")
    @classmethod
    def replace_nan_with_none(cls, data: list[list[float]]) -> list[list[float | None]]:
        """Replace NaN values with None."""
        return [[None if pd.isna(value) else value for value in row] for row in data]

    @classmethod
    def from_pandas(cls, data: pd.DataFrame) -> Self:
        """Create from a Pandas DataFrame."""
        return cls(
            columns=data.columns.tolist(), data=data.values.tolist(), index=data.index.tolist()
        )

    def to_pandas(self, timezone: str = "UTC") -> pd.DataFrame:
        """Convert to a Pandas DataFrame."""
        frame = pd.DataFrame(self.data, columns=self.columns, index=self.index)
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame.tz_convert(timezone)

    @classmethod
    def from_timeseries(cls, data: list[TimeSeries]) -> Self:
        """Create from a list of TimeSeries objects."""
        return cls.model_construct(
            columns=[series.name for series in data],
            data=[series.data for series in data],
            index=data[0].index,
        )

    def to_timeseries(self) -> list[TimeSeries]:
        """Convert to a list of TimeSeries objects."""
        return [
            TimeSeries(name=column, data=column_data, index=self.index)
            for column, column_data in zip(self.columns, self.data)
        ]
