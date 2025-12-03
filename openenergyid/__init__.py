"""Open Energy ID Python SDK."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openenergyid")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

from .enums import Granularity
from .models import TimeDataFrame, TimeSeries

__all__ = ["Granularity", "TimeDataFrame", "TimeSeries"]
