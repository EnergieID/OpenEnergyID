"""Baseload analysis package for power consumption data."""

from .models import PowerReadingSchema, PowerSeriesSchema, BaseloadResultSchema
from .analysis import BaseloadAnalyzer
from .exceptions import InsufficientDataError, InvalidDataError

__version__ = "0.1.0"
__all__ = [
    "BaseloadAnalyzer",
    "InsufficientDataError",
    "InvalidDataError",
    "PowerReadingSchema",
    "PowerSeriesSchema",
    "BaseloadResultSchema",
]
