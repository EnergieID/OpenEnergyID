"""Baseload analysis package for power consumption data."""

from .analysis import BaseloadAnalyzer
from .exceptions import InsufficientDataError, InvalidDataError
from .models import BaseloadResultSchema, PowerReadingSchema, PowerSeriesSchema

__version__ = "0.1.0"
__all__ = [
    "BaseloadAnalyzer",
    "InsufficientDataError",
    "InvalidDataError",
    "PowerReadingSchema",
    "PowerSeriesSchema",
    "BaseloadResultSchema",
]
