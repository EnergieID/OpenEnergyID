"""Baseload analysis package for power consumption data."""

from .analysis import BaseloadAnalysisResult, BaseloadAnalyzer
from .exceptions import InsufficientDataError, InvalidDataError
from .models import (
    BaseloadResultSchema,
    MonthlyMedianBaseloadSchema,
    PowerReadingSchema,
    PowerSeriesSchema,
)

__version__ = "0.1.0"
__all__ = [
    "BaseloadAnalyzer",
    "BaseloadAnalysisResult",
    "InsufficientDataError",
    "InvalidDataError",
    "PowerReadingSchema",
    "PowerSeriesSchema",
    "BaseloadResultSchema",
    "MonthlyMedianBaseloadSchema",
]
