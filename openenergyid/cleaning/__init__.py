"""Source-data cleaning module.

Standalone data-quality filters that run as an explicit pre-step before
analysis modules such as MVLR. Analysis modules stay pure; callers decide
when to clean and carry the diagnostics.
"""

from .models import (
    FilteringDiagnostics,
    FilteringStatus,
    SolarSourceFilteringParameters,
)
from .solar import clean_solar_production_frame

__all__ = [
    "FilteringDiagnostics",
    "FilteringStatus",
    "SolarSourceFilteringParameters",
    "clean_solar_production_frame",
]
