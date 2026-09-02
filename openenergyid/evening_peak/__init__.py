"""Evening peak avoidance analysis ("Avondpiek mijden").

Computes, per connection and per day, the highest quarter-hour power inside a fixed
evening window and the share of that day's net offtake which falls inside it.
"""

from .analysis import (
    EveningPeakAnalysisResult,
    EveningPeakAnalyzer,
    PeakMoment,
    summarize,
)
from .models import (
    EveningPeakInput,
    EveningPeakOutput,
    EveningPeakSummary,
    PeakMomentOutput,
)
from .schemas import DailyEveningPeakSchema, NetOfftakeSchema, WeekMedianSchema

__all__ = [
    "DailyEveningPeakSchema",
    "EveningPeakAnalysisResult",
    "EveningPeakAnalyzer",
    "EveningPeakInput",
    "EveningPeakOutput",
    "EveningPeakSummary",
    "NetOfftakeSchema",
    "PeakMoment",
    "PeakMomentOutput",
    "WeekMedianSchema",
    "summarize",
]
