"""Static enums for Open Energy ID."""

from enum import Enum


class Granularity(Enum):
    """Granularity of a time series."""

    P1Y = "P1Y"  # 1 year
    P1M = "P1M"  # 1 month
    P7D = "P7D"  # 7 days
    P1D = "P1D"  # 1 day
    PT1H = "PT1H"  # 1 hour
    PT15M = "PT15M"  # 15 minutes
    PT5M = "PT5M"  # 5 minutes
    PT1M = "PT1M"  # 1 minute
