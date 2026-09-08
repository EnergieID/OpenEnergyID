"""Bridging pandas offset aliases and ISO 8601 billing resolutions."""

import isodate
from energy_cost.resolution import Resolution

# Only frequencies with an exact ISO 8601 equivalent. Anything anchored to a weekday
# (e.g. "W-MON") has no ISO duration and no meaning as a billing period.
PANDAS_FREQ_TO_ISO: dict[str, str] = {
    "YS": "P1Y",
    "YE": "P1Y",
    "QS": "P3M",
    "MS": "P1M",
    "ME": "P1M",
    "D": "P1D",
    "h": "PT1H",
    "H": "PT1H",
    "15min": "PT15M",
    "15T": "PT15M",
}


def pandas_freq_to_resolution(freq: str) -> Resolution | None:
    """Best-effort mapping of a pandas offset alias to an ISO 8601 Resolution.

    Returns ``None`` when the frequency has no ISO equivalent, so callers can fall back
    to their own default rather than guessing.
    """
    iso = PANDAS_FREQ_TO_ISO.get(freq)
    if iso is None:
        return None
    return isodate.parse_duration(iso)
