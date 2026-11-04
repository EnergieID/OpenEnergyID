"""Constants for the Elia open data API."""

import datetime as dt
from enum import Enum


class Region(Enum):
    """
    PV regions as Elia published them until October 2026.

    The values are kept stable for callers. Elia renamed the regions in its newer
    datasets; `PV_REGION_MEMBERS` maps each value to the names Elia uses now.
    """

    Belgium = "Belgium"
    Brussels = "Brussels"
    Flanders = "Flanders"
    Wallonia = "Wallonia"
    Antwerp = "Antwerp"
    East_Flanders = "East-Flanders"
    Flemish_Brabant = "Flemish-Brabant"
    Limburg = "Limburg"
    West_Flanders = "West-Flanders"
    Hainaut = "Hainaut"
    Liège = "Liège"
    Luxembourg = "Luxembourg"
    Namur = "Namur"
    Walloon_Brabant = "Walloon-Brabant"


# First day (Europe/Brussels) served by the PV datasets that replaced ods032 and ods087.
PV_DATASET_CUTOVER = dt.date(2026, 10, 26)

PV_HISTORICAL_DATASET_LEGACY = "ods032"
PV_HISTORICAL_DATASET = "ods209"
PV_NEAR_REALTIME_DATASET = "ods210"

_FLEMISH_PROVINCES = ("Antwerpen", "Limburg", "OostVlaanderen", "VlaamsBrabant", "WestVlaanderen")
_WALLOON_PROVINCES = ("BrabantWallon", "Hainaut", "Liege", "Luxembourg", "Namur")

# Region names in ods209/ods210 that make up each Region.
# The new datasets have no Flanders or Wallonia rows: those are summed from their provinces.
PV_REGION_MEMBERS: dict[Region, tuple[str, ...]] = {
    Region.Belgium: ("Belgium",),
    Region.Brussels: ("Brussel",),
    Region.Flanders: _FLEMISH_PROVINCES,
    Region.Wallonia: _WALLOON_PROVINCES,
    Region.Antwerp: ("Antwerpen",),
    Region.East_Flanders: ("OostVlaanderen",),
    Region.Flemish_Brabant: ("VlaamsBrabant",),
    Region.Limburg: ("Limburg",),
    Region.West_Flanders: ("WestVlaanderen",),
    Region.Hainaut: ("Hainaut",),
    Region.Liège: ("Liege",),
    Region.Luxembourg: ("Luxembourg",),
    Region.Namur: ("Namur",),
    Region.Walloon_Brabant: ("BrabantWallon",),
}
