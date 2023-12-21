"""Constants for the Open Energy ID package."""

from typing import Literal

# METRICS

ELECTRICITY_DELIVERED: Literal["electricity_delivered"] = "electricity_delivered"
ELECTRICITY_EXPORTED: Literal["electricity_exported"] = "electricity_exported"
ELECTRICITY_PRODUCED: Literal["electricity_produced"] = "electricity_produced"

PRICE_DAY_AHEAD: Literal["price_day_ahead"] = "price_day_ahead"
PRICE_IMBALANCE_UPWARD: Literal["price_imbalance_upward"] = "price_imbalance_upward"
PRICE_IMBALANCE_DOWNWARD: Literal["price_imbalance_downward"] = "price_imbalance_downward"
