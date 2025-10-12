"""Constants for the Open Energy ID package."""

from typing import Literal

# METRICS

ELECTRICITY_DELIVERED: Literal["electricity_delivered"] = "electricity_delivered"
ELECTRICITY_EXPORTED: Literal["electricity_exported"] = "electricity_exported"
ELECTRICITY_PRODUCED: Literal["electricity_produced"] = "electricity_produced"
ELECTRICITY_CONSUMED: Literal["electricity_consumed"] = "electricity_consumed"
ELECTRICITY_SELF_CONSUMED: Literal["electricity_self_consumed"] = "electricity_self_consumed"

PRICE_DAY_AHEAD: Literal["price_day_ahead"] = "price_day_ahead"
PRICE_IMBALANCE_UPWARD: Literal["price_imbalance_upward"] = "price_imbalance_upward"
PRICE_IMBALANCE_DOWNWARD: Literal["price_imbalance_downward"] = "price_imbalance_downward"
PRICE_ELECTRICITY_DELIVERED: Literal["price_electricity_delivered"] = "price_electricity_delivered"
PRICE_ELECTRICITY_EXPORTED: Literal["price_electricity_exported"] = "price_electricity_exported"

RLP: Literal["RLP"] = "RLP"
SPP: Literal["SPP"] = "SPP"

COST_ELECTRICITY_DELIVERED: Literal["cost_electricity_delivered"] = "cost_electricity_delivered"
EARNINGS_ELECTRICITY_EXPORTED: Literal["earnings_electricity_exported"] = (
    "earnings_electricity_exported"
)
COST_ELECTRICITY_NET: Literal["cost_electricity_net"] = "cost_electricity_net"

RATIO_SELF_CONSUMPTION: Literal["ratio_self_consumption"] = "ratio_self_consumption"
RATIO_SELF_SUFFICIENCY: Literal["ratio_self_sufficiency"] = "ratio_self_sufficiency"
