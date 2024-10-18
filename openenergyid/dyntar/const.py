"""Constants for the dyntar analysis."""

from enum import Enum

ELECTRICITY_DELIVERED_SMR3 = "electricity_delivered_smr3"
ELECTRICITY_EXPORTED_SMR3 = "electricity_exported_smr3"
ELECTRICITY_DELIVERED_SMR2 = "electricity_delivered_smr2"
ELECTRICITY_EXPORTED_SMR2 = "electricity_exported_smr2"

COST_ELECTRICITY_DELIVERED_SMR2 = "cost_electricity_delivered_smr2"
COST_ELECTRICITY_EXPORTED_SMR2 = "cost_electricity_exported_smr2"
COST_ELECTRICITY_DELIVERED_SMR3 = "cost_electricity_delivered_smr3"
COST_ELECTRICITY_EXPORTED_SMR3 = "cost_electricity_exported_smr3"

RLP_WEIGHTED_PRICE_DELIVERED = "rlp_weighted_price_delivered"
SPP_WEIGHTED_PRICE_EXPORTED = "spp_weighted_price_exported"

HEATMAP_DELIVERED = "heatmap_delivered"
HEATMAP_EXPORTED = "heatmap_exported"
HEATMAP_TOTAL = "heatmap_total"

HEATMAP_DELIVERED_DESCRIPTION = "heatmap_delivered_description"
HEATMAP_EXPORTED_DESCRIPTION = "heatmap_exported_description"
HEATMAP_TOTAL_DESCRIPTION = "heatmap_total_description"


class Register(Enum):
    """Register for dynamic tariff analysis."""

    DELIVERY = "delivery"
    EXPORT = "export"
