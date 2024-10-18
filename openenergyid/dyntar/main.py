"""Main module of the DynTar package."""

from typing import cast
import pandas as pd

from openenergyid.const import (
    ELECTRICITY_DELIVERED,
    ELECTRICITY_EXPORTED,
    PRICE_ELECTRICITY_DELIVERED,
    PRICE_ELECTRICITY_EXPORTED,
    RLP,
    SPP,
)

from .const import (
    ELECTRICITY_DELIVERED_SMR3,
    ELECTRICITY_EXPORTED_SMR3,
    ELECTRICITY_DELIVERED_SMR2,
    ELECTRICITY_EXPORTED_SMR2,
    COST_ELECTRICITY_DELIVERED_SMR2,
    COST_ELECTRICITY_EXPORTED_SMR2,
    COST_ELECTRICITY_DELIVERED_SMR3,
    COST_ELECTRICITY_EXPORTED_SMR3,
    RLP_WEIGHTED_PRICE_DELIVERED,
    SPP_WEIGHTED_PRICE_EXPORTED,
    HEATMAP_DELIVERED,
    HEATMAP_EXPORTED,
    HEATMAP_TOTAL,
    HEATMAP_DELIVERED_DESCRIPTION,
    HEATMAP_EXPORTED_DESCRIPTION,
    HEATMAP_TOTAL_DESCRIPTION,
    Register,
)


def weigh_by_monthly_profile(df: pd.DataFrame, series_name, profile_name) -> pd.Series:
    """Weigh a time series by a monthly profile."""
    grouped = df.groupby(pd.Grouper(freq="MS"))
    return grouped[series_name].transform("sum") * grouped[profile_name].transform(
        lambda x: x / x.sum()
    )


def extend_dataframe_with_smr2(
    df: pd.DataFrame,
    inplace: bool = False,
    registers: list[Register] | None = None,
) -> pd.DataFrame | None:
    """Extend a DataFrame with the SMR2 columns."""
    if not inplace:
        result_df = df.copy()
    else:
        result_df = df

    if registers is None:
        registers = [Register.DELIVERY, Register.EXPORT]

    if Register.DELIVERY in registers:
        result_df[ELECTRICITY_DELIVERED_SMR2] = weigh_by_monthly_profile(
            df, ELECTRICITY_DELIVERED, RLP
        )
    if Register.EXPORT in registers:
        result_df[ELECTRICITY_EXPORTED_SMR2] = weigh_by_monthly_profile(
            df, ELECTRICITY_EXPORTED, SPP
        )

    result_df.rename(
        columns={
            ELECTRICITY_DELIVERED: ELECTRICITY_DELIVERED_SMR3,
            ELECTRICITY_EXPORTED: ELECTRICITY_EXPORTED_SMR3,
        },
        inplace=True,
        errors="ignore",
    )

    if not inplace:
        return result_df
    return None


def extend_dataframe_with_costs(
    df: pd.DataFrame, inplace: bool = False, registers: list[Register] | None = None
) -> pd.DataFrame | None:
    """Extend a DataFrame with the cost columns."""
    if not inplace:
        result_df = df.copy()
    else:
        result_df = df

    if registers is None:
        registers = [Register.DELIVERY, Register.EXPORT]

    if Register.DELIVERY in registers:
        result_df[COST_ELECTRICITY_DELIVERED_SMR2] = (
            df[ELECTRICITY_DELIVERED_SMR2] * df[PRICE_ELECTRICITY_DELIVERED]
        )
        result_df[COST_ELECTRICITY_DELIVERED_SMR3] = (
            df[ELECTRICITY_DELIVERED_SMR3] * df[PRICE_ELECTRICITY_DELIVERED]
        )

    if Register.EXPORT in registers:
        result_df[COST_ELECTRICITY_EXPORTED_SMR2] = (
            df[ELECTRICITY_EXPORTED_SMR2] * df[PRICE_ELECTRICITY_EXPORTED] * -1
        )
        result_df[COST_ELECTRICITY_EXPORTED_SMR3] = (
            df[ELECTRICITY_EXPORTED_SMR3] * df[PRICE_ELECTRICITY_EXPORTED] * -1
        )

    if not inplace:
        return result_df
    return None


def extend_dataframe_with_weighted_prices(
    df: pd.DataFrame, inplace: bool = False, registers: list[Register] | None = None
) -> pd.DataFrame | None:
    """Extend a DataFrame with the weighted price columns."""
    if not inplace:
        df = df.copy()

    if registers is None:
        registers = [Register.DELIVERY, Register.EXPORT]

    if Register.DELIVERY in registers:
        rlp_weighted_price_delivered = (df[PRICE_ELECTRICITY_DELIVERED] * df[RLP]).resample(
            "MS"
        ).sum() / df[RLP].resample("MS").sum()
        df[RLP_WEIGHTED_PRICE_DELIVERED] = rlp_weighted_price_delivered.reindex_like(
            df[RLP], method="ffill"
        )

    if Register.EXPORT in registers:
        spp_weighted_price_exported = (df[PRICE_ELECTRICITY_EXPORTED] * df[SPP]).resample(
            "MS"
        ).sum() / df[SPP].resample("MS").sum()
        df[SPP_WEIGHTED_PRICE_EXPORTED] = spp_weighted_price_exported.reindex_like(
            df[SPP], method="ffill"
        )

    if not inplace:
        return df
    return None


def extend_dataframe_with_heatmap(
    df: pd.DataFrame, inplace: bool = False, registers: list[Register] | None = None
) -> pd.DataFrame | None:
    """Extend a DataFrame with the heatmap columns."""
    if not inplace:
        df = df.copy()

    if registers is None:
        registers = [Register.DELIVERY, Register.EXPORT]

    if Register.DELIVERY in registers:
        energy_delta_delivered = df[ELECTRICITY_DELIVERED_SMR2] - df[ELECTRICITY_DELIVERED_SMR3]
        price_delta_delivered = df[RLP_WEIGHTED_PRICE_DELIVERED] - df[PRICE_ELECTRICITY_DELIVERED]
        heatmap_score_delivered = energy_delta_delivered * price_delta_delivered
        heatmap_score_delivered.fillna(0, inplace=True)
        # Invert score so that positive values indicate a positive impact
        heatmap_score_delivered = -heatmap_score_delivered
        df[HEATMAP_DELIVERED] = heatmap_score_delivered

    if Register.EXPORT in registers:
        energy_delta_exported = df[ELECTRICITY_EXPORTED_SMR2] - df[ELECTRICITY_EXPORTED_SMR3]
        price_delta_exported = df[SPP_WEIGHTED_PRICE_EXPORTED] - df[PRICE_ELECTRICITY_EXPORTED]
        heatmap_score_exported = energy_delta_exported * price_delta_exported
        heatmap_score_exported.fillna(0, inplace=True)
        df[HEATMAP_EXPORTED] = heatmap_score_exported

    if Register.DELIVERY in registers and Register.EXPORT in registers:
        heatmap_score_delivered = cast(pd.Series, df[HEATMAP_DELIVERED])
        heatmap_score_exported = cast(pd.Series, df[HEATMAP_EXPORTED])
        heatmap_score_combined = heatmap_score_delivered + heatmap_score_exported
    elif Register.DELIVERY in registers:
        heatmap_score_combined = heatmap_score_delivered
    else:
        heatmap_score_combined = heatmap_score_exported
    df[HEATMAP_TOTAL] = heatmap_score_combined

    if not inplace:
        return df
    return None


def map_delivery_description(
    price_delivered, price_rlp, electricity_delivered_smr3, electricity_delivered_smr2
):
    """Map the delivery description."""
    if price_delivered > price_rlp and electricity_delivered_smr3 > electricity_delivered_smr2:
        return 1
    if price_delivered > price_rlp and electricity_delivered_smr3 < electricity_delivered_smr2:
        return 2
    if price_delivered < price_rlp and electricity_delivered_smr3 > electricity_delivered_smr2:
        return 3
    if price_delivered < price_rlp and electricity_delivered_smr3 < electricity_delivered_smr2:
        return 4
    return 0


def map_export_description(
    price_exported, price_spp, electricity_exported_smr3, electricity_exported_smr2
):
    """Map the export description."""
    if price_exported > price_spp and electricity_exported_smr3 > electricity_exported_smr2:
        return 5
    if price_exported > price_spp and electricity_exported_smr3 < electricity_exported_smr2:
        return 6
    if price_exported < price_spp and electricity_exported_smr3 > electricity_exported_smr2:
        return 7
    if price_exported < price_spp and electricity_exported_smr3 < electricity_exported_smr2:
        return 8
    return 0


def map_total_description(
    abs_heatmap_delivered, abs_heatmap_exported, delivered_description, exported_description
):
    """Map the total description."""
    if abs_heatmap_delivered > abs_heatmap_exported:
        return delivered_description
    return exported_description


def extend_dataframe_with_heatmap_description(
    df: pd.DataFrame, inplace: bool = False, registers: list[Register] | None = None
) -> pd.DataFrame | None:
    """Extend a DataFrame with the heatmap description columns."""
    if not inplace:
        df = df.copy()

    if registers is None:
        registers = [Register.DELIVERY, Register.EXPORT]

    if Register.DELIVERY in registers:
        df[HEATMAP_DELIVERED_DESCRIPTION] = list(
            map(
                map_delivery_description,
                df[PRICE_ELECTRICITY_DELIVERED],
                df[RLP_WEIGHTED_PRICE_DELIVERED],
                df[ELECTRICITY_DELIVERED_SMR3],
                df[ELECTRICITY_DELIVERED_SMR2],
            )
        )

    if Register.EXPORT in registers:
        df[HEATMAP_EXPORTED_DESCRIPTION] = list(
            map(
                map_export_description,
                df[PRICE_ELECTRICITY_EXPORTED],
                df[SPP_WEIGHTED_PRICE_EXPORTED],
                df[ELECTRICITY_EXPORTED_SMR3],
                df[ELECTRICITY_EXPORTED_SMR2],
            )
        )

    if Register.DELIVERY in registers and Register.EXPORT in registers:
        df[HEATMAP_TOTAL_DESCRIPTION] = list(
            map(
                map_total_description,
                df[HEATMAP_DELIVERED].abs(),
                df[HEATMAP_EXPORTED].abs(),
                df[HEATMAP_DELIVERED_DESCRIPTION],
                df[HEATMAP_EXPORTED_DESCRIPTION],
            )
        )
    elif Register.DELIVERY in registers:
        df[HEATMAP_TOTAL_DESCRIPTION] = df[HEATMAP_DELIVERED_DESCRIPTION]
    else:
        df[HEATMAP_TOTAL_DESCRIPTION] = df[HEATMAP_EXPORTED_DESCRIPTION]

    if not inplace:
        return df


def calculate_dyntar_columns(
    df: pd.DataFrame,
    inplace: bool = False,
    registers: list[Register] | None = None,
) -> pd.DataFrame | None:
    """Calculate all columns required for the dynamic tariff analysis."""
    if not inplace:
        df = df.copy()

    if registers is None:
        registers = [Register.DELIVERY, Register.EXPORT]

    extend_dataframe_with_smr2(df, inplace=True, registers=registers)
    extend_dataframe_with_costs(df, inplace=True, registers=registers)
    extend_dataframe_with_weighted_prices(df, inplace=True, registers=registers)
    extend_dataframe_with_heatmap(df, inplace=True, registers=registers)
    extend_dataframe_with_heatmap_description(df, inplace=True, registers=registers)

    if not inplace:
        return df
    return None


def summarize_result(df: pd.DataFrame) -> pd.Series:
    """Summarize the dynamic tariff analysis result."""
    summary = df.filter(like="cost").sum()

    abs_smr2 = summary.filter(like="smr2").abs().sum()

    summary["cost_electricity_total_smr2"] = summary.filter(like="smr2").sum()
    summary["cost_electricity_total_smr3"] = summary.filter(like="smr3").sum()

    summary["ratio"] = (
        summary["cost_electricity_total_smr3"] - summary["cost_electricity_total_smr2"]
    ) / abs_smr2

    return summary
