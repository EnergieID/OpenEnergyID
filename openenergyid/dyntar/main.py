"""Main module of the DynTar package."""

from typing import Optional
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
)


def weigh_by_monthly_profile(series: pd.Series, profile: pd.Series) -> pd.Series:
    """Weigh a time series by a monthly profile."""
    df = pd.DataFrame({"series": series, "profile": profile})
    results = []
    for _, frame in df.groupby(pd.Grouper(freq="MS")):
        frame = frame.copy()
        frame["weighted"] = frame["series"].sum() * (frame["profile"] / frame["profile"].sum())
        results.append(frame)
    return pd.concat(results)["weighted"]


def extend_dataframe_with_smr2(df: pd.DataFrame, inplace: bool = False) -> Optional[pd.DataFrame]:
    """Extend a DataFrame with the SMR2 columns."""
    if not inplace:
        result_df = df.copy()
    else:
        result_df = df

    result_df[ELECTRICITY_DELIVERED_SMR2] = weigh_by_monthly_profile(
        df[ELECTRICITY_DELIVERED], df[RLP]
    )
    result_df[ELECTRICITY_EXPORTED_SMR2] = weigh_by_monthly_profile(
        df[ELECTRICITY_EXPORTED], df[SPP]
    )

    result_df.rename(
        columns={
            ELECTRICITY_DELIVERED: ELECTRICITY_DELIVERED_SMR3,
            ELECTRICITY_EXPORTED: ELECTRICITY_EXPORTED_SMR3,
        },
        inplace=True,
    )

    if not inplace:
        return result_df
    return None


def extend_dataframe_with_costs(df: pd.DataFrame, inplace: bool = False) -> Optional[pd.DataFrame]:
    """Extend a DataFrame with the cost columns."""
    if not inplace:
        result_df = df.copy()
    else:
        result_df = df

    result_df[COST_ELECTRICITY_DELIVERED_SMR2] = (
        df[ELECTRICITY_DELIVERED_SMR2] * df[PRICE_ELECTRICITY_DELIVERED]
    )
    result_df[COST_ELECTRICITY_EXPORTED_SMR2] = (
        df[ELECTRICITY_EXPORTED_SMR2] * df[PRICE_ELECTRICITY_EXPORTED]
    )

    result_df[COST_ELECTRICITY_DELIVERED_SMR3] = (
        df[ELECTRICITY_DELIVERED_SMR3] * df[PRICE_ELECTRICITY_DELIVERED]
    )
    result_df[COST_ELECTRICITY_EXPORTED_SMR3] = (
        df[ELECTRICITY_EXPORTED_SMR3] * df[PRICE_ELECTRICITY_EXPORTED]
    )

    if not inplace:
        return result_df
    return None


def extend_dataframe_with_weighted_prices(
    df: pd.DataFrame, inplace: bool = False
) -> Optional[pd.DataFrame]:
    """Extend a DataFrame with the weighted price columns."""
    if not inplace:
        df = df.copy()

    rlp_weighted_price_delivered = (df[PRICE_ELECTRICITY_DELIVERED] * df[RLP]).resample(
        "MS"
    ).sum() / df[RLP].resample("MS").sum()
    df[RLP_WEIGHTED_PRICE_DELIVERED] = rlp_weighted_price_delivered.reindex_like(
        df[RLP], method="ffill"
    )
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
    df: pd.DataFrame, inplace: bool = False
) -> Optional[pd.DataFrame]:
    """Extend a DataFrame with the heatmap columns."""
    if not inplace:
        df = df.copy()

    heatmap_score_delivered = (
        (df[ELECTRICITY_DELIVERED_SMR2] - df[ELECTRICITY_DELIVERED_SMR3])
        / df[ELECTRICITY_DELIVERED_SMR2]
        * (df[RLP_WEIGHTED_PRICE_DELIVERED] - df[PRICE_ELECTRICITY_DELIVERED])
        / df[RLP_WEIGHTED_PRICE_DELIVERED]
    )
    heatmap_score_exported = (
        (df[ELECTRICITY_EXPORTED_SMR2] - df[ELECTRICITY_EXPORTED_SMR3])
        / df[ELECTRICITY_EXPORTED_SMR2]
        * (df[SPP_WEIGHTED_PRICE_EXPORTED] - df[PRICE_ELECTRICITY_EXPORTED])
        / df[SPP_WEIGHTED_PRICE_EXPORTED]
    )
    heatmap_score_delivered.fillna(0, inplace=True)
    heatmap_score_exported.fillna(0, inplace=True)

    # Invert scores so that positive values indicate a positive impact
    heatmap_score_delivered = -heatmap_score_delivered
    heatmap_score_exported = -heatmap_score_exported
    heatmap_score_combined = heatmap_score_delivered - heatmap_score_exported

    df[HEATMAP_DELIVERED] = heatmap_score_delivered
    df[HEATMAP_EXPORTED] = heatmap_score_exported
    df[HEATMAP_TOTAL] = heatmap_score_combined

    if not inplace:
        return df
    return None


def calculate_dyntar_columns(df: pd.DataFrame, inplace: bool = False) -> Optional[pd.DataFrame]:
    """Calculate all columns required for the dynamic tariff analysis."""
    if not inplace:
        df = df.copy()

    extend_dataframe_with_smr2(df, inplace=True)
    extend_dataframe_with_costs(df, inplace=True)
    extend_dataframe_with_weighted_prices(df, inplace=True)
    extend_dataframe_with_heatmap(df, inplace=True)

    if not inplace:
        return df
    return None
