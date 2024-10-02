"""Main module of the DynTar package."""

import numpy as np
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


def extend_dataframe_with_smr2(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame | None:
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


def extend_dataframe_with_costs(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame | None:
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
) -> pd.DataFrame | None:
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


def extend_dataframe_with_heatmap(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame | None:
    """Extend a DataFrame with the heatmap columns."""
    if not inplace:
        df = df.copy()

    normalized_energy_delta_delivered = (
        df[ELECTRICITY_DELIVERED_SMR2] - df[ELECTRICITY_DELIVERED_SMR3]
    ) / df[ELECTRICITY_DELIVERED_SMR2]
    normalized_price_delta_delivered = (
        df[RLP_WEIGHTED_PRICE_DELIVERED] - df[PRICE_ELECTRICITY_DELIVERED]
    ) / df[RLP_WEIGHTED_PRICE_DELIVERED]
    heatmap_score_delivered = normalized_energy_delta_delivered * normalized_price_delta_delivered

    normalized_energy_delta_exported = (
        df[ELECTRICITY_EXPORTED_SMR2] - df[ELECTRICITY_EXPORTED_SMR3]
    ) / df[ELECTRICITY_EXPORTED_SMR2]
    normalized_energy_delta_exported = normalized_energy_delta_exported.replace(
        [np.inf, -np.inf], np.nan
    )
    normalized_price_delta_exported = (
        df[SPP_WEIGHTED_PRICE_EXPORTED] - df[PRICE_ELECTRICITY_EXPORTED]
    ) / df[SPP_WEIGHTED_PRICE_EXPORTED]
    heatmap_score_exported = normalized_energy_delta_exported * normalized_price_delta_exported

    heatmap_score_delivered.fillna(0, inplace=True)
    heatmap_score_exported.fillna(0, inplace=True)

    # Invert scores so that positive values indicate a positive impact
    heatmap_score_delivered = -heatmap_score_delivered
    heatmap_score_combined = heatmap_score_delivered + heatmap_score_exported

    df[HEATMAP_DELIVERED] = heatmap_score_delivered
    df[HEATMAP_EXPORTED] = heatmap_score_exported
    df[HEATMAP_TOTAL] = heatmap_score_combined

    if not inplace:
        return df
    return None


def extend_dataframe_with_heatmap_description(
    df: pd.DataFrame, inplace: bool = False
) -> pd.DataFrame | None:
    """Extend a DataFrame with the heatmap description columns."""
    if not inplace:
        df = df.copy()

    # Delivered

    # Where Heatmap is 0, we put a desription of 0 (No impact)
    df[HEATMAP_DELIVERED_DESCRIPTION] = df[HEATMAP_DELIVERED].apply(
        lambda x: 0 if x == 0 else float("NaN")
    )
    # When the energy delta is positive, and the price delta is positive, we put a description of 1 (high consumption, high price)
    df[HEATMAP_DELIVERED_DESCRIPTION] = df.apply(
        lambda x: 1
        if x[PRICE_ELECTRICITY_DELIVERED] > x[RLP_WEIGHTED_PRICE_DELIVERED]
        and x[ELECTRICITY_DELIVERED_SMR3] > x[ELECTRICITY_DELIVERED_SMR2]
        else x[HEATMAP_DELIVERED_DESCRIPTION],
        axis=1,
    )
    # When the energy delta is negative, and the price delta is positive, we put a description of 2 (low consumption, high price)
    df[HEATMAP_DELIVERED_DESCRIPTION] = df.apply(
        lambda x: 2
        if x[PRICE_ELECTRICITY_DELIVERED] > x[RLP_WEIGHTED_PRICE_DELIVERED]
        and x[ELECTRICITY_DELIVERED_SMR3] < x[ELECTRICITY_DELIVERED_SMR2]
        else x[HEATMAP_DELIVERED_DESCRIPTION],
        axis=1,
    )
    # When the energy delta is positive, and the price delta is negative, we put a description of 3 (high consumption, low price)
    df[HEATMAP_DELIVERED_DESCRIPTION] = df.apply(
        lambda x: 3
        if x[PRICE_ELECTRICITY_DELIVERED] < x[RLP_WEIGHTED_PRICE_DELIVERED]
        and x[ELECTRICITY_DELIVERED_SMR3] > x[ELECTRICITY_DELIVERED_SMR2]
        else x[HEATMAP_DELIVERED_DESCRIPTION],
        axis=1,
    )
    # When the energy delta is negative, and the price delta is negative, we put a description of 4 (low consumption, low price)
    df[HEATMAP_DELIVERED_DESCRIPTION] = df.apply(
        lambda x: 4
        if x[PRICE_ELECTRICITY_DELIVERED] < x[RLP_WEIGHTED_PRICE_DELIVERED]
        and x[ELECTRICITY_DELIVERED_SMR3] < x[ELECTRICITY_DELIVERED_SMR2]
        else x[HEATMAP_DELIVERED_DESCRIPTION],
        axis=1,
    )
    # All other cases are put as 0
    df[HEATMAP_DELIVERED_DESCRIPTION] = df[HEATMAP_DELIVERED_DESCRIPTION].replace(np.nan, 0)

    # Exported

    # Where Heatmap is 0, we put a desription of 0 (No impact)
    df[HEATMAP_EXPORTED_DESCRIPTION] = df[HEATMAP_EXPORTED].apply(
        lambda x: 0 if x == 0 else float("NaN")
    )
    # When the energy delta is positive, and the price delta is positive, we put a description of 5 (high injection, high price)
    df[HEATMAP_EXPORTED_DESCRIPTION] = df.apply(
        lambda x: 5
        if x[PRICE_ELECTRICITY_EXPORTED] > x[SPP_WEIGHTED_PRICE_EXPORTED]
        and x[ELECTRICITY_EXPORTED_SMR3] > x[ELECTRICITY_EXPORTED_SMR2]
        else x[HEATMAP_EXPORTED_DESCRIPTION],
        axis=1,
    )
    # When the energy delta is negative, and the price delta is positive, we put a description of 6 (low injection, high price)
    df[HEATMAP_EXPORTED_DESCRIPTION] = df.apply(
        lambda x: 6
        if x[PRICE_ELECTRICITY_EXPORTED] > x[SPP_WEIGHTED_PRICE_EXPORTED]
        and x[ELECTRICITY_EXPORTED_SMR3] < x[ELECTRICITY_EXPORTED_SMR2]
        else x[HEATMAP_EXPORTED_DESCRIPTION],
        axis=1,
    )
    # When the energy delta is positive, and the price delta is negative, we put a description of 7 (high injection, low price)
    df[HEATMAP_EXPORTED_DESCRIPTION] = df.apply(
        lambda x: 7
        if x[PRICE_ELECTRICITY_EXPORTED] < x[SPP_WEIGHTED_PRICE_EXPORTED]
        and x[ELECTRICITY_EXPORTED_SMR3] > x[ELECTRICITY_EXPORTED_SMR2]
        else x[HEATMAP_EXPORTED_DESCRIPTION],
        axis=1,
    )
    # When the energy delta is negative, and the price delta is negative, we put a description of 8 (low injection, low price)
    df[HEATMAP_EXPORTED_DESCRIPTION] = df.apply(
        lambda x: 8
        if x[PRICE_ELECTRICITY_EXPORTED] < x[SPP_WEIGHTED_PRICE_EXPORTED]
        and x[ELECTRICITY_EXPORTED_SMR3] < x[ELECTRICITY_EXPORTED_SMR2]
        else x[HEATMAP_EXPORTED_DESCRIPTION],
        axis=1,
    )
    # All other cases are put as 0
    df[HEATMAP_EXPORTED_DESCRIPTION] = df[HEATMAP_EXPORTED_DESCRIPTION].replace(np.nan, 0)

    # Total

    # We see which of the individual heatmaps has the highest absolute value
    # We put the description of the highest absolute value
    df[HEATMAP_TOTAL_DESCRIPTION] = df.apply(
        lambda x: x[HEATMAP_DELIVERED_DESCRIPTION]
        if abs(x[HEATMAP_DELIVERED]) > abs(x[HEATMAP_EXPORTED])
        else x[HEATMAP_EXPORTED_DESCRIPTION],
        axis=1,
    )
    # Where Heatmap is 0, we put a desription of 0 (No impact)
    df[HEATMAP_TOTAL_DESCRIPTION] = df.apply(
        lambda x: 0 if x[HEATMAP_TOTAL] == 0 else x[HEATMAP_TOTAL_DESCRIPTION], axis=1
    )
    # All other cases are put as 0
    df[HEATMAP_TOTAL_DESCRIPTION] = df[HEATMAP_TOTAL_DESCRIPTION].replace(np.nan, 0)

    if not inplace:
        return df


def calculate_dyntar_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame | None:
    """Calculate all columns required for the dynamic tariff analysis."""
    if not inplace:
        df = df.copy()

    extend_dataframe_with_smr2(df, inplace=True)
    extend_dataframe_with_costs(df, inplace=True)
    extend_dataframe_with_weighted_prices(df, inplace=True)
    extend_dataframe_with_heatmap(df, inplace=True)
    extend_dataframe_with_heatmap_description(df, inplace=True)

    if not inplace:
        return df
    return None
