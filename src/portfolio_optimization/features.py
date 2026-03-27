import numpy as np
import pandas as pd

from .config import ASSETS, TEST_START, TRAIN_END


def combine_frames(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train_df, test_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def monthly_asset_log_returns(
    train_asset: pd.DataFrame,
    test_asset: pd.DataFrame,
    assets: list[str] | None = None,
) -> pd.DataFrame:
    assets = assets or ASSETS
    combined_assets = combine_frames(train_asset, test_asset)
    monthly_prices = combined_assets[assets].resample("ME").last().ffill()
    return np.log(monthly_prices / monthly_prices.shift(1)).dropna()


def engineer_macro_features(macro_monthly: pd.DataFrame) -> pd.DataFrame:
    features = macro_monthly.copy().ffill().bfill()

    feature_map = {
        "CPI_Change": "CPI",
        "M2_Change": "M2",
        "VIX_Change": "India_VIX",
        "Oil_Change": "Oil_Brent_USD",
        "Gold_Change": "Gold_USD",
        "Forex_Change": "IN_Forex_Reserves_USD",
        "DXY_Change": "DXY_Dollar_Index",
        "US_IP_Change": "US_Industrial_Production",
        "USDINR_Change": "USDINR",
        "US_VIX_Change": "US_VIX",
    }

    for engineered_col, source_col in feature_map.items():
        if source_col in features.columns:
            features[engineered_col] = features[source_col].pct_change()

    return features.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def monthly_engineered_macro_features(
    train_macro: pd.DataFrame,
    test_macro: pd.DataFrame,
) -> pd.DataFrame:
    combined_macro = combine_frames(train_macro, test_macro)
    monthly_macro = combined_macro.resample("ME").last().ffill().bfill()
    return engineer_macro_features(monthly_macro)


def build_supervised_datasets(
    train_asset: pd.DataFrame,
    test_asset: pd.DataFrame,
    train_macro: pd.DataFrame,
    test_macro: pd.DataFrame,
    assets: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    assets = assets or ASSETS
    monthly_returns = monthly_asset_log_returns(train_asset, test_asset, assets)
    monthly_macro = monthly_engineered_macro_features(train_macro, test_macro)
    lagged_features = monthly_macro.shift(1)

    dataset = pd.concat([monthly_returns, lagged_features], axis=1).dropna()
    feature_cols = [col for col in lagged_features.columns if col in dataset.columns]

    train_data = dataset.loc[dataset.index <= TRAIN_END].copy()
    test_data = dataset.loc[dataset.index >= TEST_START].copy()
    asset_returns = monthly_returns.loc[dataset.index].copy()
    return train_data, test_data, feature_cols, asset_returns

