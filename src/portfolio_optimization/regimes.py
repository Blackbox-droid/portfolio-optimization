import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import (
    ASSETS,
    ASSET_COLORS,
    NUM_REGIMES,
    PCA_VARIANCE_THRESHOLD,
    RANDOM_STATE,
    REGIME_COLORS,
    REGIME_NAMES,
)
from .features import build_supervised_datasets, combine_frames


def merge_macro_sources(
    india_train: pd.DataFrame,
    india_test: pd.DataFrame,
    global_train: pd.DataFrame,
    global_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    india_only_cols = [col for col in india_train.columns if col not in global_train.columns]
    train_merged = global_train.join(india_train[india_only_cols], how="outer").sort_index()
    test_merged = global_test.join(india_test[india_only_cols], how="outer").sort_index()
    return train_merged, test_merged


def _preferred_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _map_clusters_to_regimes(
    centroid_df: pd.DataFrame,
) -> dict[int, str]:
    columns = list(centroid_df.columns)
    risk_col = _preferred_column(columns, ["India_VIX", "US_VIX", "VIX_Change", "US_VIX_Change"])
    inflation_col = _preferred_column(columns, ["CPI", "IN_CPI_YoY", "US_CPI_YoY"])
    activity_col = _preferred_column(columns, ["PMI", "US_CFNAI", "US_Industrial_Production"])

    remaining = set(centroid_df.index.tolist())
    mapping = {}

    if risk_col is not None:
        risk_cluster = centroid_df[risk_col].idxmax()
    else:
        risk_cluster = centroid_df.var(axis=1).idxmax()
    mapping[risk_cluster] = "Risk-Off"
    remaining.remove(risk_cluster)

    if activity_col is not None and inflation_col is not None:
        activity_scale = centroid_df[activity_col].std() or 1.0
        inflation_scale = centroid_df[inflation_col].std() or 1.0
        stag_score = (
            centroid_df[inflation_col] / inflation_scale
            - centroid_df[activity_col] / activity_scale
        )
        stag_cluster = stag_score.loc[list(remaining)].idxmax()
    elif inflation_col is not None:
        stag_cluster = centroid_df.loc[list(remaining), inflation_col].idxmax()
    else:
        stag_cluster = centroid_df.loc[list(remaining)].mean(axis=1).idxmin()
    mapping[stag_cluster] = "Stagflation"
    remaining.remove(stag_cluster)

    if inflation_col is not None:
        inflation_cluster = centroid_df.loc[list(remaining), inflation_col].idxmax()
    else:
        inflation_cluster = centroid_df.loc[list(remaining)].mean(axis=1).idxmax()
    mapping[inflation_cluster] = "Inflationary"
    remaining.remove(inflation_cluster)

    growth_cluster = next(iter(remaining))
    mapping[growth_cluster] = "Growth"
    return mapping


def fit_regime_model(feature_frame: pd.DataFrame) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_frame.values)

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_pca = int(np.searchsorted(cum_var, PCA_VARIANCE_THRESHOLD) + 1)

    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(
        n_components=NUM_REGIMES,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=20,
        max_iter=300,
    )
    gmm.fit(Z_train)

    centroid_scaled = pca.inverse_transform(gmm.means_)
    centroid_original = scaler.inverse_transform(centroid_scaled)
    centroid_df = pd.DataFrame(centroid_original, columns=feature_frame.columns)
    centroid_df.index = range(NUM_REGIMES)
    cluster_to_regime = _map_clusters_to_regimes(centroid_df)

    return {
        "scaler": scaler,
        "pca": pca,
        "gmm": gmm,
        "cluster_to_regime": cluster_to_regime,
        "n_pca_components": n_pca,
        "explained_variance": float(cum_var[n_pca - 1]),
    }


def detect_regimes(
    asset_train: pd.DataFrame,
    asset_test: pd.DataFrame,
    merged_macro_train: pd.DataFrame,
    merged_macro_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train_data, test_data, feature_cols, asset_returns = build_supervised_datasets(
        asset_train,
        asset_test,
        merged_macro_train,
        merged_macro_test,
        ASSETS,
    )

    feature_train = train_data[feature_cols]
    regime_model = fit_regime_model(feature_train)
    Z_train = regime_model["pca"].transform(regime_model["scaler"].transform(feature_train.values))
    Z_test = regime_model["pca"].transform(regime_model["scaler"].transform(test_data[feature_cols].values))
    train_clusters = regime_model["gmm"].predict(Z_train)
    test_clusters = regime_model["gmm"].predict(Z_test)
    train_probs = regime_model["gmm"].predict_proba(Z_train)
    test_probs = regime_model["gmm"].predict_proba(Z_test)
    cluster_to_regime = regime_model["cluster_to_regime"]

    full_index = train_data.index.append(test_data.index)
    full_clusters = np.concatenate([train_clusters, test_clusters])
    full_probs = np.vstack([train_probs, test_probs])
    regime_names = pd.Series(full_clusters, index=full_index).map(cluster_to_regime)

    prob_cols = [f"prob_cluster_{i}" for i in range(NUM_REGIMES)]
    probs_df = pd.DataFrame(full_probs, index=full_index, columns=prob_cols)
    probs_df = probs_df.rename(columns={f"prob_cluster_{k}": f"prob_{v}" for k, v in cluster_to_regime.items()})

    labels_df = pd.DataFrame(
        {
            "Date": full_index,
            "split": ["train"] * len(train_data) + ["test"] * len(test_data),
            "cluster": full_clusters,
            "regime": regime_names.values,
        }
    ).set_index("Date")
    labels_df = labels_df.join(probs_df)

    asset_returns_aligned = asset_returns.loc[full_index].copy()
    asset_returns_aligned["regime"] = regime_names.values
    asset_returns_aligned["split"] = ["train"] * len(train_data) + ["test"] * len(test_data)

    stats_rows = []
    train_only = asset_returns_aligned[asset_returns_aligned["split"] == "train"]
    for regime in REGIME_NAMES:
        regime_slice = train_only[train_only["regime"] == regime]
        for asset in ASSETS:
            series = regime_slice[asset].dropna()
            annualized_return = series.mean() * 12
            annualized_vol = series.std() * np.sqrt(12)
            sharpe = (annualized_return / annualized_vol) if annualized_vol and not np.isnan(annualized_vol) else np.nan
            stats_rows.append(
                {
                    "Regime": regime,
                    "Asset": asset,
                    "Count": int(series.shape[0]),
                    "Mean_Monthly_Log_Return": series.mean(),
                    "Std_Monthly_Log_Return": series.std(),
                    "Annualized_Return": annualized_return,
                    "Annualized_Volatility": annualized_vol,
                    "Sharpe_Ratio": sharpe,
                }
            )

    diagnostics = {
        "n_pca_components": regime_model["n_pca_components"],
        "explained_variance": regime_model["explained_variance"],
        "gmm_lower_bound": float(regime_model["gmm"].lower_bound_),
    }
    return labels_df.reset_index(), pd.DataFrame(stats_rows), diagnostics


def plot_regime_timeline(
    labels_df: pd.DataFrame,
    asset_train: pd.DataFrame,
    asset_test: pd.DataFrame,
    output_path,
) -> None:
    labels = labels_df.copy()
    labels["Date"] = pd.to_datetime(labels["Date"])
    labels = labels.set_index("Date").sort_index()

    combined_assets = combine_frames(asset_train, asset_test)
    monthly_prices = combined_assets[ASSETS].resample("ME").last().ffill()
    normalized = monthly_prices / monthly_prices.iloc[0] * 100

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.35, 1]},
    )

    for regime in REGIME_NAMES:
        mask = labels["regime"] == regime
        axes[0].fill_between(
            labels.index,
            0,
            1,
            where=mask,
            color=REGIME_COLORS[regime],
            alpha=0.9,
            transform=axes[0].get_xaxis_transform(),
        )

    axes[0].set_yticks([])
    axes[0].set_title("Detected Macroeconomic Regimes", fontweight="bold")
    handles = [Patch(color=REGIME_COLORS[regime], label=regime) for regime in REGIME_NAMES]
    axes[0].legend(handles=handles, loc="upper left", ncol=4, fontsize=9)

    for asset in ASSETS:
        axes[1].plot(normalized.index, normalized[asset], label=asset, linewidth=1.7, color=ASSET_COLORS[asset])

    axes[1].set_ylabel("Normalized Price (Base = 100)")
    axes[1].set_title("Asset Performance Across Regimes", fontweight="bold")
    axes[1].legend(loc="upper left", ncol=2, fontsize=9)
    axes[1].set_xlabel("Date")

    plt.suptitle("PCA + GMM Regime Detection Timeline", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
