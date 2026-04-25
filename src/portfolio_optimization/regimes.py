import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
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


def _map_clusters_to_regimes(centroid_df: pd.DataFrame) -> dict[int, str]:
    columns = list(centroid_df.columns)
    risk_col = _preferred_column(columns, ["India_VIX", "US_VIX", "VIX_Change", "US_VIX_Change"])
    inflation_col = _preferred_column(columns, ["CPI", "IN_CPI_YoY", "US_CPI_YoY"])
    activity_col = _preferred_column(columns, ["PMI", "US_CFNAI", "US_Industrial_Production"])

    remaining = set(centroid_df.index.tolist())
    mapping: dict[int, str] = {}

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


def fit_pca(feature_frame: pd.DataFrame) -> tuple[StandardScaler, PCA, np.ndarray, int, float]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_frame.values)

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(x_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_pca = int(np.searchsorted(cum_var, PCA_VARIANCE_THRESHOLD) + 1)

    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
    z_train = pca.fit_transform(x_scaled)
    return scaler, pca, z_train, n_pca, float(cum_var[n_pca - 1])


def _cluster_to_regime_mapping(
    z_train: np.ndarray,
    train_clusters: np.ndarray,
    feature_frame: pd.DataFrame,
    scaler: StandardScaler,
    pca: PCA,
) -> dict[int, str]:
    centroid_scaled = np.vstack(
        [z_train[train_clusters == k].mean(axis=0) for k in range(NUM_REGIMES)]
    )
    centroid_scaled = pca.inverse_transform(centroid_scaled)
    centroid_original = scaler.inverse_transform(centroid_scaled)
    centroid_df = pd.DataFrame(centroid_original, columns=feature_frame.columns)
    centroid_df.index = range(NUM_REGIMES)
    return _map_clusters_to_regimes(centroid_df)


def fit_gmm_regime_model(feature_frame: pd.DataFrame) -> dict:
    scaler, pca, z_train, n_pca, explained = fit_pca(feature_frame)
    gmm = GaussianMixture(
        n_components=NUM_REGIMES,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=20,
        max_iter=300,
    )
    gmm.fit(z_train)
    train_clusters = gmm.predict(z_train)
    cluster_to_regime = _cluster_to_regime_mapping(
        z_train, train_clusters, feature_frame, scaler, pca
    )
    return {
        "method": "GMM",
        "scaler": scaler,
        "pca": pca,
        "model": gmm,
        "cluster_to_regime": cluster_to_regime,
        "n_pca_components": n_pca,
        "explained_variance": explained,
        "train_embedding": z_train,
        "train_clusters": train_clusters,
    }


def fit_kmeans_regime_model(feature_frame: pd.DataFrame) -> dict:
    scaler, pca, z_train, n_pca, explained = fit_pca(feature_frame)
    kmeans = KMeans(
        n_clusters=NUM_REGIMES,
        random_state=RANDOM_STATE,
        n_init=20,
        max_iter=500,
    )
    kmeans.fit(z_train)
    train_clusters = kmeans.labels_
    cluster_to_regime = _cluster_to_regime_mapping(
        z_train, train_clusters, feature_frame, scaler, pca
    )
    return {
        "method": "KMeans",
        "scaler": scaler,
        "pca": pca,
        "model": kmeans,
        "cluster_to_regime": cluster_to_regime,
        "n_pca_components": n_pca,
        "explained_variance": explained,
        "train_embedding": z_train,
        "train_clusters": train_clusters,
    }


def _predict_clusters(model_info: dict, x_scaled: np.ndarray) -> np.ndarray:
    z = model_info["pca"].transform(x_scaled)
    return model_info["model"].predict(z)


def detect_regimes_with_method(
    asset_train: pd.DataFrame,
    asset_test: pd.DataFrame,
    merged_macro_train: pd.DataFrame,
    merged_macro_test: pd.DataFrame,
    method: str = "gmm",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    method = method.lower()
    train_data, test_data, feature_cols, asset_returns = build_supervised_datasets(
        asset_train,
        asset_test,
        merged_macro_train,
        merged_macro_test,
        ASSETS,
    )
    feature_train = train_data[feature_cols]

    if method == "kmeans":
        model_info = fit_kmeans_regime_model(feature_train)
    else:
        model_info = fit_gmm_regime_model(feature_train)

    scaler = model_info["scaler"]
    train_clusters = model_info["train_clusters"]
    x_test_scaled = scaler.transform(test_data[feature_cols].values)
    test_clusters = _predict_clusters(model_info, x_test_scaled)
    cluster_to_regime = model_info["cluster_to_regime"]

    full_index = train_data.index.append(test_data.index)
    full_clusters = np.concatenate([train_clusters, test_clusters])
    regime_series = pd.Series(full_clusters, index=full_index).map(cluster_to_regime)

    labels_df = pd.DataFrame(
        {
            "Date": full_index,
            "split": ["train"] * len(train_data) + ["test"] * len(test_data),
            "cluster": full_clusters,
            "regime": regime_series.values,
            "method": model_info["method"],
        }
    ).set_index("Date")

    asset_returns_aligned = asset_returns.loc[full_index].copy()
    asset_returns_aligned["regime"] = regime_series.values
    asset_returns_aligned["split"] = ["train"] * len(train_data) + ["test"] * len(test_data)

    stats_rows = []
    train_only = asset_returns_aligned[asset_returns_aligned["split"] == "train"]
    for regime in REGIME_NAMES:
        regime_slice = train_only[train_only["regime"] == regime]
        for asset in ASSETS:
            series = regime_slice[asset].dropna()
            annualized_return = series.mean() * 12
            annualized_vol = series.std() * np.sqrt(12)
            sharpe = (
                (annualized_return / annualized_vol)
                if annualized_vol and not np.isnan(annualized_vol)
                else np.nan
            )
            stats_rows.append(
                {
                    "Method": model_info["method"],
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

    try:
        silhouette = float(
            silhouette_score(model_info["train_embedding"], train_clusters)
        )
    except Exception:
        silhouette = float("nan")

    diagnostics = {
        "method": model_info["method"],
        "n_pca_components": model_info["n_pca_components"],
        "explained_variance": model_info["explained_variance"],
        "silhouette": silhouette,
    }
    if method == "gmm":
        diagnostics["gmm_lower_bound"] = float(model_info["model"].lower_bound_)
    else:
        diagnostics["kmeans_inertia"] = float(model_info["model"].inertia_)

    return labels_df.reset_index(), pd.DataFrame(stats_rows), diagnostics


def detect_regimes(
    asset_train: pd.DataFrame,
    asset_test: pd.DataFrame,
    merged_macro_train: pd.DataFrame,
    merged_macro_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    return detect_regimes_with_method(
        asset_train, asset_test, merged_macro_train, merged_macro_test, method="gmm"
    )


def compare_regime_methods(
    gmm_labels: pd.DataFrame,
    kmeans_labels: pd.DataFrame,
) -> pd.DataFrame:
    merged = gmm_labels.merge(
        kmeans_labels[["Date", "regime"]].rename(columns={"regime": "regime_kmeans"}),
        on="Date",
        how="inner",
    ).rename(columns={"regime": "regime_gmm"})

    total = len(merged)
    agreement = float((merged["regime_gmm"] == merged["regime_kmeans"]).mean())
    ari = float(
        adjusted_rand_score(
            merged["regime_gmm"].astype("category").cat.codes,
            merged["regime_kmeans"].astype("category").cat.codes,
        )
    )

    per_regime_rows = []
    for regime in REGIME_NAMES:
        gmm_count = int((merged["regime_gmm"] == regime).sum())
        kmeans_count = int((merged["regime_kmeans"] == regime).sum())
        both_count = int(
            ((merged["regime_gmm"] == regime) & (merged["regime_kmeans"] == regime)).sum()
        )
        per_regime_rows.append(
            {
                "Regime": regime,
                "GMM_Count": gmm_count,
                "KMeans_Count": kmeans_count,
                "Intersection_Count": both_count,
                "GMM_Share": gmm_count / total if total else 0.0,
                "KMeans_Share": kmeans_count / total if total else 0.0,
            }
        )

    summary = pd.DataFrame(per_regime_rows)
    summary.loc[len(summary)] = {
        "Regime": "__overall__",
        "GMM_Count": total,
        "KMeans_Count": total,
        "Intersection_Count": int(agreement * total),
        "GMM_Share": 1.0,
        "KMeans_Share": 1.0,
    }
    summary["Adjusted_Rand_Index"] = ari
    summary["Label_Agreement"] = agreement
    return summary


def plot_regime_method_comparison(
    gmm_labels: pd.DataFrame,
    kmeans_labels: pd.DataFrame,
    asset_train: pd.DataFrame,
    asset_test: pd.DataFrame,
    output_path,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [0.3, 0.3, 1]},
    )
    for ax, df, title in (
        (axes[0], gmm_labels, "GMM Regimes"),
        (axes[1], kmeans_labels, "K-Means Regimes"),
    ):
        df_local = df.copy()
        df_local["Date"] = pd.to_datetime(df_local["Date"])
        df_local = df_local.set_index("Date").sort_index()
        for regime in REGIME_NAMES:
            mask = df_local["regime"] == regime
            ax.fill_between(
                df_local.index,
                0,
                1,
                where=mask,
                color=REGIME_COLORS[regime],
                alpha=0.9,
                transform=ax.get_xaxis_transform(),
            )
        ax.set_yticks([])
        ax.set_title(title, fontweight="bold", fontsize=11)

    handles = [Patch(color=REGIME_COLORS[regime], label=regime) for regime in REGIME_NAMES]
    axes[0].legend(handles=handles, loc="upper left", ncol=4, fontsize=9)

    combined_assets = combine_frames(asset_train, asset_test)
    monthly_prices = combined_assets[ASSETS].resample("ME").last().ffill()
    normalized = monthly_prices / monthly_prices.iloc[0] * 100
    for asset in ASSETS:
        axes[2].plot(
            normalized.index,
            normalized[asset],
            label=asset,
            linewidth=1.6,
            color=ASSET_COLORS[asset],
        )
    axes[2].set_ylabel("Normalized Price (Base = 100)")
    axes[2].set_title("Asset Performance Across Regimes", fontweight="bold")
    axes[2].legend(loc="upper left", ncol=2, fontsize=9)
    axes[2].set_xlabel("Date")

    plt.suptitle("Regime Detection: GMM vs K-Means", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        axes[1].plot(
            normalized.index,
            normalized[asset],
            label=asset,
            linewidth=1.7,
            color=ASSET_COLORS[asset],
        )
    axes[1].set_ylabel("Normalized Price (Base = 100)")
    axes[1].set_title("Asset Performance Across Regimes", fontweight="bold")
    axes[1].legend(loc="upper left", ncol=2, fontsize=9)
    axes[1].set_xlabel("Date")

    plt.suptitle("PCA + GMM Regime Detection Timeline", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
