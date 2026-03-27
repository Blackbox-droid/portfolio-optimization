import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .config import ASSETS, ASSET_COLORS


def fit_log_trend(series: pd.Series) -> dict[str, float | pd.Series | np.ndarray]:
    values = series.dropna().copy()
    time_years = ((values.index - values.index[0]).days.values / 365.25).reshape(-1, 1)
    log_price = np.log(values.values)

    model = LinearRegression().fit(time_years, log_price)
    log_pred = model.predict(time_years)
    latest_log_gap = float(log_price[-1] - log_pred[-1])

    return {
        "series": values,
        "log_price": log_price,
        "log_pred": log_pred,
        "price_pred": np.exp(log_pred),
        "Intercept_a": float(model.intercept_),
        "Slope_b": float(model.coef_[0]),
        "CAGR": float(np.exp(model.coef_[0]) - 1.0),
        "R2": float(r2_score(log_price, log_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(log_price, log_pred))),
        "StdDev": float(np.std(log_price)),
        "Variance": float(np.var(log_price)),
        "Latest_Trend_Gap_Log": latest_log_gap,
        "Latest_Trend_Gap_Pct": float(np.exp(latest_log_gap) - 1.0),
    }


def build_asset_trend_summary(asset_prices: pd.DataFrame, assets: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, dict]]:
    assets = assets or ASSETS
    trend_results = {asset: fit_log_trend(asset_prices[asset]) for asset in assets}

    summary_rows = []
    for asset in assets:
        result = trend_results[asset]
        summary_rows.append(
            {
                "Asset": asset,
                "Intercept_a": result["Intercept_a"],
                "Slope_b": result["Slope_b"],
                "CAGR": result["CAGR"],
                "R2": result["R2"],
                "RMSE": result["RMSE"],
                "StdDev": result["StdDev"],
                "Variance": result["Variance"],
                "Latest_Trend_Gap_Log": result["Latest_Trend_Gap_Log"],
                "Latest_Trend_Gap_Pct": result["Latest_Trend_Gap_Pct"],
            }
        )

    return pd.DataFrame(summary_rows), trend_results


def daily_return_matrices(asset_prices: pd.DataFrame, assets: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = assets or ASSETS
    log_returns = np.log(asset_prices[assets] / asset_prices[assets].shift(1)).dropna()
    return log_returns.corr(), log_returns.cov()


def plot_log_trend_regression(trend_results: dict[str, dict], output_path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, asset in zip(axes, ASSETS):
        result = trend_results[asset]
        ax.scatter(
            result["series"].index,
            result["log_price"],
            s=2,
            color=ASSET_COLORS[asset],
            alpha=0.45,
        )
        ax.plot(
            result["series"].index,
            result["log_pred"],
            color="#d62728",
            linewidth=2,
            label=f"CAGR={result['CAGR'] * 100:.2f}% | R2={result['R2']:.3f}",
        )
        ax.set_title(asset, fontweight="bold")
        ax.set_ylabel("log(Price)")
        ax.legend(fontsize=9)

    plt.suptitle("Log-Linear Trend Regression (2005-2024)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cagr_comparison(summary_df: pd.DataFrame, output_path) -> None:
    summary = summary_df.set_index("Asset").reindex(ASSETS).reset_index()
    cagrs = summary["CAGR"].values * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        summary["Asset"],
        cagrs,
        color=[ASSET_COLORS[asset] for asset in summary["Asset"]],
        edgecolor="white",
        width=0.55,
    )
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10, fontweight="bold")
    ax.set_title("Long-Run CAGR by Asset (2005-2024)", fontsize=13, fontweight="bold")
    ax.set_ylabel("CAGR (%)")
    ax.set_ylim(0, max(cagrs) * 1.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix: Daily Log Returns (2005-2024)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
