import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from .config import ASSETS, ASSET_COLORS, REGIME_COLORS, REGIME_NAMES, STRATEGY_COLORS, STRATEGY_ORDER
from .metrics import cumulative_wealth, drawdown_series, log_to_simple, summarize_strategy
from .models import walk_forward_supervised_predictions
from .optimize import regime_bounds_for, solve_markowitz
from .regimes import walk_forward_regime_predictions


def _strategy_metric_column(strategy: str, suffix: str) -> str:
    return f"{strategy}_{suffix}"


def _wide_predictions(prediction_rows: pd.DataFrame) -> pd.DataFrame:
    wide = prediction_rows.pivot(index="Date", columns="Asset", values="Predicted_Log_Return").reindex(columns=ASSETS)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def summarize_walk_forward_predictions(prediction_rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for asset, group in prediction_rows.groupby("Asset"):
        actual = group["Actual_Log_Return"].values.astype(float)
        pred = group["Predicted_Log_Return"].values.astype(float)
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        mae = float(np.mean(np.abs(actual - pred)))
        actual_mean = float(np.mean(actual))
        ss_res = float(np.sum((actual - pred) ** 2))
        ss_tot = float(np.sum((actual - actual_mean) ** 2))
        r2 = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - (ss_res / ss_tot)
        dir_acc = float(np.mean(np.sign(actual) == np.sign(pred)))
        summary_rows.append(
            {
                "Asset": asset,
                "WalkForward_RMSE": rmse,
                "WalkForward_MAE": mae,
                "WalkForward_R2": r2,
                "WalkForward_Dir_Accuracy": dir_acc,
            }
        )
    return pd.DataFrame(summary_rows)


def run_walk_forward_backtest(
    supervised_full_data: pd.DataFrame,
    supervised_feature_cols: list[str],
    regime_full_data: pd.DataFrame,
    regime_feature_cols: list[str],
    test_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows, selection_df = walk_forward_supervised_predictions(
        supervised_full_data,
        supervised_feature_cols,
        test_dates,
        ASSETS,
    )
    regime_df = walk_forward_regime_predictions(
        regime_full_data,
        regime_feature_cols,
        test_dates,
    )

    expected_log_returns = _wide_predictions(prediction_rows)
    asset_log_returns = supervised_full_data[ASSETS].copy().sort_index()
    asset_simple_returns = log_to_simple(asset_log_returns)
    equal_weights = np.array([1.0 / len(ASSETS)] * len(ASSETS), dtype=float)

    strategy_rows = []
    weight_rows = []

    regime_lookup = regime_df.set_index("Date").sort_index()
    for date in expected_log_returns.index:
        history_simple = asset_simple_returns.loc[asset_simple_returns.index < date, ASSETS].copy()
        if history_simple.empty:
            continue

        mu_classical = history_simple.mean().reindex(ASSETS).values.astype(float)
        cov_hist = history_simple.cov().reindex(index=ASSETS, columns=ASSETS).values.astype(float)
        regime = regime_lookup.loc[date, "regime"]
        mu_ml = np.exp(expected_log_returns.loc[date].values.astype(float)) - 1.0
        realized_simple = asset_simple_returns.loc[date, ASSETS].values.astype(float)

        regime_aware_weights = solve_markowitz(mu_ml, cov_hist, bounds=regime_bounds_for(regime))
        classical_weights = solve_markowitz(
            mu_classical,
            cov_hist,
            bounds={asset: (0.0, 1.0) for asset in ASSETS},
        )

        strategy_rows.append(
            {
                "Date": date,
                "regime": regime,
                "RegimeAware_ML": float(np.dot(regime_aware_weights, realized_simple)),
                "EqualWeight": float(np.dot(equal_weights, realized_simple)),
                "ClassicalMarkowitz": float(np.dot(classical_weights, realized_simple)),
            }
        )
        weight_rows.append(
            {
                "Date": date,
                "regime": regime,
                **{asset: weight for asset, weight in zip(ASSETS, regime_aware_weights)},
            }
        )

    backtest_returns = pd.DataFrame(strategy_rows).set_index("Date").sort_index()
    weights_df = pd.DataFrame(weight_rows).set_index("Date").sort_index()

    for strategy in STRATEGY_ORDER:
        backtest_returns[f"{strategy}_CumulativeWealth"] = cumulative_wealth(backtest_returns[strategy])
        backtest_returns[f"{strategy}_Drawdown"] = drawdown_series(backtest_returns[strategy])

    summary_rows = []
    for strategy in STRATEGY_ORDER:
        metrics = summarize_strategy(backtest_returns[strategy])
        summary_rows.append({"Strategy": strategy, **metrics})

    summary_df = pd.DataFrame(summary_rows)
    walk_forward_model_summary = summarize_walk_forward_predictions(prediction_rows)
    return (
        backtest_returns.reset_index(),
        summary_df,
        weights_df.reset_index(),
        regime_df,
        walk_forward_model_summary.merge(
            selection_df.groupby("Asset")["Selected_Model"].agg(lambda s: s.value_counts().index[0]).reset_index().rename(
                columns={"Selected_Model": "Most_Frequent_Selected_Model"}
            ),
            on="Asset",
            how="left",
        ),
    )


def plot_cumulative_returns(backtest_returns: pd.DataFrame, output_path) -> None:
    df = backtest_returns.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in STRATEGY_ORDER:
        color = STRATEGY_COLORS[strategy]
        ax.plot(
            df["Date"],
            df[_strategy_metric_column(strategy, "CumulativeWealth")],
            linewidth=2,
            label=strategy,
            color=color,
        )

    ax.set_title("Walk-Forward Cumulative Wealth Comparison (Test: 2025-2026)", fontweight="bold")
    ax.set_ylabel("Wealth (Base = 1.0)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(backtest_returns: pd.DataFrame, output_path) -> None:
    df = backtest_returns.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in STRATEGY_ORDER:
        color = STRATEGY_COLORS[strategy]
        ax.plot(
            df["Date"],
            df[_strategy_metric_column(strategy, "Drawdown")],
            linewidth=2,
            label=strategy,
            color=color,
        )

    ax.set_title("Walk-Forward Drawdown Comparison (Test: 2025-2026)", fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_weights(weights_df: pd.DataFrame, output_path) -> None:
    df = weights_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.35, 1]},
    )

    for regime in REGIME_NAMES:
        mask = df["regime"] == regime
        axes[0].fill_between(
            df["Date"],
            0,
            1,
            where=mask,
            color=REGIME_COLORS[regime],
            alpha=0.9,
            transform=axes[0].get_xaxis_transform(),
        )
    axes[0].set_yticks([])
    axes[0].set_title("Regime Sequence During Test Period", fontweight="bold")
    axes[0].legend(
        handles=[Patch(color=REGIME_COLORS[name], label=name) for name in REGIME_NAMES],
        loc="upper left",
        ncol=4,
        fontsize=9,
    )

    axes[1].stackplot(
        df["Date"],
        [df[asset].values for asset in ASSETS],
        labels=ASSETS,
        colors=[ASSET_COLORS[asset] for asset in ASSETS],
        alpha=0.9,
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Portfolio Weight")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Regime-Aware Portfolio Weights", fontweight="bold")
    axes[1].legend(loc="upper left", ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
