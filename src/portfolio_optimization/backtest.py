"""Walk-forward monthly backtest and stress-period analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from .config import (
    ASSETS,
    ASSET_COLORS,
    DEFAULT_WEIGHT_BOUNDS,
    MV_RISK_AVERSION,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    STRESS_PERIODS,
    WALK_FORWARD,
)
from .features import monthly_asset_log_returns, monthly_engineered_macro_features
from .metrics import cumulative_wealth, drawdown_series, log_to_simple, summarize_strategy
from .models import build_model_specs
from .optimizer import equal_weight_portfolio, markowitz_weights, regime_aware_weights
from .regimes import fit_gmm_regime_model


def _predict_asset_returns(
    history_train: pd.DataFrame,
    row_features: pd.Series,
    feature_cols: list[str],
    assets: list[str],
    model_name: str,
) -> pd.Series:
    specs = build_model_specs()
    model_template, needs_scaling = specs[model_name]

    x_train = history_train[feature_cols].values
    x_test = row_features[feature_cols].values.reshape(1, -1)

    if needs_scaling:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    preds = {}
    for asset in assets:
        y_train = history_train[asset].values
        model = clone(model_template)
        model.fit(x_train, y_train)
        preds[asset] = float(model.predict(x_test)[0])
    return pd.Series(preds)


def _rolling_log_return_cov(
    monthly_returns: pd.DataFrame,
    end_date: pd.Timestamp,
    lookback: int,
    assets: list[str],
) -> pd.DataFrame:
    window = monthly_returns.loc[:end_date].iloc[-lookback:]
    if window.shape[0] < 6:
        window = monthly_returns.loc[:end_date]
    return window[assets].cov()


def walk_forward_backtest(
    asset_train: pd.DataFrame,
    asset_test: pd.DataFrame,
    merged_macro_train: pd.DataFrame,
    merged_macro_test: pd.DataFrame,
    model_name: str = "RandomForest",
    min_train_months: int | None = None,
    cov_lookback_months: int | None = None,
    risk_aversion: float = MV_RISK_AVERSION,
    assets: list[str] | None = None,
    regime_refit_every: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assets = assets or ASSETS
    min_train_months = min_train_months or WALK_FORWARD["min_train_months"]
    cov_lookback_months = cov_lookback_months or WALK_FORWARD["cov_lookback_months"]

    monthly_returns = monthly_asset_log_returns(asset_train, asset_test, assets)
    monthly_macro = monthly_engineered_macro_features(merged_macro_train, merged_macro_test)
    lagged_features = monthly_macro.shift(1)
    dataset = pd.concat([monthly_returns, lagged_features], axis=1).dropna()
    feature_cols = [col for col in lagged_features.columns if col in dataset.columns]

    returns_records = []
    weights_records = []
    regime_records = []
    prediction_records = []

    cached_regime_info: dict | None = None
    steps_since_refit = 10**9

    for t_idx in range(min_train_months, len(dataset)):
        date_t = dataset.index[t_idx]
        history = dataset.iloc[:t_idx]
        row_t = dataset.iloc[t_idx]

        if history.shape[0] < 12:
            continue

        ml_mu = _predict_asset_returns(history, row_t, feature_cols, assets, model_name)
        prediction_records.append({"Date": date_t, **ml_mu.to_dict()})

        historical_mu = history[assets].mean()
        cov = _rolling_log_return_cov(
            monthly_returns,
            date_t - pd.Timedelta(days=1),
            cov_lookback_months,
            assets,
        )

        if cached_regime_info is None or steps_since_refit >= regime_refit_every:
            history_feat = history[feature_cols]
            try:
                cached_regime_info = fit_gmm_regime_model(history_feat)
                steps_since_refit = 0
            except Exception:
                cached_regime_info = None
        if cached_regime_info is not None:
            scaler = cached_regime_info["scaler"]
            x_scaled = scaler.transform(row_t[feature_cols].values.reshape(1, -1))
            z = cached_regime_info["pca"].transform(x_scaled)
            cluster = int(cached_regime_info["model"].predict(z)[0])
            regime_t = cached_regime_info["cluster_to_regime"].get(cluster, "Growth")
        else:
            regime_t = "Growth"
        steps_since_refit += 1
        regime_records.append({"Date": date_t, "regime": regime_t})

        realised_simple = log_to_simple(row_t[assets])

        weights_for_t = {
            "EqualWeight": equal_weight_portfolio(assets),
            "ClassicalMarkowitz": markowitz_weights(
                historical_mu,
                cov,
                assets=assets,
                weight_bounds=DEFAULT_WEIGHT_BOUNDS,
                risk_aversion=risk_aversion,
            ),
            "MLMarkowitz": markowitz_weights(
                ml_mu,
                cov,
                assets=assets,
                weight_bounds=DEFAULT_WEIGHT_BOUNDS,
                risk_aversion=risk_aversion,
            ),
            "MLRegimeAware": regime_aware_weights(
                regime_t,
                ml_mu,
                cov,
                assets=assets,
                risk_aversion=risk_aversion,
            ),
        }

        ret_row = {"Date": date_t}
        wt_row = {"Date": date_t}
        for strat in STRATEGY_ORDER:
            weights = weights_for_t[strat].reindex(assets)
            ret_row[strat] = float((weights * realised_simple.reindex(assets)).sum())
            for asset in assets:
                wt_row[(strat, asset)] = float(weights.loc[asset])
        returns_records.append(ret_row)
        weights_records.append(wt_row)

    returns_df = pd.DataFrame(returns_records).set_index("Date").sort_index()
    weights_df = pd.DataFrame(weights_records).set_index("Date").sort_index()
    weights_df.columns = pd.MultiIndex.from_tuples(weights_df.columns, names=["strategy", "asset"])
    regimes_df = pd.DataFrame(regime_records).set_index("Date").sort_index()
    predictions_df = pd.DataFrame(prediction_records).set_index("Date").sort_index()
    return returns_df, weights_df, regimes_df, predictions_df


def summarise_strategies(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strat in returns_df.columns:
        summary = summarize_strategy(returns_df[strat].dropna())
        summary["Strategy"] = strat
        rows.append(summary)
    df = pd.DataFrame(rows)
    cols = ["Strategy"] + [c for c in df.columns if c != "Strategy"]
    return df[cols]


def stress_period_returns(
    returns_df: pd.DataFrame,
    stress_periods: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    stress_periods = stress_periods or STRESS_PERIODS
    rows = []
    for name, (start, end) in stress_periods.items():
        window = returns_df.loc[start:end]
        if window.empty:
            continue
        for strat in returns_df.columns:
            series = window[strat].dropna()
            if series.empty:
                continue
            wealth = cumulative_wealth(series)
            rows.append(
                {
                    "Stress_Period": name,
                    "Start": start,
                    "End": end,
                    "Strategy": strat,
                    "Months": int(series.shape[0]),
                    "Total_Return": float(wealth.iloc[-1] - 1.0),
                    "Annualized_Vol": float(series.std(ddof=1) * np.sqrt(12)),
                    "Max_Drawdown": float(drawdown_series(series).min()),
                }
            )
    return pd.DataFrame(rows)


def plot_wealth_curves(returns_df: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for strat in STRATEGY_ORDER:
        if strat not in returns_df.columns:
            continue
        wealth = cumulative_wealth(returns_df[strat].dropna())
        ax.plot(
            wealth.index,
            wealth.values,
            label=STRATEGY_LABELS.get(strat, strat),
            color=STRATEGY_COLORS.get(strat),
            linewidth=1.8,
        )
    ax.set_ylabel("Growth of 1 unit")
    ax.set_title("Walk-Forward Cumulative Wealth", fontweight="bold")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(returns_df: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for strat in STRATEGY_ORDER:
        if strat not in returns_df.columns:
            continue
        dd = drawdown_series(returns_df[strat].dropna())
        ax.plot(
            dd.index,
            dd.values * 100,
            label=STRATEGY_LABELS.get(strat, strat),
            color=STRATEGY_COLORS.get(strat),
            linewidth=1.6,
        )
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Walk-Forward Drawdown by Strategy", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weight_stacks(weights_df: pd.DataFrame, output_path) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in weights_df.columns.get_level_values(0).unique()]
    fig, axes = plt.subplots(len(strategies), 1, figsize=(12, 2.4 * len(strategies)), sharex=True)
    if len(strategies) == 1:
        axes = [axes]
    for ax, strat in zip(axes, strategies):
        sub = weights_df[strat]
        ax.stackplot(
            sub.index,
            [sub[a].values for a in ASSETS],
            labels=ASSETS,
            colors=[ASSET_COLORS[a] for a in ASSETS],
            alpha=0.9,
        )
        ax.set_ylabel(STRATEGY_LABELS.get(strat, strat))
        ax.set_ylim(0, 1)
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Date")
    plt.suptitle("Walk-Forward Portfolio Weights by Strategy", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stress_returns(stress_df: pd.DataFrame, output_path) -> None:
    if stress_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No overlap with stress periods", ha="center", va="center")
        ax.set_axis_off()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    pivot = stress_df.pivot(index="Stress_Period", columns="Strategy", values="Total_Return")
    pivot = pivot.reindex(columns=[s for s in STRATEGY_ORDER if s in pivot.columns])
    fig, ax = plt.subplots(figsize=(12, 0.7 * len(pivot) + 3))
    x = np.arange(len(pivot.index))
    width = 0.22
    for i, strat in enumerate(pivot.columns):
        ax.bar(
            x + i * width,
            pivot[strat].values * 100,
            width,
            label=STRATEGY_LABELS.get(strat, strat),
            color=STRATEGY_COLORS.get(strat),
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + (len(pivot.columns) - 1) * width / 2)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Strategy Performance Across Historical Stress Periods", fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
