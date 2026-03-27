from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_optimization.backtest import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_portfolio_weights,
    run_walk_forward_backtest,
)
from portfolio_optimization.config import ASSETS, FIGURES_DIR, TABLES_DIR
from portfolio_optimization.data_io import (
    ensure_output_dirs,
    load_asset_frames,
    load_global_macro_frames,
    load_india_macro_frames,
    save_dataframe,
)
from portfolio_optimization.features import build_supervised_datasets
from portfolio_optimization.regimes import merge_macro_sources


def build_overview_markdown(
    fixed_metrics: pd.DataFrame,
    walk_forward_model_summary: pd.DataFrame,
    walk_forward_regimes: pd.DataFrame,
    backtest_summary: pd.DataFrame,
) -> str:
    fixed_best = fixed_metrics.loc[fixed_metrics.groupby("Asset")["Test_RMSE"].idxmin()][
        ["Asset", "Model", "Test_RMSE", "Dir_Accuracy"]
    ].sort_values("Asset")

    wf_model_lines = []
    for _, row in walk_forward_model_summary.sort_values("Asset").iterrows():
        wf_model_lines.append(
            f"- `{row['Asset']}`: {row['Most_Frequent_Selected_Model']} | "
            f"WF RMSE `{row['WalkForward_RMSE']:.4f}` | "
            f"WF Dir.Acc `{row['WalkForward_Dir_Accuracy']:.1%}`"
        )

    fixed_lines = []
    for _, row in fixed_best.iterrows():
        fixed_lines.append(
            f"- `{row['Asset']}`: {row['Model']} | "
            f"Test RMSE `{row['Test_RMSE']:.4f}` | "
            f"Dir.Acc `{row['Dir_Accuracy']:.1%}`"
        )

    regime_counts = walk_forward_regimes["regime"].value_counts()
    regime_lines = [f"- `{regime}`: {count} month(s)" for regime, count in regime_counts.items()]

    backtest_lines = []
    for _, row in backtest_summary.iterrows():
        backtest_lines.append(
            f"- `{row['Strategy']}`: total return `{row['Total_Return']:.1%}`, "
            f"ann. return `{row['Annualized_Return']:.1%}`, "
            f"Sharpe `{row['Sharpe_Ratio']:.2f}`, "
            f"Max DD `{row['Max_Drawdown']:.1%}`"
        )

    best_strategy = backtest_summary.sort_values("Sharpe_Ratio", ascending=False).iloc[0]
    note = (
        f"The strongest walk-forward benchmark on this test slice is "
        f"`{best_strategy['Strategy']}` by Sharpe ratio, so the regime-aware strategy should still be presented as a "
        f"working prototype with preliminary results."
    )

    return "\n".join(
        [
            "# Mid-Review Results Overview",
            "",
            "## Scope",
            "- Train period: January 2005 to December 2024",
            "- Test period: January 2025 to February 2026",
            "- Asset universe: `Nifty50_USD`, `SP500`, `Gold`, `USBond`",
            "- Backtest methodology: expanding-window walk-forward with monthly rebalancing",
            "",
            "## Fixed-Split Supervised Highlights",
            *fixed_lines,
            "",
            "## Walk-Forward Prediction Highlights",
            *wf_model_lines,
            "",
            "## Walk-Forward Regime Sequence",
            *regime_lines,
            "",
            "## Walk-Forward Backtest Summary",
            *backtest_lines,
            "",
            "## Interpretation",
            note,
            "",
        ]
    )


def main() -> None:
    ensure_output_dirs()

    metrics_df = pd.read_csv(TABLES_DIR / "model_metrics_test_2025_2026.csv")

    asset_train, asset_test = load_asset_frames()
    global_train, global_test = load_global_macro_frames()
    india_train, india_test = load_india_macro_frames()

    train_data, test_data, feature_cols, _ = build_supervised_datasets(
        asset_train,
        asset_test,
        global_train,
        global_test,
        ASSETS,
    )
    supervised_full_data = pd.concat([train_data, test_data]).sort_index()

    merged_train, merged_test = merge_macro_sources(
        india_train,
        india_test,
        global_train,
        global_test,
    )
    regime_train_data, regime_test_data, regime_feature_cols, _ = build_supervised_datasets(
        asset_train,
        asset_test,
        merged_train,
        merged_test,
        ASSETS,
    )
    regime_full_data = pd.concat([regime_train_data, regime_test_data]).sort_index()

    (
        backtest_returns,
        summary_df,
        weights_df,
        walk_forward_regimes,
        walk_forward_model_summary,
    ) = run_walk_forward_backtest(
        supervised_full_data,
        feature_cols,
        regime_full_data,
        regime_feature_cols,
        list(test_data.index),
    )

    returns_path = TABLES_DIR / "backtest_returns.csv"
    summary_path = TABLES_DIR / "backtest_summary.csv"
    weights_path = TABLES_DIR / "portfolio_weights_test_2025_2026.csv"
    wf_regime_path = TABLES_DIR / "walk_forward_regimes_test_2025_2026.csv"
    wf_model_summary_path = TABLES_DIR / "walk_forward_model_summary.csv"
    overview_path = TABLES_DIR / "final_overview.md"
    cumulative_path = FIGURES_DIR / "cumulative_return_comparison.png"
    drawdown_path = FIGURES_DIR / "drawdown_comparison.png"
    weights_fig_path = FIGURES_DIR / "portfolio_weights.png"

    save_dataframe(backtest_returns, returns_path, index=False)
    save_dataframe(summary_df, summary_path, index=False)
    save_dataframe(weights_df, weights_path, index=False)
    save_dataframe(walk_forward_regimes, wf_regime_path, index=False)
    save_dataframe(walk_forward_model_summary, wf_model_summary_path, index=False)
    plot_cumulative_returns(backtest_returns, cumulative_path)
    plot_drawdowns(backtest_returns, drawdown_path)
    plot_portfolio_weights(weights_df, weights_fig_path)
    overview_path.write_text(
        build_overview_markdown(
            metrics_df,
            walk_forward_model_summary,
            walk_forward_regimes,
            summary_df,
        )
    )

    print("Saved walk-forward portfolio backtest outputs:")
    print(f"  {returns_path}")
    print(f"  {summary_path}")
    print(f"  {weights_path}")
    print(f"  {wf_regime_path}")
    print(f"  {wf_model_summary_path}")
    print(f"  {overview_path}")
    print(f"  {cumulative_path}")
    print(f"  {drawdown_path}")
    print(f"  {weights_fig_path}")


if __name__ == "__main__":
    main()
