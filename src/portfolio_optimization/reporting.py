import pandas as pd

from .config import ASSETS, STRATEGY_ORDER, TEST_END, TEST_START, TRAIN_END


def build_overview_markdown(
    fixed_metrics: pd.DataFrame,
    walk_forward_model_summary: pd.DataFrame,
    walk_forward_regimes: pd.DataFrame,
    backtest_summary: pd.DataFrame,
) -> str:
    fixed_best = fixed_metrics.loc[fixed_metrics.groupby("Asset")["Test_RMSE"].idxmin()][
        ["Asset", "Model", "Test_RMSE", "Dir_Accuracy"]
    ].sort_values("Asset")

    fixed_lines = [
        f"- `{row['Asset']}`: {row['Model']} | Test RMSE `{row['Test_RMSE']:.4f}` | Dir.Acc `{row['Dir_Accuracy']:.1%}`"
        for _, row in fixed_best.iterrows()
    ]
    walk_forward_lines = [
        f"- `{row['Asset']}`: {row['Most_Frequent_Selected_Model']} | WF RMSE `{row['WalkForward_RMSE']:.4f}` | WF Dir.Acc `{row['WalkForward_Dir_Accuracy']:.1%}`"
        for _, row in walk_forward_model_summary.sort_values("Asset").iterrows()
    ]
    regime_lines = [
        f"- `{regime}`: {count} month(s)"
        for regime, count in walk_forward_regimes["regime"].value_counts().items()
    ]
    backtest_lines = [
        f"- `{row['Strategy']}`: total return `{row['Total_Return']:.1%}`, ann. return `{row['Annualized_Return']:.1%}`, Sharpe `{row['Sharpe_Ratio']:.2f}`, Max DD `{row['Max_Drawdown']:.1%}`"
        for _, row in backtest_summary.set_index("Strategy").loc[STRATEGY_ORDER].reset_index().iterrows()
    ]

    best_strategy = backtest_summary.sort_values("Sharpe_Ratio", ascending=False).iloc[0]
    interpretation = (
        f"The strongest walk-forward benchmark on this test slice is `{best_strategy['Strategy']}` by Sharpe ratio."
    )

    return "\n".join(
        [
            "# Mid-Review Results Overview",
            "",
            "## Scope",
            f"- Train period: January 2005 to {pd.Timestamp(TRAIN_END).strftime('%B %Y')}",
            f"- Test period: {pd.Timestamp(TEST_START).strftime('%B %Y')} to {pd.Timestamp(TEST_END).strftime('%B %Y')}",
            f"- Asset universe: {', '.join(f'`{asset}`' for asset in ASSETS)}",
            "- Backtest methodology: expanding-window walk-forward with monthly rebalancing",
            "",
            "## Fixed-Split Supervised Highlights",
            *fixed_lines,
            "",
            "## Walk-Forward Prediction Highlights",
            *walk_forward_lines,
            "",
            "## Walk-Forward Regime Sequence",
            *regime_lines,
            "",
            "## Walk-Forward Backtest Summary",
            *backtest_lines,
            "",
            "## Interpretation",
            interpretation,
            "",
        ]
    )
