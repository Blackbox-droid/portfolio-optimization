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
from portfolio_optimization.config import ASSETS, FIGURE_OUTPUTS, TABLE_OUTPUTS
from portfolio_optimization.data_io import (
    ensure_output_dirs,
    load_asset_frames,
    load_global_macro_frames,
    load_india_macro_frames,
    save_dataframe,
)
from portfolio_optimization.features import build_supervised_datasets
from portfolio_optimization.regimes import merge_macro_sources
from portfolio_optimization.reporting import build_overview_markdown


def main() -> None:
    ensure_output_dirs()

    metrics_df = pd.read_csv(TABLE_OUTPUTS["model_metrics"])

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

    save_dataframe(backtest_returns, TABLE_OUTPUTS["backtest_returns"], index=False)
    save_dataframe(summary_df, TABLE_OUTPUTS["backtest_summary"], index=False)
    save_dataframe(weights_df, TABLE_OUTPUTS["portfolio_weights"], index=False)
    save_dataframe(walk_forward_regimes, TABLE_OUTPUTS["walk_forward_regimes"], index=False)
    save_dataframe(walk_forward_model_summary, TABLE_OUTPUTS["walk_forward_model_summary"], index=False)
    plot_cumulative_returns(backtest_returns, FIGURE_OUTPUTS["cumulative_return_comparison"])
    plot_drawdowns(backtest_returns, FIGURE_OUTPUTS["drawdown_comparison"])
    plot_portfolio_weights(weights_df, FIGURE_OUTPUTS["portfolio_weights"])
    TABLE_OUTPUTS["overview"].write_text(
        build_overview_markdown(
            metrics_df,
            walk_forward_model_summary,
            walk_forward_regimes,
            summary_df,
        ),
        encoding="utf-8",
    )

    print("Saved walk-forward portfolio backtest outputs:")
    print(f"  {TABLE_OUTPUTS['backtest_returns']}")
    print(f"  {TABLE_OUTPUTS['backtest_summary']}")
    print(f"  {TABLE_OUTPUTS['portfolio_weights']}")
    print(f"  {TABLE_OUTPUTS['walk_forward_regimes']}")
    print(f"  {TABLE_OUTPUTS['walk_forward_model_summary']}")
    print(f"  {TABLE_OUTPUTS['overview']}")
    print(f"  {FIGURE_OUTPUTS['cumulative_return_comparison']}")
    print(f"  {FIGURE_OUTPUTS['drawdown_comparison']}")
    print(f"  {FIGURE_OUTPUTS['portfolio_weights']}")


if __name__ == "__main__":
    main()
