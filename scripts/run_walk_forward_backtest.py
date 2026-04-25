"""Walk-forward monthly backtest across the full sample with stress-period analysis."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_optimization.backtest import (
    plot_drawdowns,
    plot_stress_returns,
    plot_weight_stacks,
    plot_wealth_curves,
    stress_period_returns,
    summarise_strategies,
    walk_forward_backtest,
)
from portfolio_optimization.config import FIGURE_OUTPUTS, TABLE_OUTPUTS
from portfolio_optimization.data_io import (
    ensure_output_dirs,
    load_asset_frames,
    load_global_macro_frames,
    load_india_macro_frames,
    save_dataframe,
)
from portfolio_optimization.regimes import merge_macro_sources


def main() -> None:
    ensure_output_dirs()

    asset_train, asset_test = load_asset_frames()
    global_train, global_test = load_global_macro_frames()
    india_train, india_test = load_india_macro_frames()

    merged_train, merged_test = merge_macro_sources(
        india_train, india_test, global_train, global_test
    )

    returns_df, weights_df, regimes_df, _ = walk_forward_backtest(
        asset_train,
        asset_test,
        merged_train,
        merged_test,
        model_name="RandomForest",
    )

    save_dataframe(returns_df.join(regimes_df), TABLE_OUTPUTS["walk_forward_returns"])

    weights_flat = weights_df.copy()
    weights_flat.columns = [f"{s}__{a}" for s, a in weights_flat.columns]
    save_dataframe(weights_flat, TABLE_OUTPUTS["walk_forward_weights"])

    summary = summarise_strategies(returns_df)
    save_dataframe(summary, TABLE_OUTPUTS["walk_forward_summary"], index=False)

    stress = stress_period_returns(returns_df)
    save_dataframe(stress, TABLE_OUTPUTS["stress_summary"], index=False)

    plot_wealth_curves(returns_df, FIGURE_OUTPUTS["walk_forward_wealth"])
    plot_drawdowns(returns_df, FIGURE_OUTPUTS["walk_forward_drawdown"])
    plot_weight_stacks(weights_df, FIGURE_OUTPUTS["walk_forward_weights"])
    plot_stress_returns(stress, FIGURE_OUTPUTS["stress_returns"])

    print("Walk-forward backtest complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
