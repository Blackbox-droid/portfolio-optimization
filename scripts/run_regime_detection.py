from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_optimization.config import FIGURE_OUTPUTS, TABLE_OUTPUTS
from portfolio_optimization.data_io import (
    ensure_output_dirs,
    load_asset_frames,
    load_global_macro_frames,
    load_india_macro_frames,
    save_dataframe,
)
from portfolio_optimization.regimes import detect_regimes, merge_macro_sources, plot_regime_timeline


def main() -> None:
    ensure_output_dirs()

    asset_train, asset_test = load_asset_frames()
    global_train, global_test = load_global_macro_frames()
    india_train, india_test = load_india_macro_frames()

    merged_train, merged_test = merge_macro_sources(
        india_train,
        india_test,
        global_train,
        global_test,
    )
    labels_df, stats_df, diagnostics = detect_regimes(
        asset_train,
        asset_test,
        merged_train,
        merged_test,
    )

    save_dataframe(labels_df, TABLE_OUTPUTS["regime_labels"], index=False)
    save_dataframe(stats_df, TABLE_OUTPUTS["regime_asset_stats"], index=False)
    plot_regime_timeline(labels_df, asset_train, asset_test, FIGURE_OUTPUTS["regime_timeline"])

    print("Saved regime detection outputs:")
    print(f"  {TABLE_OUTPUTS['regime_labels']}")
    print(f"  {TABLE_OUTPUTS['regime_asset_stats']}")
    print(f"  {FIGURE_OUTPUTS['regime_timeline']}")
    print("Diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
