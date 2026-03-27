from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_optimization.config import FIGURES_DIR, TABLES_DIR
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

    labels_path = TABLES_DIR / "regime_labels.csv"
    stats_path = TABLES_DIR / "regime_asset_stats.csv"
    figure_path = FIGURES_DIR / "regime_timeline.png"

    save_dataframe(labels_df, labels_path, index=False)
    save_dataframe(stats_df, stats_path, index=False)
    plot_regime_timeline(labels_df, asset_train, asset_test, figure_path)

    print("Saved regime detection outputs:")
    print(f"  {labels_path}")
    print(f"  {stats_path}")
    print(f"  {figure_path}")
    print("Diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

