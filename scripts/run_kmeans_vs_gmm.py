"""Compare GMM and K-Means regime detection on the merged macro feature set."""
from pathlib import Path
import sys

import pandas as pd


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
from portfolio_optimization.regimes import (
    compare_regime_methods,
    detect_regimes_with_method,
    merge_macro_sources,
    plot_regime_method_comparison,
    plot_regime_timeline,
)


def main() -> None:
    ensure_output_dirs()

    asset_train, asset_test = load_asset_frames()
    global_train, global_test = load_global_macro_frames()
    india_train, india_test = load_india_macro_frames()

    merged_train, merged_test = merge_macro_sources(
        india_train, india_test, global_train, global_test
    )

    gmm_labels, gmm_stats, gmm_diag = detect_regimes_with_method(
        asset_train, asset_test, merged_train, merged_test, method="gmm"
    )
    kmeans_labels, kmeans_stats, kmeans_diag = detect_regimes_with_method(
        asset_train, asset_test, merged_train, merged_test, method="kmeans"
    )

    stats_combined = pd.concat([gmm_stats, kmeans_stats], ignore_index=True)
    comparison = compare_regime_methods(gmm_labels, kmeans_labels)

    save_dataframe(gmm_labels, TABLE_OUTPUTS["regime_labels"], index=False)
    save_dataframe(kmeans_labels, TABLE_OUTPUTS["kmeans_labels"], index=False)
    save_dataframe(stats_combined, TABLE_OUTPUTS["regime_asset_stats"], index=False)
    save_dataframe(comparison, TABLE_OUTPUTS["regime_method_comparison"], index=False)

    plot_regime_timeline(gmm_labels, asset_train, asset_test, FIGURE_OUTPUTS["regime_timeline"])
    plot_regime_method_comparison(
        gmm_labels,
        kmeans_labels,
        asset_train,
        asset_test,
        FIGURE_OUTPUTS["regime_method_comparison"],
    )

    print("Regime detection diagnostics:")
    print("  GMM:   ", gmm_diag)
    print("  KMeans:", kmeans_diag)


if __name__ == "__main__":
    main()
