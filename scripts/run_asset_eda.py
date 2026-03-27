from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_optimization.config import FIGURE_OUTPUTS, TABLE_OUTPUTS
from portfolio_optimization.data_io import ensure_output_dirs, load_asset_frames, save_dataframe
from portfolio_optimization.eda import (
    build_asset_trend_summary,
    daily_return_matrices,
    plot_cagr_comparison,
    plot_correlation_heatmap,
    plot_log_trend_regression,
)


def main() -> None:
    ensure_output_dirs()

    asset_train, _ = load_asset_frames()

    trend_summary, trend_results = build_asset_trend_summary(asset_train)
    corr_df, cov_df = daily_return_matrices(asset_train)

    save_dataframe(trend_summary, TABLE_OUTPUTS["asset_trend_summary"], index=False)
    save_dataframe(corr_df, TABLE_OUTPUTS["asset_return_correlation"])
    save_dataframe(cov_df, TABLE_OUTPUTS["asset_return_covariance"])

    plot_log_trend_regression(trend_results, FIGURE_OUTPUTS["asset_trend_regression"])
    plot_cagr_comparison(trend_summary, FIGURE_OUTPUTS["asset_cagr_comparison"])
    plot_correlation_heatmap(corr_df, FIGURE_OUTPUTS["asset_correlation_heatmap"])

    print("Saved supplementary asset EDA outputs:")
    print(f"  {TABLE_OUTPUTS['asset_trend_summary']}")
    print(f"  {TABLE_OUTPUTS['asset_return_correlation']}")
    print(f"  {TABLE_OUTPUTS['asset_return_covariance']}")
    print(f"  {FIGURE_OUTPUTS['asset_trend_regression']}")
    print(f"  {FIGURE_OUTPUTS['asset_cagr_comparison']}")
    print(f"  {FIGURE_OUTPUTS['asset_correlation_heatmap']}")


if __name__ == "__main__":
    main()
