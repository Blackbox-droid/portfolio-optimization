from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_optimization.config import FIGURES_DIR, TABLES_DIR
from portfolio_optimization.data_io import ensure_output_dirs, load_asset_frames, load_global_macro_frames, save_dataframe
from portfolio_optimization.features import build_supervised_datasets
from portfolio_optimization.models import evaluate_supervised_models, plot_supervised_model_comparison


def main() -> None:
    ensure_output_dirs()

    asset_train, asset_test = load_asset_frames()
    macro_train, macro_test = load_global_macro_frames()

    train_data, test_data, feature_cols, _ = build_supervised_datasets(
        asset_train,
        asset_test,
        macro_train,
        macro_test,
    )
    metrics_df, predictions_df = evaluate_supervised_models(train_data, test_data, feature_cols)

    metrics_path = TABLES_DIR / "model_metrics_test_2025_2026.csv"
    predictions_path = TABLES_DIR / "predictions_test_2025_2026.csv"
    figure_path = FIGURES_DIR / "supervised_model_comparison.png"

    save_dataframe(metrics_df, metrics_path, index=False)
    save_dataframe(predictions_df, predictions_path, index=False)
    plot_supervised_model_comparison(metrics_df, figure_path)

    print("Saved supervised prediction outputs:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")
    print(f"  {figure_path}")


if __name__ == "__main__":
    main()

