"""Stage 2 - supervised return prediction."""
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
    save_dataframe,
)
from portfolio_optimization.features import build_supervised_datasets
from portfolio_optimization.models import evaluate_supervised_models, plot_supervised_model_comparison


def main() -> None:
    ensure_output_dirs()

    asset_train, asset_test = load_asset_frames()
    macro_train, macro_test = load_global_macro_frames()

    train_data, test_data, feature_cols, _ = build_supervised_datasets(
        asset_train, asset_test, macro_train, macro_test
    )
    metrics_df, predictions_df = evaluate_supervised_models(train_data, test_data, feature_cols)

    save_dataframe(metrics_df, TABLE_OUTPUTS["model_metrics"], index=False)
    save_dataframe(predictions_df, TABLE_OUTPUTS["predictions"], index=False)
    plot_supervised_model_comparison(metrics_df, FIGURE_OUTPUTS["supervised_model_comparison"])

    print("Stage 2 (supervised prediction) complete.")


if __name__ == "__main__":
    main()
