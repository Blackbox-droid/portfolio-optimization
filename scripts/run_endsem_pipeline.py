"""Master entrypoint: run every stage of the end-review pipeline end-to-end."""
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

STAGES = [
    ("Stage 1 - Asset EDA", SCRIPTS / "run_asset_eda.py"),
    ("Stage 2 - Supervised Prediction", SCRIPTS / "run_supervised_prediction.py"),
    ("Stage 3 - Regime Detection (GMM + K-Means)", SCRIPTS / "run_kmeans_vs_gmm.py"),
    ("Stage 4 - Walk-Forward Backtest + Stress", SCRIPTS / "run_walk_forward_backtest.py"),
    ("Stage 5 - Final Overview", SCRIPTS / "build_final_overview.py"),
]


def main() -> None:
    for label, script in STAGES:
        print("\n" + "=" * 72)
        print(f"== {label}")
        print("=" * 72)
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name="__main__")
    print("\nAll stages completed.")


if __name__ == "__main__":
    main()
