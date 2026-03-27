from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "mid_review"
TABLES_DIR = ARTIFACTS_DIR / "tables" / "mid_review"

ASSETS = ["Nifty50_USD", "SP500", "Gold", "USBond"]
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
RANDOM_STATE = 42
PCA_VARIANCE_THRESHOLD = 0.90
NUM_REGIMES = 4
RISK_AVERSION_GAMMA = 3.0

CANONICAL_FILES = {
    "asset_train": DATA_DIR / "data_daily_2005_2024.csv",
    "asset_test": DATA_DIR / "data_daily_2025_2026.csv",
    "macro_global_train": DATA_DIR / "macro_finaldata_2005_2024.csv",
    "macro_global_test": DATA_DIR / "macro_finaldata_2025_2026.csv",
    "macro_india_train": DATA_DIR / "macro_finaldata_india_14.csv",
    "macro_india_test": DATA_DIR / "macro_india_2025_2026.csv",
}

MODEL_ORDER = [
    "LinearRegression",
    "SVR",
    "RandomForest",
    "GradientBoosting",
]

MODEL_LABELS = {
    "LinearRegression": "Linear Regression",
    "SVR": "SVR",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
}

MODEL_COLORS = {
    "LinearRegression": "#1f77b4",
    "SVR": "#ff7f0e",
    "RandomForest": "#2ca02c",
    "GradientBoosting": "#d62728",
}

REGIME_NAMES = ["Growth", "Inflationary", "Risk-Off", "Stagflation"]
REGIME_COLORS = {
    "Growth": "#4CAF50",
    "Inflationary": "#FF9800",
    "Risk-Off": "#F44336",
    "Stagflation": "#9C27B0",
}

REGIME_BOUNDS = {
    "Growth": {
        "Nifty50_USD": (0.20, 0.50),
        "SP500": (0.20, 0.50),
        "Gold": (0.05, 0.20),
        "USBond": (0.05, 0.20),
    },
    "Inflationary": {
        "Nifty50_USD": (0.10, 0.30),
        "SP500": (0.05, 0.25),
        "Gold": (0.25, 0.50),
        "USBond": (0.05, 0.25),
    },
    "Risk-Off": {
        "Nifty50_USD": (0.00, 0.15),
        "SP500": (0.00, 0.15),
        "Gold": (0.20, 0.45),
        "USBond": (0.30, 0.65),
    },
    "Stagflation": {
        "Nifty50_USD": (0.00, 0.15),
        "SP500": (0.00, 0.15),
        "Gold": (0.30, 0.55),
        "USBond": (0.15, 0.45),
    },
}
