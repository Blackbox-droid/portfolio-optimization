from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "mid_review"
TABLES_DIR = ARTIFACTS_DIR / "tables" / "mid_review"

ASSETS = ["Nifty50_USD", "SP500", "Gold", "USBond"]
ASSET_COLORS = {
    "Nifty50_USD": "#1f77b4",
    "SP500": "#ff7f0e",
    "Gold": "#FFD700",
    "USBond": "#2ca02c",
}
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-02-28"
RANDOM_STATE = 42
PCA_VARIANCE_THRESHOLD = 0.90
NUM_REGIMES = 4

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

DATASET_COLUMNS = {
    "asset_prices": ["Date", *ASSETS],
    "macro_global": [
        "Date",
        "IN_CPI_YoY",
        "US_CPI_YoY",
        "Oil_Brent_USD",
        "Gold_USD",
        "US_Industrial_Production",
        "US_CFNAI",
        "US_Fed_Funds_Rate",
        "US_M2_YoY",
        "US_10Y_Yield",
        "US_2Y_Yield",
        "US_BAA_Yield",
        "US_AAA_Yield",
        "US_3M_TBill",
        "US_Unemployment_Rate",
        "US_NFP_MoM_Change",
        "USDINR",
        "IN_Forex_Reserves_USD",
        "DXY_Dollar_Index",
        "US_VIX",
        "US_Housing_Starts",
        "US_Retail_Sales_YoY",
        "US_Yield_Curve_Spread",
        "US_Credit_Spread",
        "USDINR_MoM_Return",
    ],
    "macro_india_train": [
        "Date",
        "CPI",
        "PMI",
        "Repo_Rate",
        "M2",
        "IN_CPI_YoY",
        "US_CPI_YoY",
        "Oil_Brent_USD",
        "USDINR",
        "USDINR_MoM_Return",
        "IN_Forex_Reserves_USD",
        "US_Fed_Funds_Rate",
        "US_10Y_Yield",
        "US_VIX",
        "DXY_Dollar_Index",
    ],
    "macro_india_test_required": [
        "Date",
        "CPI",
        "PMI",
        "Repo_Rate",
        "M2",
        "IN_CPI_YoY",
        "US_CPI_YoY",
        "Oil_Brent_USD",
        "USDINR",
        "USDINR_MoM_Return",
        "IN_Forex_Reserves_USD",
        "US_Fed_Funds_Rate",
        "US_10Y_Yield",
        "US_VIX",
        "DXY_Dollar_Index",
    ],
}

TABLE_OUTPUTS = {
    "model_metrics": TABLES_DIR / "model_metrics_test_2025_2026.csv",
    "predictions": TABLES_DIR / "predictions_test_2025_2026.csv",
    "asset_trend_summary": TABLES_DIR / "asset_trend_regression_summary.csv",
    "asset_return_correlation": TABLES_DIR / "asset_return_correlation.csv",
    "asset_return_covariance": TABLES_DIR / "asset_return_covariance.csv",
    "regime_labels": TABLES_DIR / "regime_labels.csv",
    "regime_asset_stats": TABLES_DIR / "regime_asset_stats.csv",
    "overview": TABLES_DIR / "final_overview.md",
}

FIGURE_OUTPUTS = {
    "asset_trend_regression": FIGURES_DIR / "asset_trend_regression.png",
    "asset_cagr_comparison": FIGURES_DIR / "asset_cagr_comparison.png",
    "asset_correlation_heatmap": FIGURES_DIR / "asset_correlation_heatmap.png",
    "supervised_model_comparison": FIGURES_DIR / "supervised_model_comparison.png",
    "regime_timeline": FIGURES_DIR / "regime_timeline.png",
}
