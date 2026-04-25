from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "end_review"
TABLES_DIR = ARTIFACTS_DIR / "tables" / "end_review"

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

# Uniform per-asset weight bounds (long-only). Used by every Markowitz variant
# so all optimized strategies are compared under the same 5%-45% constraints.
DEFAULT_WEIGHT_BOUNDS = {
    "Nifty50_USD": (0.05, 0.45),
    "SP500": (0.05, 0.45),
    "Gold": (0.05, 0.45),
    "USBond": (0.05, 0.45),
}

# Keep regime-aware models on the same bounds as the other strategies.
REGIME_WEIGHT_BOUNDS = {
    regime: DEFAULT_WEIGHT_BOUNDS.copy()
    for regime in REGIME_NAMES
}

MV_RISK_AVERSION = 5.0

WALK_FORWARD = {
    "min_train_months": 60,
    "cov_lookback_months": 36,
}

STRESS_PERIODS = {
    "GFC_2008_2009": ("2008-09-01", "2009-06-30"),
    "EuroDebt_2011": ("2011-07-01", "2011-12-31"),
    "Taper_Tantrum_2013": ("2013-05-01", "2013-09-30"),
    "Oil_China_2015_2016": ("2015-08-01", "2016-02-29"),
    "Volmageddon_2018": ("2018-01-01", "2018-04-30"),
    "COVID_Crash_2020": ("2020-02-01", "2020-04-30"),
    "Russia_Inflation_2022": ("2022-01-01", "2022-10-31"),
}

STRATEGY_ORDER = [
    "EqualWeight",
    "ClassicalMarkowitz",
    "MLMarkowitz",
    "MLRegimeAware",
]

STRATEGY_LABELS = {
    "EqualWeight": "Equal-Weight",
    "ClassicalMarkowitz": "Classical Markowitz",
    "MLMarkowitz": "ML-Driven Markowitz",
    "MLRegimeAware": "ML + Regime-Aware",
}

STRATEGY_COLORS = {
    "EqualWeight": "#7f7f7f",
    "ClassicalMarkowitz": "#1f77b4",
    "MLMarkowitz": "#ff7f0e",
    "MLRegimeAware": "#d62728",
}

TABLE_OUTPUTS = {
    "model_metrics": TABLES_DIR / "model_metrics_test_2025_2026.csv",
    "predictions": TABLES_DIR / "predictions_test_2025_2026.csv",
    "asset_trend_summary": TABLES_DIR / "asset_trend_regression_summary.csv",
    "asset_return_correlation": TABLES_DIR / "asset_return_correlation.csv",
    "asset_return_covariance": TABLES_DIR / "asset_return_covariance.csv",
    "regime_labels": TABLES_DIR / "regime_labels.csv",
    "regime_asset_stats": TABLES_DIR / "regime_asset_stats.csv",
    "kmeans_labels": TABLES_DIR / "kmeans_regime_labels.csv",
    "regime_method_comparison": TABLES_DIR / "regime_method_comparison.csv",
    "walk_forward_returns": TABLES_DIR / "walk_forward_monthly_returns.csv",
    "walk_forward_weights": TABLES_DIR / "walk_forward_monthly_weights.csv",
    "walk_forward_summary": TABLES_DIR / "walk_forward_strategy_summary.csv",
    "stress_summary": TABLES_DIR / "stress_period_summary.csv",
    "overview": TABLES_DIR / "final_overview.md",
}

FIGURE_OUTPUTS = {
    "asset_trend_regression": FIGURES_DIR / "asset_trend_regression.png",
    "asset_cagr_comparison": FIGURES_DIR / "asset_cagr_comparison.png",
    "asset_correlation_heatmap": FIGURES_DIR / "asset_correlation_heatmap.png",
    "supervised_model_comparison": FIGURES_DIR / "supervised_model_comparison.png",
    "regime_timeline": FIGURES_DIR / "regime_timeline.png",
    "regime_method_comparison": FIGURES_DIR / "regime_method_comparison.png",
    "walk_forward_wealth": FIGURES_DIR / "walk_forward_wealth_curve.png",
    "walk_forward_drawdown": FIGURES_DIR / "walk_forward_drawdown.png",
    "walk_forward_weights": FIGURES_DIR / "walk_forward_weights_stacked.png",
    "stress_returns": FIGURES_DIR / "stress_period_returns.png",
}
