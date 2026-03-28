# ML-Driven Asset Analysis and Macroeconomic Regime Detection

This repository contains the implementation for our EEE G513 course project.
The current asset universe is `Nifty50_USD`, `SP500`, `Gold`, and `USBond`, where `USBond` uses the IEF ETF proxy.
The main experiment trains on January 2005 to December 2024 and evaluates on January 2025 to February 2026.
For the midsem checkpoint, the retained scope is limited to asset-level exploratory analysis, supervised return prediction, and macroeconomic regime detection.

## Current Project Scope

- **Stage 1:** asset EDA summarizes long-run trend, CAGR, and cross-asset diversification using log-linear trend analysis plus return correlation and covariance.
- **Stage 2:** supervised models predict monthly log returns for all four assets using lagged macro features.
- **Stage 3:** a single cross-asset regime model uses merged India and global macro features with PCA plus GMM.

## Dataset Summary

- `data_daily_2005_2024.csv`: daily asset prices for the training period.
- `data_daily_2025_2026.csv`: daily asset prices for the held-out test period.
- `macro_finaldata_2005_2024.csv`: monthly global macro features for the training period.
- `macro_finaldata_2025_2026.csv`: monthly global macro features for the test period.
- `macro_finaldata_india_14.csv`: monthly India-focused macro features for the training period.
- `macro_india_2025_2026.csv`: monthly India-focused macro features for the test period.

## Reproduce the Pipeline

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the three retained stages from the repository root:

```bash
python3 scripts/run_asset_eda.py
python3 scripts/run_supervised_prediction.py
python3 scripts/run_regime_detection.py
```

## Expected Output Files

Tables under `artifacts/tables/mid_review`:

- `model_metrics_test_2025_2026.csv`
- `predictions_test_2025_2026.csv`
- `asset_trend_regression_summary.csv`
- `asset_return_correlation.csv`
- `asset_return_covariance.csv`
- `regime_labels.csv`
- `regime_asset_stats.csv`
- `final_overview.md`

Figures under `artifacts/figures/mid_review`:

- `supervised_model_comparison.png`
- `asset_trend_regression.png`
- `asset_cagr_comparison.png`
- `asset_correlation_heatmap.png`
- `regime_timeline.png`

## Key Outputs

- Asset trend and CAGR summary: `artifacts/tables/mid_review/asset_trend_regression_summary.csv`
- Asset return correlation: `artifacts/tables/mid_review/asset_return_correlation.csv`
- Supervised model metrics: `artifacts/tables/mid_review/model_metrics_test_2025_2026.csv`
- Regime labels and probabilities: `artifacts/tables/mid_review/regime_labels.csv`
- Consolidated summary note: `artifacts/tables/mid_review/final_overview.md`

## Known Limitations

- Stages 2 and 3 currently expose fixed-split summary outputs for the checkpoint.
- The log-linear trend regression is descriptive EDA for long-run trend/CAGR analysis; it is not used as the main predictive model in Stage 2.
- GMM is the canonical regime model for mid-review; K-Means remains future work.
- The asset universe is limited to the current 4-asset prototype and does not yet include Bitcoin.
- The India-specific training macro file uses the cleaner `macro_finaldata_india_14.csv` feature set for mid-review consistency.

## Repo Layout

- `src/portfolio_optimization`: reusable pipeline modules.
- `scripts`: canonical entrypoints.
- `data/processed`: canonical input datasets used by the pipeline.
- `artifacts`: generated tables and figures for review and presentation.
