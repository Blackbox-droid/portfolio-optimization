# ML-Driven Portfolio Optimization with Macroeconomic Regime Detection

This repository contains the mid-project review prototype for our EEE G513 course project.
The goal is to test whether macroeconomic features can improve portfolio allocation across four asset classes:
`Nifty50_USD`, `SP500`, `Gold`, and `USBond`.
`USBond` uses the IEF ETF proxy for US government bonds.
The current scope focuses on a reproducible fixed-split experiment:
train on January 2005 to December 2024 and evaluate on January 2025 to February 2026.
The pipeline includes supervised return prediction, PCA plus GMM regime detection, and regime-aware Markowitz optimization with benchmark backtests.

## Current Project Scope

- **Stage 1:** processed daily asset data and monthly macro features are stored under [`data/processed`](/Users/nikhilsheoran/Documents/Projects/portfolio-optimization/data/processed).
- **Stage 2:** supervised models predict monthly log returns for all four assets using lagged macro features.
- **Stage 3:** a single cross-asset regime model uses merged India and global macro features with PCA plus GMM.
- **Stage 4:** a regime-aware Markowitz portfolio is backtested with an expanding-window walk-forward setup against equal-weight and classical Markowitz baselines.

## Dataset Summary

- `data_daily_2005_2024.csv`: daily asset prices for the training period.
- `data_daily_2025_2026.csv`: daily asset prices for the held-out test period.
- `macro_finaldata_2005_2024.csv`: monthly global macro features for the training period.
- `macro_finaldata_2025_2026.csv`: monthly global macro features for the test period.
- `macro_finaldata_india_14.csv`: monthly India-focused macro features for the training period.
- `macro_india_2025_2026.csv`: monthly India-focused macro features for the test period.

## Reproduce the Mid-Review Pipeline

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the three canonical stages from the repository root:

```bash
python3 scripts/run_supervised_prediction.py
python3 scripts/run_regime_detection.py
python3 scripts/run_portfolio_backtest.py
```

## Expected Output Files

Tables under [`artifacts/tables/mid_review`](/Users/nikhilsheoran/Documents/Projects/portfolio-optimization/artifacts/tables/mid_review):

- `model_metrics_test_2025_2026.csv`
- `predictions_test_2025_2026.csv`
- `regime_labels.csv`
- `regime_asset_stats.csv`
- `backtest_returns.csv`
- `backtest_summary.csv`
- `portfolio_weights_test_2025_2026.csv`
- `walk_forward_regimes_test_2025_2026.csv`
- `walk_forward_model_summary.csv`
- `final_overview.md`

Figures under [`artifacts/figures/mid_review`](/Users/nikhilsheoran/Documents/Projects/portfolio-optimization/artifacts/figures/mid_review):

- `supervised_model_comparison.png`
- `regime_timeline.png`
- `cumulative_return_comparison.png`
- `drawdown_comparison.png`
- `portfolio_weights.png`

## Mid-Review Highlights

- Best supervised models can be found in [`model_metrics_test_2025_2026.csv`](/Users/nikhilsheoran/Documents/Projects/portfolio-optimization/artifacts/tables/mid_review/model_metrics_test_2025_2026.csv).
- The detected regimes and probabilities are saved in [`regime_labels.csv`](/Users/nikhilsheoran/Documents/Projects/portfolio-optimization/artifacts/tables/mid_review/regime_labels.csv).
- The main portfolio comparison for review is [`cumulative_return_comparison.png`](/Users/nikhilsheoran/Documents/Projects/portfolio-optimization/artifacts/figures/mid_review/cumulative_return_comparison.png).

## Known Limitations

- Stages 2 and 3 still expose fixed-split summary outputs, while Stage 4 now uses an expanding-window walk-forward backtest.
- GMM is the canonical regime model for mid-review; K-Means remains future work.
- The asset universe is limited to the current 4-asset prototype and does not yet include Bitcoin.
- The India-specific training macro file uses the cleaner `macro_finaldata_india_14.csv` feature set for mid-review consistency.
- The current 2025-2026 held-out results do not yet outperform the classical Markowitz baseline, so the portfolio stage should be presented as a working prototype with preliminary findings rather than a finalized winning strategy.

## Repo Layout

- [`src/portfolio_optimization`](/Users/nikhilsheoran/Documents/Projects/portfolio-optimization/src/portfolio_optimization): reusable pipeline modules.
- [`scripts`](/Users/nikhilsheoran/Documents/Projects/portfolio-optimization/scripts): canonical entrypoints.
- [`archive/experiments`](/Users/nikhilsheoran/Documents/Projects/portfolio-optimization/archive/experiments): archived experimental code, plots, notebooks, and legacy outputs.
