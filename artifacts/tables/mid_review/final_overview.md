# Mid-Review Results Overview

## Scope
- Train period: January 2005 to December 2024
- Test period: January 2025 to February 2026
- Asset universe: `Nifty50_USD`, `SP500`, `Gold`, `USBond`
- Scope retained for checkpoint: asset EDA, supervised return prediction, and PCA + GMM regime detection

## Asset EDA Highlights
- `SP500` has the highest long-run estimated CAGR at `8.73%`, followed by `Nifty50_USD` at `6.71%`.
- `Gold` remains a moderate-growth diversifier with estimated CAGR `5.77%`.
- `USBond` is the lowest-growth asset in the current universe with estimated CAGR `4.17%`.

## Supervised Learning Highlights
- `Gold`: GradientBoosting | Test RMSE `0.0497` | Dir.Acc `85.7%`
- `Nifty50_USD`: RandomForest | Test RMSE `0.0419` | Dir.Acc `42.9%`
- `SP500`: RandomForest | Test RMSE `0.0388` | Dir.Acc `57.1%`
- `USBond`: RandomForest | Test RMSE `0.0146` | Dir.Acc `50.0%`

## PCA Regime Detection Highlights
- The regime model uses PCA on merged India and global macro features, followed by a 4-state GMM.
- PCA retains `13` components while explaining `90.4%` of the variance in the macro feature space.
- The detected regimes are `Growth`, `Inflationary`, `Risk-Off`, and `Stagflation`.
- Regime-conditioned asset summary statistics are provided in `regime_asset_stats.csv` for interpretation.

## Interpretation
This checkpoint focuses on demonstrating the data understanding, prediction, and macro-regime modeling pipeline without portfolio backtesting or comparisons against alternate portfolio construction methods.
