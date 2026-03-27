# Mid-Review Results Overview

## Scope
- Train period: January 2005 to December 2024
- Test period: January 2025 to February 2026
- Asset universe: `Nifty50_USD`, `SP500`, `Gold`, `USBond`
- Backtest methodology: expanding-window walk-forward with monthly rebalancing

## Fixed-Split Supervised Highlights
- `Gold`: GradientBoosting | Test RMSE `0.0497` | Dir.Acc `85.7%`
- `Nifty50_USD`: RandomForest | Test RMSE `0.0419` | Dir.Acc `42.9%`
- `SP500`: RandomForest | Test RMSE `0.0388` | Dir.Acc `57.1%`
- `USBond`: RandomForest | Test RMSE `0.0146` | Dir.Acc `50.0%`

## Walk-Forward Prediction Highlights
- `Gold`: RandomForest | WF RMSE `0.0454` | WF Dir.Acc `85.7%`
- `Nifty50_USD`: RandomForest | WF RMSE `0.0418` | WF Dir.Acc `35.7%`
- `SP500`: LinearRegression | WF RMSE `0.0482` | WF Dir.Acc `35.7%`
- `USBond`: RandomForest | WF RMSE `0.0140` | WF Dir.Acc `64.3%`

## Walk-Forward Regime Sequence
- `Growth`: 14 month(s)

## Walk-Forward Backtest Summary
- `RegimeAware_ML`: total return `21.4%`, ann. return `18.1%`, Sharpe `3.28`, Max DD `-2.9%`
- `EqualWeight`: total return `27.8%`, ann. return `23.4%`, Sharpe `4.69`, Max DD `-1.1%`
- `ClassicalMarkowitz`: total return `56.2%`, ann. return `46.6%`, Sharpe `5.50`, Max DD `-0.4%`

## Interpretation
The strongest walk-forward benchmark on this test slice is `ClassicalMarkowitz` by Sharpe ratio, so the regime-aware strategy should still be presented as a working prototype with preliminary results.
