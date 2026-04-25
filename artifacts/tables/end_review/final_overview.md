# End-Review Results Overview

## Scope
- Full project pipeline: EDA -> supervised prediction -> regime detection (GMM + K-Means) -> Markowitz walk-forward backtest -> stress testing.
- Asset universe: Nifty50_USD, SP500, Gold, USBond. Bitcoin is intentionally out of scope.
- Strategies compared: EqualWeight, Classical Markowitz, ML-Driven Markowitz, ML + Regime-Aware Markowitz.

## Long-Run Asset Trend (2005-2024)
- `Nifty50_USD`: CAGR 6.71% | R2 0.794
- `SP500`: CAGR 8.73% | R2 0.881
- `Gold`: CAGR 5.77% | R2 0.681
- `USBond`: CAGR 4.17% | R2 0.735

## Supervised Return Prediction (fixed test split: 2025-2026)
- `Gold`: GradientBoosting | Test RMSE 0.0497 | Dir.Acc 85.7%
- `Nifty50_USD`: RandomForest | Test RMSE 0.0419 | Dir.Acc 42.9%
- `SP500`: RandomForest | Test RMSE 0.0388 | Dir.Acc 57.1%
- `USBond`: RandomForest | Test RMSE 0.0146 | Dir.Acc 50.0%

## Regime Detection: GMM vs K-Means
- Label agreement: 6.72%; Adjusted Rand Index: 0.600
- Growth: GMM 52.57% of months, K-Means 15.02%
- Inflationary: GMM 15.02% of months, K-Means 17.79%
- Risk-Off: GMM 4.74% of months, K-Means 29.64%
- Stagflation: GMM 27.67% of months, K-Means 37.55%

## Walk-Forward Backtest Summary
- EqualWeight: AnnRet 8.76% | AnnVol 9.54% | Sharpe 0.92 | MaxDD -14.87%
- ClassicalMarkowitz: AnnRet 7.67% | AnnVol 10.16% | Sharpe 0.75 | MaxDD -19.92%
- MLMarkowitz: AnnRet 9.49% | AnnVol 9.46% | Sharpe 1.00 | MaxDD -16.35%
- MLRegimeAware: AnnRet 9.49% | AnnVol 9.46% | Sharpe 1.00 | MaxDD -16.35%

## Stress Period Performance (Total Return)
| Stress Period | ClassicalMarkowitz | EqualWeight | MLMarkowitz | MLRegimeAware |
| --- | --- | --- | --- | --- |
| COVID_Crash_2020 | -6.65% | -4.13% | -5.39% | -5.39% |
| EuroDebt_2011 | 7.79% | -0.92% | 6.15% | 6.15% |
| Oil_China_2015_2016 | -0.14% | -3.06% | -6.58% | -6.58% |
| Russia_Inflation_2022 | -14.39% | -12.92% | -12.84% | -12.84% |
| Taper_Tantrum_2013 | -11.49% | -8.25% | -4.70% | -4.70% |
| Volmageddon_2018 | -0.65% | -1.81% | 0.22% | 0.22% |

## Interpretation
The final pipeline combines asset EDA, supervised return prediction, macro-regime detection, constrained Markowitz optimization, monthly walk-forward backtesting, and stress-period analysis. The ML-driven and regime-aware strategies are compared against Equal-Weight and Classical Markowitz baselines using Sharpe ratio, cumulative return, and maximum drawdown.
