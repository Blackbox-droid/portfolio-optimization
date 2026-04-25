# ML-Driven Portfolio Optimization

This repository contains the final version of our EEE G513 course project.
It includes:

- `optimizer.py` - constrained Markowitz mean-variance QP (`cvxpy`) with 5%-45% per-asset bounds.
- `regimes.py` - K-Means regime baseline alongside GMM, plus side-by-side comparison metrics.
- `backtest.py` - monthly walk-forward backtest across the full 2005-2026 sample, stress-period analysis, and plotting helpers.
- `scripts/run_walk_forward_backtest.py` - runs the 4-strategy walk-forward comparison.
- `scripts/run_kmeans_vs_gmm.py` - GMM vs K-Means regime comparison.
- `scripts/build_final_overview.py` - writes a consolidated markdown report.
- `scripts/run_endsem_pipeline.py` - master entrypoint that chains all stages.

Asset universe: `Nifty50_USD`, `SP500`, `Gold`, `USBond`. Bitcoin is intentionally not included in this project.

## Reproduce

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the full pipeline:

```bash
python3 scripts/run_endsem_pipeline.py
```

Or run individual stages:

```bash
python3 scripts/run_asset_eda.py
python3 scripts/run_supervised_prediction.py
python3 scripts/run_kmeans_vs_gmm.py
python3 scripts/run_walk_forward_backtest.py
python3 scripts/build_final_overview.py
```

## Expected Output Files

Under `artifacts/tables/end_review`:

- `asset_trend_regression_summary.csv`, `asset_return_correlation.csv`, `asset_return_covariance.csv`
- `model_metrics_test_2025_2026.csv`, `predictions_test_2025_2026.csv`
- `regime_labels.csv` (GMM), `kmeans_regime_labels.csv` (K-Means)
- `regime_asset_stats.csv`, `regime_method_comparison.csv`
- `walk_forward_monthly_returns.csv`, `walk_forward_monthly_weights.csv`, `walk_forward_strategy_summary.csv`
- `stress_period_summary.csv`
- `final_overview.md`

Under `artifacts/figures/end_review`:

- `asset_trend_regression.png`, `asset_cagr_comparison.png`, `asset_correlation_heatmap.png`
- `supervised_model_comparison.png`
- `regime_timeline.png`, `regime_method_comparison.png`
- `walk_forward_wealth_curve.png`, `walk_forward_drawdown.png`, `walk_forward_weights_stacked.png`
- `stress_period_returns.png`

## Strategy Definitions

| Strategy | Expected returns | Covariance | Weight bounds |
| --- | --- | --- | --- |
| EqualWeight | n/a | n/a | 1/N constant |
| ClassicalMarkowitz | rolling historical mean of monthly log returns | 36-month rolling covariance | default per-asset bounds |
| MLMarkowitz | monthly ML prediction (Random Forest by default) | 36-month rolling covariance | default per-asset bounds |
| MLRegimeAware | monthly ML prediction | 36-month rolling covariance | same 5%-45% per-asset bounds |

## Notes And Caveats

- Walk-forward warm-up is 60 months; the ML-driven backtest starts after the initial 5-year history window.
- Covariance is estimated from a 36-month rolling window of realized monthly log returns and projected onto the PSD cone for numerical stability.
- All optimized strategies use the same 5%-45% per-asset bounds for a consistent comparison.
- All reported portfolio results are computed from monthly log-return forecasts and converted to simple returns for portfolio accounting.
