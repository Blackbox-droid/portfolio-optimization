"""Build the final markdown summary from result tables."""
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_optimization.config import TABLE_OUTPUTS


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:.2f}%"


def _load_if_exists(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def main() -> None:
    trend = _load_if_exists(TABLE_OUTPUTS["asset_trend_summary"])
    metrics = _load_if_exists(TABLE_OUTPUTS["model_metrics"])
    walk = _load_if_exists(TABLE_OUTPUTS["walk_forward_summary"])
    stress = _load_if_exists(TABLE_OUTPUTS["stress_summary"])
    regime_comp = _load_if_exists(TABLE_OUTPUTS["regime_method_comparison"])

    lines: list[str] = []
    lines.append("# End-Review Results Overview")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Full project pipeline: EDA -> supervised prediction -> regime detection (GMM + K-Means) -> Markowitz walk-forward backtest -> stress testing.")
    lines.append("- Asset universe: Nifty50_USD, SP500, Gold, USBond. Bitcoin is intentionally out of scope.")
    lines.append("- Strategies compared: EqualWeight, Classical Markowitz, ML-Driven Markowitz, ML + Regime-Aware Markowitz.")
    lines.append("")

    if trend is not None:
        lines.append("## Long-Run Asset Trend (2005-2024)")
        sub = trend.set_index("Asset")[["CAGR", "R2"]]
        for asset, row in sub.iterrows():
            lines.append(f"- `{asset}`: CAGR {_fmt_pct(row['CAGR'])} | R2 {row['R2']:.3f}")
        lines.append("")

    if metrics is not None:
        lines.append("## Supervised Return Prediction (fixed test split: 2025-2026)")
        best_by_asset = metrics.sort_values(["Asset", "Test_RMSE"]).groupby("Asset").first().reset_index()
        for _, row in best_by_asset.iterrows():
            lines.append(
                f"- `{row['Asset']}`: {row['Model']} | Test RMSE {row['Test_RMSE']:.4f} | Dir.Acc {row['Dir_Accuracy'] * 100:.1f}%"
            )
        lines.append("")

    if regime_comp is not None:
        overall = regime_comp[regime_comp["Regime"] == "__overall__"]
        if not overall.empty:
            row = overall.iloc[0]
            lines.append("## Regime Detection: GMM vs K-Means")
            lines.append(
                f"- Label agreement: {_fmt_pct(row['Label_Agreement'])}; Adjusted Rand Index: {row['Adjusted_Rand_Index']:.3f}"
            )
            per = regime_comp[regime_comp["Regime"] != "__overall__"]
            for _, regime_row in per.iterrows():
                lines.append(
                    f"- {regime_row['Regime']}: GMM {_fmt_pct(regime_row['GMM_Share'])} of months, K-Means {_fmt_pct(regime_row['KMeans_Share'])}"
                )
            lines.append("")

    if walk is not None:
        lines.append("## Walk-Forward Backtest Summary")
        for _, row in walk.iterrows():
            lines.append(
                f"- {row['Strategy']}: AnnRet {_fmt_pct(row['Annualized_Return'])} | AnnVol {_fmt_pct(row['Annualized_Volatility'])} | Sharpe {row['Sharpe_Ratio']:.2f} | MaxDD {_fmt_pct(row['Max_Drawdown'])}"
            )
        lines.append("")

    if stress is not None and not stress.empty:
        lines.append("## Stress Period Performance (Total Return)")
        pivot = stress.pivot(index="Stress_Period", columns="Strategy", values="Total_Return")
        header = "| Stress Period | " + " | ".join(pivot.columns.tolist()) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(pivot.columns)) + " |"
        lines.append(header)
        lines.append(sep)
        for name, row in pivot.iterrows():
            vals = " | ".join(_fmt_pct(v) for v in row.tolist())
            lines.append(f"| {name} | {vals} |")
        lines.append("")

    lines.append("## Interpretation")
    lines.append(
        "The final pipeline combines asset EDA, supervised return prediction, macro-regime detection, constrained Markowitz optimization, monthly walk-forward backtesting, and stress-period analysis. The ML-driven and regime-aware strategies are compared against Equal-Weight and Classical Markowitz baselines using Sharpe ratio, cumulative return, and maximum drawdown."
    )

    TABLE_OUTPUTS["overview"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TABLE_OUTPUTS['overview']}")


if __name__ == "__main__":
    main()
