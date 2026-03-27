import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

START_DATE = '2008-01-01'
END_DATE   = '2022-12-31'
TICKERS = {
    'Nifty50' : '^NSEI',
    'SP500'   : '^GSPC',
    'Gold'    : 'GLD',
    'USBond'  : 'TLT',
    'VIX'     : '^VIX',
    'USDINR'  : 'USDINR=X',
}

import time

def download(ticker, start, end, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return pd.Series(dtype=float)
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize(None)
            return close
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  FAILED {ticker}: {e}")
                return pd.Series(dtype=float)

print("Downloading data...")
data = pd.DataFrame({
    name: download(ticker, START_DATE, END_DATE)
    for name, ticker in TICKERS.items()
})
data.index.name = 'Date'
data['Nifty50_USD'] = data['Nifty50'] / data['USDINR']
data = data.drop(columns=['Nifty50'])
data = data[['Nifty50_USD', 'SP500', 'Gold', 'USBond', 'VIX', 'USDINR']]
print(f"Data downloaded : {data.shape[0]} rows x {data.shape[1]} columns")
print(f"  Date range    : {data.index[0].date()} -> {data.index[-1].date()}")
print(f"  Columns       : {list(data.columns)}")

print("\n" + "="*60)
print("  FIRST 5 ROWS")
print("="*60)
print(data.head().to_string())
print("\n" + "="*60)
print("  DESCRIPTIVE STATISTICS")
print("="*60)
print(data.describe().round(4))
print("\n" + "="*60)
print("  MISSING VALUES")
print("="*60)
print(data.isnull().sum())

# ════════════════════════════════════════════════════════════════════
# SECTION 2 — LOG-LINEAR REGRESSION (CAGR)
# ════════════════════════════════════════════════════════════════════
ASSETS = ['Nifty50_USD', 'SP500', 'Gold', 'USBond']

def log_linear_regression(series, name):
    s         = series.dropna()
    t0        = s.index[0]
    time_yrs  = ((s.index - t0).days.values / 365.25).reshape(-1, 1)
    log_price = np.log(s.values)
    model = LinearRegression()
    model.fit(time_yrs, log_price)
    a = model.intercept_
    b = model.coef_[0]
    log_pred   = model.predict(time_yrs)
    price_pred = np.exp(log_pred)
    residuals  = log_price - log_pred
    return {
        'name'        : name,
        'series'      : s,
        'time_yrs'    : time_yrs,
        'log_price'   : log_price,
        'log_pred'    : log_pred,
        'price_pred'  : price_pred,
        'residuals'   : residuals,
        'a'           : a,
        'b'           : b,
        'CAGR'        : np.exp(b) - 1,
        'RMSE'        : np.sqrt(mean_squared_error(log_price, log_pred)),
        'R2'          : r2_score(log_price, log_pred),
        'start_price' : s.iloc[0],
        'end_price'   : s.iloc[-1],
        'end_fitted'  : price_pred[-1],
        'years'       : time_yrs[-1][0],
    }

results = {asset: log_linear_regression(data[asset], asset) for asset in ASSETS}

print("\n" + "="*75)
print("  LOG-LINEAR REGRESSION RESULTS - ALL ASSETS")
print("="*75)
header = f"  {'Asset':<14} {'CAGR':>8} {'Slope b':>10} {'Intercept a':>12} {'RMSE':>10} {'R2':>8}"
print(header)
print("  " + "-"*73)
for name, r in results.items():
    print(f"  {name:<14} {r['CAGR']*100:>7.2f}%  {r['b']:>10.6f}  "
          f"{r['a']:>12.4f}  {r['RMSE']:>10.6f}  {r['R2']:>8.6f}")

for name, r in results.items():
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Model      :  log(Price) = {r['a']:.4f} + {r['b']:.6f} x t")
    print(f"  CAGR       :  {r['CAGR']*100:.2f}% per year")
    print(f"  RMSE       :  {r['RMSE']:.6f}  (log space)")
    print(f"  R2         :  {r['R2']:.6f}")
    print(f"  Period     :  {r['series'].index[0].date()} -> {r['series'].index[-1].date()}")
    print(f"  Years      :  {r['years']:.2f}")
    print(f"  Start Price (actual) : ${r['start_price']:.2f}")
    print(f"  End Price   (actual) : ${r['end_price']:.2f}")
    print(f"  End Price   (fitted) : ${r['end_fitted']:.2f}")

# ════════════════════════════════════════════════════════════════════
# SECTION 3 — PLOTS
# ════════════════════════════════════════════════════════════════════
COLORS = {
    'Nifty50_USD' : '#1f77b4',
    'SP500'       : '#ff7f0e',
    'Gold'        : '#FFD700',
    'USBond'      : '#2ca02c',
}

# Plot 1: log(Price) vs Time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (name, r) in zip(axes, results.items()):
    color = COLORS[name]
    ax.scatter(r['series'].index, r['log_price'], s=0.8, color=color, alpha=0.6, label='log(Price) actual')
    ax.plot(r['series'].index, r['log_pred'], color='red', linewidth=2,
            label=f"OLS fit (CAGR = {r['CAGR']*100:.2f}%)")
    ax.set_title(f"{name}\nlog(Price) vs Time", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("log(Price)")
    ax.legend(fontsize=8)
plt.suptitle("Log-Linear Regression - log(Price) vs Time (2008-2022)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_log_price_regression.png', dpi=150, bbox_inches='tight')
print("\nPlot 1 saved: plot1_log_price_regression.png")

# Plot 2: Actual vs Fitted Price
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (name, r) in zip(axes, results.items()):
    color = COLORS[name]
    ax.plot(r['series'].index, r['series'].values, color=color, linewidth=1, label='Actual Price')
    ax.plot(r['series'].index, r['price_pred'], color='red', linewidth=2, linestyle='--',
            label='Fitted: P0 * e^(bt)')
    ax.set_title(f"{name}\nActual vs Fitted Price", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8)
plt.suptitle("Actual vs Fitted Price - Original Scale (2008-2022)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_actual_vs_fitted.png', dpi=150, bbox_inches='tight')
print("Plot 2 saved: plot2_actual_vs_fitted.png")

# Plot 3: Residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for ax, (name, r) in zip(axes, results.items()):
    color = COLORS[name]
    ax.bar(r['series'].index, r['residuals'], width=2, color=color, alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"{name} - Residuals (log space)", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
plt.suptitle("Regression Residuals - log(Price) Space (2008-2022)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_residuals.png', dpi=150, bbox_inches='tight')
print("Plot 3 saved: plot3_residuals.png")

# Plot 4: CAGR comparison
fig, ax = plt.subplots(figsize=(8, 5))
names_list = list(results.keys())
cagrs = [results[n]['CAGR'] * 100 for n in names_list]
colors_list = [COLORS[n] for n in names_list]
bars = ax.bar(names_list, cagrs, color=colors_list, edgecolor='white', width=0.5)
ax.bar_label(bars, fmt='%.2f%%', padding=4, fontsize=11, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title("CAGR Comparison - All Assets (2008-2022)", fontsize=13, fontweight='bold')
ax.set_ylabel("CAGR (%)")
ax.set_ylim(0, max(cagrs) * 1.25)
plt.tight_layout()
plt.savefig('plot4_cagr_comparison.png', dpi=150, bbox_inches='tight')
print("Plot 4 saved: plot4_cagr_comparison.png")
