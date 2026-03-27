import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time as _time

START_DATE = '2005-01-01'
END_DATE   = '2019-12-31'

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA COLLECTION
# ════════════════════════════════════════════════════════════════════
csv_path = 'data_daily_combined_2005_2019.csv'
csv_data = pd.read_csv(csv_path, parse_dates=['Date'])
csv_data = csv_data.set_index('Date')
csv_data.index = pd.to_datetime(csv_data.index).tz_localize(None)
csv_data = csv_data.loc[START_DATE:END_DATE]
nifty_usd = csv_data['Nifty50_USD']
print("Nifty50_USD loaded from CSV")
print(f"  Rows       : {len(nifty_usd)}")
print(f"  Date range : {nifty_usd.index[0].date()} -> {nifty_usd.index[-1].date()}")
print(f"  Missing    : {nifty_usd.isna().sum()}")

YF_TICKERS = {
    'SP500'  : '^GSPC',
    'Gold'   : 'GLD',
    'USBond' : 'TLT',
}

def download(ticker, start, end):
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                close = df['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index).tz_localize(None)
                return close
        except:
            pass
        _time.sleep(2)
    return pd.Series(dtype=float)

print("\nDownloading from Yahoo Finance...")
yf_data = pd.DataFrame({
    name: download(ticker, START_DATE, END_DATE)
    for name, ticker in YF_TICKERS.items()
})
yf_data.index.name = 'Date'
for col in yf_data.columns:
    s = yf_data[col].dropna()
    print(f"  {col:8s}  rows={len(s)}  {s.index[0].date()} -> {s.index[-1].date()}")

data_daily = yf_data.copy()
data_daily['Nifty50_USD'] = nifty_usd
data_daily = data_daily[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]
data_daily.index.name = 'Date'

print("\n" + "="*65)
print("  DATASET OVERVIEW")
print("="*65)
print(f"  Shape      : {data_daily.shape[0]} rows x {data_daily.shape[1]} columns")
print(f"  Date range : {data_daily.index[0].date()} -> {data_daily.index[-1].date()}")
print("\n" + "="*65)
print("  FIRST 5 ROWS")
print("="*65)
print(data_daily.head().to_string())
print("\n" + "="*65)
print("  LAST 5 ROWS")
print("="*65)
print(data_daily.tail().to_string())
print("\n" + "="*65)
print("  DESCRIPTIVE STATISTICS")
print("="*65)
print(data_daily.describe().round(4))
print("\n" + "="*65)
print("  MISSING VALUES")
print("="*65)
print(data_daily.isnull().sum())

# ════════════════════════════════════════════════════════════════════
# SECTION 2 — LOG-LINEAR REGRESSION
# ════════════════════════════════════════════════════════════════════
ASSETS = ['Nifty50_USD', 'SP500', 'Gold', 'USBond']
COLORS = {
    'Nifty50_USD' : '#1f77b4',
    'SP500'       : '#ff7f0e',
    'Gold'        : '#FFD700',
    'USBond'      : '#2ca02c',
}

def log_linear_regression(series, name):
    s         = series.dropna()
    t0        = s.index[0]
    time_yrs  = ((s.index - t0).days.values / 365.25).reshape(-1, 1)
    log_price = np.log(s.values)
    model = LinearRegression()
    model.fit(time_yrs, log_price)
    a          = model.intercept_
    b          = model.coef_[0]
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
        'std_dev'     : np.std(log_price),
        'start_price' : s.iloc[0],
        'end_price'   : s.iloc[-1],
        'end_fitted'  : price_pred[-1],
        'years'       : time_yrs[-1][0],
    }

results = {asset: log_linear_regression(data_daily[asset], asset) for asset in ASSETS}

print("\n" + "="*80)
print("  LOG-LINEAR REGRESSION RESULTS - ALL ASSETS  (2005-2019)")
print("="*80)
header = f"  {'Asset':<14} {'CAGR':>8} {'Slope b':>12} {'Intercept a':>13} {'Std Dev':>10} {'RMSE':>10} {'R2':>8}"
print(header)
print("  " + "-"*78)
for name, r in results.items():
    print(f"  {name:<14} {r['CAGR']*100:>7.2f}%  {r['b']:>12.6f}  "
          f"{r['a']:>13.6f}  {r['std_dev']:>10.6f}  "
          f"{r['RMSE']:>10.6f}  {r['R2']:>8.6f}")

for name, r in results.items():
    print(f"\n{'='*58}")
    print(f"  {name}")
    print(f"{'='*58}")
    print(f"  Model      :  log(Price) = {r['a']:.6f} + {r['b']:.6f} x t")
    print(f"  Intercept  :  a = {r['a']:.6f}  ->  e^a = ${np.exp(r['a']):.2f}")
    print(f"  Slope      :  b = {r['b']:.6f}  (annual log growth rate)")
    print(f"  CAGR       :  e^b - 1 = {r['CAGR']*100:.2f}% per year")
    print(f"  Std Dev    :  {r['std_dev']:.6f}  (of log prices)")
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

# Plot 1: log(Price) vs Time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (name, r) in zip(axes, results.items()):
    ax.scatter(r['series'].index, r['log_price'], s=0.5, color=COLORS[name], alpha=0.5, label=f'log({name})')
    ax.plot(r['series'].index, r['log_pred'], color='red', linewidth=2,
            label=f"OLS fit (CAGR={r['CAGR']*100:.2f}%)")
    ax.set_title(f"{name} - log(Price) vs Time", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("log(Price)")
    ax.legend(fontsize=8)
plt.suptitle("log(Price) vs Time - OLS Regression (2005-2019)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_log_regression_2005_2019.png', dpi=150, bbox_inches='tight')
print("\nPlot 1 saved: plot1_log_regression_2005_2019.png")

# Plot 2: Actual vs Fitted
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (name, r) in zip(axes, results.items()):
    ax.plot(r['series'].index, r['series'].values, color=COLORS[name], linewidth=1, label='Actual Price')
    ax.plot(r['series'].index, r['price_pred'], color='red', linewidth=2, linestyle='--', label='Fitted: P0 * e^(bt)')
    ax.set_title(f"{name} - Actual vs Fitted", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8)
plt.suptitle("Actual vs Fitted Price - Original Scale (2005-2019)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_actual_fitted_2005_2019.png', dpi=150, bbox_inches='tight')
print("Plot 2 saved: plot2_actual_fitted_2005_2019.png")

# Plot 3: Residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for ax, (name, r) in zip(axes, results.items()):
    ax.bar(r['series'].index, r['residuals'], width=2, color=COLORS[name], alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"{name} - Residuals (log space)", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
plt.suptitle("Regression Residuals - log(Price) Space (2005-2019)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_residuals_2005_2019.png', dpi=150, bbox_inches='tight')
print("Plot 3 saved: plot3_residuals_2005_2019.png")

# Plot 4: CAGR comparison
fig, ax = plt.subplots(figsize=(8, 5))
names_list = list(results.keys())
cagrs = [results[n]['CAGR'] * 100 for n in names_list]
colors_list = [COLORS[n] for n in names_list]
bars = ax.bar(names_list, cagrs, color=colors_list, edgecolor='white', width=0.5)
ax.bar_label(bars, fmt='%.2f%%', padding=4, fontsize=11, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title("CAGR Comparison - All Assets (2005-2019)", fontsize=13, fontweight='bold')
ax.set_ylabel("CAGR (%)")
ax.set_ylim(0, max(cagrs) * 1.3)
plt.tight_layout()
plt.savefig('plot4_cagr_2005_2019.png', dpi=150, bbox_inches='tight')
print("Plot 4 saved: plot4_cagr_2005_2019.png")

# Plot 5: Normalised performance
fig, ax = plt.subplots(figsize=(14, 6))
for name, r in results.items():
    normalised = r['series'] / r['series'].iloc[0] * 100
    ax.plot(r['series'].index, normalised, color=COLORS[name], linewidth=1.5, label=name)
ax.axhline(100, color='black', linewidth=0.8, linestyle='--', label='Base = 100')
ax.set_title("Normalised Asset Performance (Jan 2005 = 100)", fontsize=13, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Normalised Price (Base = 100)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('plot5_normalised_2005_2019.png', dpi=150, bbox_inches='tight')
print("Plot 5 saved: plot5_normalised_2005_2019.png")
