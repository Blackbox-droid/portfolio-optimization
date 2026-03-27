import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

START_DATE = '2005-01-01'
END_DATE   = '2024-12-31'

# Step 1: Load Nifty50_USD from CSV (2005-2019)
csv_path = 'data_daily_combined_2005_2019.csv'
csv_data = pd.read_csv(csv_path, parse_dates=['Date'])
csv_data = csv_data.set_index('Date')
csv_data.index = pd.to_datetime(csv_data.index).tz_localize(None)
csv_data = csv_data.sort_index()
nifty_csv = csv_data['Nifty50_USD']
print("Nifty50_USD loaded from CSV (2005-2019)")
print(f"  Rows       : {len(nifty_csv)}")
print(f"  Date range : {nifty_csv.index[0].date()} -> {nifty_csv.index[-1].date()}")

# Step 2: Download Nifty50 + USDINR from YF (2020-2024)
def download(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close

print("\nDownloading Nifty50 + USDINR (2020-2024)...")
nifty_yf  = download('^NSEI',    '2020-01-01', END_DATE)
usdinr_yf = download('USDINR=X', '2020-01-01', END_DATE)
nifty_yf_usd = (nifty_yf / usdinr_yf).rename('Nifty50_USD')

# Step 3: Stitch Nifty50_USD
nifty_full = pd.concat([nifty_csv, nifty_yf_usd])
nifty_full = nifty_full.sort_index()
nifty_full = nifty_full[~nifty_full.index.duplicated(keep='first')]
nifty_full = nifty_full.loc[START_DATE:END_DATE]

# Step 4: Download SP500, Gold, IEF (2005-2024)
YF_TICKERS = {
    'SP500'  : '^GSPC',
    'Gold'   : 'GLD',
    'USBond' : 'IEF',
}
print("Downloading SP500, Gold, IEF (2005-2024)...")
yf_data = pd.DataFrame({
    name: download(ticker, START_DATE, END_DATE)
    for name, ticker in YF_TICKERS.items()
})
yf_data.index.name = 'Date'

# Step 5: Combine all
data_daily = yf_data.copy()
data_daily['Nifty50_USD'] = nifty_full
data_daily = data_daily[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]
data_daily.index.name = 'Date'
print(f"\nDataset ready : {data_daily.shape[0]} rows x {data_daily.shape[1]} columns")
print(f"  Date range  : {data_daily.index[0].date()} -> {data_daily.index[-1].date()}")
print("\nMissing values:")
print(data_daily.isnull().sum())

# SECTION 2: LOG-LINEAR REGRESSION (ordinal date methodology)
ASSETS = ['Nifty50_USD', 'SP500', 'Gold', 'USBond']
COLORS = {
    'Nifty50_USD' : '#1f77b4',
    'SP500'       : '#ff7f0e',
    'Gold'        : '#FFD700',
    'USBond'      : '#2ca02c',
}

def log_linear_regression_ordinal(series, name):
    s = series.dropna()
    X = s.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = np.log(s.values)
    model = LinearRegression()
    model.fit(X, y)
    a = model.intercept_
    b = model.coef_[0]
    y_pred = model.predict(X)
    p_pred = np.exp(y_pred)
    resid = y - y_pred
    CAGR = np.exp(b * 365) - 1
    return {
        'name': name, 'series': s, 'X': X, 'y': y,
        'y_pred': y_pred, 'price_pred': p_pred, 'residuals': resid,
        'a': a, 'b': b, 'CAGR': CAGR,
        'std_dev': np.std(y),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'R2': r2_score(y, y_pred),
        'start_price': s.iloc[0], 'end_price': s.iloc[-1],
        'end_fitted': p_pred[-1],
    }

results = {asset: log_linear_regression_ordinal(data_daily[asset], asset) for asset in ASSETS}

print("\n" + "="*85)
print("  LOG-LINEAR REGRESSION RESULTS - ALL ASSETS  (2005-2024)")
print("  Method : y = ln(Close),  x = ordinal date,  CAGR = e^(b*365) - 1")
print("  USBond = IEF  (7-10 Year Treasury ETF)")
print("="*85)
hdr = f"  {'Asset':<14} {'CAGR':>8} {'Slope b':>16} {'Intercept a':>14} {'Std Dev':>10} {'RMSE':>10} {'R2':>8}"
print(hdr)
print("  " + "-"*83)
for name, r in results.items():
    print(f"  {name:<14} {r['CAGR']*100:>7.2f}%  {r['b']:>16.10f}  "
          f"{r['a']:>14.6f}  {r['std_dev']:>10.6f}  "
          f"{r['RMSE']:>10.6f}  {r['R2']:>8.6f}")

for name, r in results.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Model      :  ln(Price) = {r['a']:.6f} + {r['b']:.10f} x ordinal_date")
    print(f"  Intercept  :  a = {r['a']:.6f}")
    print(f"  Slope      :  b = {r['b']:.10f}  (log growth per calendar day)")
    print(f"  CAGR       :  e^(b x 365) - 1 = {r['CAGR']*100:.2f}% per year")
    print(f"  Std Dev    :  {r['std_dev']:.6f}  (of ln prices)")
    print(f"  RMSE       :  {r['RMSE']:.6f}  (log space)")
    print(f"  R2         :  {r['R2']:.6f}")
    print(f"  Period     :  {r['series'].index[0].date()} -> {r['series'].index[-1].date()}")
    print(f"  Start Price (actual) : ${r['start_price']:.2f}")
    print(f"  End Price   (actual) : ${r['end_price']:.2f}")
    print(f"  End Price   (fitted) : ${r['end_fitted']:.2f}")

# SECTION 3: PLOTS
# Plot 1: ln(Price) vs Date
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (name, r) in zip(axes, results.items()):
    ax.scatter(r['series'].index, r['y'], s=0.5, color=COLORS[name], alpha=0.5, label=f'ln({name})')
    ax.plot(r['series'].index, r['y_pred'], color='red', linewidth=2,
            label=f"Regression line (CAGR={r['CAGR']*100:.2f}%)")
    ax.set_title(f"{name} - ln(Price) vs Date", fontweight='bold')
    ax.set_xlabel("Date"); ax.set_ylabel("ln(Price)"); ax.legend(fontsize=8)
plt.suptitle("Linear Regression: ln(Price) vs Date (2005-2024)\nx = ordinal date, y = ln(Close), USBond = IEF",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_log_ordinal_2005_2024.png', dpi=150, bbox_inches='tight')
print("\nPlot 1 saved: plot1_log_ordinal_2005_2024.png")

# Plot 2: Actual vs Fitted
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (name, r) in zip(axes, results.items()):
    ax.plot(r['series'].index, r['series'].values, color=COLORS[name], linewidth=1, label='Actual Price')
    ax.plot(r['series'].index, r['price_pred'], color='red', linewidth=2, linestyle='--', label='Fitted: e^(a+bx)')
    ax.set_title(f"{name} - Actual vs Fitted", fontweight='bold')
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)"); ax.legend(fontsize=8)
plt.suptitle("Actual vs Fitted Price - Original Scale (2005-2024)\nUSBond = IEF (7-10 Year Treasury ETF)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_fitted_ordinal_2005_2024.png', dpi=150, bbox_inches='tight')
print("Plot 2 saved: plot2_fitted_ordinal_2005_2024.png")

# Plot 3: Residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for ax, (name, r) in zip(axes, results.items()):
    ax.bar(r['series'].index, r['residuals'], width=2, color=COLORS[name], alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"{name} - Residuals (log space)", fontweight='bold')
    ax.set_xlabel("Date"); ax.set_ylabel("Residual")
plt.suptitle("Regression Residuals - ln(Price) Space (2005-2024)\nUSBond = IEF (7-10 Year Treasury ETF)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_resid_ordinal_2005_2024.png', dpi=150, bbox_inches='tight')
print("Plot 3 saved: plot3_resid_ordinal_2005_2024.png")

# Plot 4: CAGR comparison
fig, ax = plt.subplots(figsize=(8, 5))
names_list = list(results.keys())
cagrs = [results[n]['CAGR'] * 100 for n in names_list]
colors_list = [COLORS[n] for n in names_list]
bars = ax.bar(names_list, cagrs, color=colors_list, edgecolor='white', width=0.5)
ax.bar_label(bars, fmt='%.2f%%', padding=4, fontsize=11, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title("CAGR Comparison - All Assets (2005-2024)\nUSBond = IEF (7-10 Year Treasury ETF)",
             fontsize=13, fontweight='bold')
ax.set_ylabel("CAGR (%)")
ax.set_ylim(0, max(cagrs) * 1.3)
plt.tight_layout()
plt.savefig('plot4_cagr_ordinal_2005_2024.png', dpi=150, bbox_inches='tight')
print("Plot 4 saved: plot4_cagr_ordinal_2005_2024.png")

# Plot 5: Normalised performance
fig, ax = plt.subplots(figsize=(14, 6))
for name, r in results.items():
    normalised = r['series'] / r['series'].iloc[0] * 100
    ax.plot(r['series'].index, normalised, color=COLORS[name], linewidth=1.5, label=name)
ax.axhline(100, color='black', linewidth=0.8, linestyle='--', label='Base = 100')
ax.set_title("Normalised Asset Performance (Jan 2005 = 100)\nUSBond = IEF (7-10 Year Treasury ETF)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Date"); ax.set_ylabel("Normalised Price (Base = 100)"); ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('plot5_norm_ordinal_2005_2024.png', dpi=150, bbox_inches='tight')
print("Plot 5 saved: plot5_norm_ordinal_2005_2024.png")
