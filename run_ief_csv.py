import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

START_DATE = '2005-01-01'
END_DATE   = '2024-12-31'

# Load Nifty50_USD from CSV (2005-2019)
csv_data = pd.read_csv('data_daily_combined_2005_2019.csv', parse_dates=['Date'])
csv_data = csv_data.set_index('Date')
csv_data.index = pd.to_datetime(csv_data.index).tz_localize(None)
nifty_csv = csv_data['Nifty50_USD']
print("Nifty50_USD loaded from CSV (2005-2019)")
print(f"  Rows       : {len(nifty_csv)}")
print(f"  Date range : {nifty_csv.index[0].date()} -> {nifty_csv.index[-1].date()}")

# Load IEF (USBond) from CSV - Date format DD/MM/YYYY
ief_raw = pd.read_csv('ief_daily_2005_2025.csv', parse_dates=['Date'], dayfirst=True)
ief_raw = ief_raw.set_index('Date')
ief_raw.index = pd.to_datetime(ief_raw.index).tz_localize(None)
ief_raw = ief_raw.sort_index()
ief_data = ief_raw['Adj_Close'].loc[START_DATE:END_DATE]
ief_data.name = 'USBond'
print(f"\nIEF (USBond) loaded from CSV")
print(f"  Rows       : {len(ief_data)}")
print(f"  Date range : {ief_data.index[0].date()} -> {ief_data.index[-1].date()}")
print(f"  Missing    : {ief_data.isna().sum()}")
print(f"  Price range: ${ief_data.min():.2f} -> ${ief_data.max():.2f}")

# Download Nifty50 + USDINR from YF (2020-2024)
def download(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close

print("\nDownloading Nifty50 + USDINR from Yahoo Finance (2020-2024)...")
nifty_yf = download('^NSEI', '2020-01-01', END_DATE)
usdinr_yf = download('USDINR=X', '2020-01-01', END_DATE)
nifty_yf_usd = (nifty_yf / usdinr_yf).rename('Nifty50_USD')
print(f"  Nifty50_USD (YF)  "
      f"{nifty_yf_usd.first_valid_index().date()} -> "
      f"{nifty_yf_usd.last_valid_index().date()}")

# Stitch Nifty50_USD
nifty_full = pd.concat([nifty_csv, nifty_yf_usd])
nifty_full = nifty_full.sort_index()
nifty_full = nifty_full[~nifty_full.index.duplicated(keep='first')]
nifty_full = nifty_full.loc[START_DATE:END_DATE]
print(f"\nNifty50_USD stitched")
print(f"  Rows       : {len(nifty_full)}")
print(f"  Date range : {nifty_full.index[0].date()} -> {nifty_full.index[-1].date()}")

# Download SP500, Gold from YF
print("\nDownloading SP500, Gold from Yahoo Finance (2005-2024)...")
yf_data = pd.DataFrame({
    'SP500' : download('^GSPC', START_DATE, END_DATE),
    'Gold'  : download('GLD',   START_DATE, END_DATE),
})
yf_data.index.name = 'Date'
for col in yf_data.columns:
    valid = yf_data[col].dropna()
    print(f"  {col:8s}  {valid.index[0].date()} -> {valid.index[-1].date()}")

# Final dataset
data = yf_data.copy()
data['Nifty50_USD'] = nifty_full
data['USBond'] = ief_data
data = data[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]
data.index.name = 'Date'
print(f"\nFinal dataset : {data.shape[0]} rows x {data.shape[1]} columns")
print(f"  Date range  : {data.index[0].date()} -> {data.index[-1].date()}")
print("\n" + "="*65)
print("  FIRST 5 ROWS")
print("="*65)
print(data.head().to_string())
print("\n" + "="*65)
print("  LAST 5 ROWS")
print("="*65)
print(data.tail().to_string())
print("\n" + "="*65)
print("  MISSING VALUES")
print("="*65)
print(data.isnull().sum())

# SECTION 2: LOG-LINEAR REGRESSION
ASSETS = ['Nifty50_USD', 'SP500', 'Gold', 'USBond']
COLORS = {
    'Nifty50_USD' : '#1f77b4',
    'SP500'       : '#ff7f0e',
    'Gold'        : '#FFD700',
    'USBond'      : '#2ca02c',
}

def regress(series, name):
    s = series.dropna()
    t0 = s.index[0]
    t = ((s.index - t0).days.values / 365.25).reshape(-1, 1)
    log_price = np.log(s.values)
    model = LinearRegression()
    model.fit(t, log_price)
    a = model.intercept_
    b = model.coef_[0]
    log_pred = model.predict(t)
    return {
        'series': s, 't': t, 'log_price': log_price, 'log_pred': log_pred,
        'a': a, 'b': b,
        'CAGR': np.exp(b) - 1,
        'R2': r2_score(log_price, log_pred),
        'std_dev': np.std(log_price),
        'variance': np.var(log_price),
    }

results = {asset: regress(data[asset], asset) for asset in ASSETS}

print("\n" + "="*75)
print("  REGRESSION RESULTS  -  log(Price) = a + b*t")
print("  USBond = IEF Adj_Close from CSV (2005-2024)")
print("="*75)
hdr = f"  {'Asset':<14} {'Intercept a':>13} {'Slope b':>10} {'CAGR':>8} {'R2':>8} {'Std Dev':>10} {'Variance':>10}"
print(hdr)
print("  " + "-"*73)
for name, r in results.items():
    print(f"  {name:<14} {r['a']:>13.6f} {r['b']:>10.6f} "
          f"{r['CAGR']*100:>7.2f}% {r['R2']:>8.6f} "
          f"{r['std_dev']:>10.6f} {r['variance']:>10.6f}")

for name, r in results.items():
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Model      :  log(Price) = {r['a']:.6f} + {r['b']:.6f} x t")
    print(f"  Intercept  :  a = {r['a']:.6f}  ->  e^a = ${np.exp(r['a']):.2f}")
    print(f"  Slope      :  b = {r['b']:.6f}  (annual log growth rate)")
    print(f"  CAGR       :  e^b - 1 = {r['CAGR']*100:.2f}% per year")
    print(f"  Std Dev    :  {r['std_dev']:.6f}  (of log prices)")
    print(f"  Variance   :  {r['variance']:.6f}  (of log prices)")
    print(f"  R2         :  {r['R2']:.6f}")
    print(f"  Start Price: ${r['series'].iloc[0]:.2f}")
    print(f"  End Price  : ${r['series'].iloc[-1]:.2f}")

# SECTION 3: COVARIANCE & CORRELATION
log_returns = np.log(data / data.shift(1)).dropna()
cov_matrix = log_returns.cov()
corr_matrix = log_returns.corr()

print("\n" + "="*65)
print("  COVARIANCE MATRIX  (daily log returns)")
print("="*65)
print(cov_matrix.round(8))
print("\n" + "="*65)
print("  CORRELATION MATRIX  (daily log returns)")
print("="*65)
print(corr_matrix.round(6))

# SECTION 4: PLOTS
# Plot 1: Regression lines
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, name in zip(axes, ASSETS):
    r = results[name]
    ax.scatter(r['series'].index, r['log_price'], s=0.5, color=COLORS[name], alpha=0.5, label='log(Price)')
    ax.plot(r['series'].index, r['log_pred'], color='red', linewidth=2,
            label=f"Fit: a={r['a']:.3f}, b={r['b']:.4f}\nCAGR={r['CAGR']*100:.2f}%  R2={r['R2']:.4f}")
    ax.set_title(f"{name}", fontweight='bold')
    ax.set_xlabel("Date"); ax.set_ylabel("log(Price)"); ax.legend(fontsize=8)
plt.suptitle("log(Price) vs Time - Regression Lines (2005-2024)\nUSBond = IEF Adj_Close from CSV",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_regression_ief_csv.png', dpi=150, bbox_inches='tight')
print("\nPlot 1 saved: plot1_regression_ief_csv.png")

# Plot 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Pearson r'})
ax.set_title("Correlation Matrix - Daily Log Returns (2005-2024)", fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_corr_ief_csv.png', dpi=150, bbox_inches='tight')
print("Plot 2 saved: plot2_corr_ief_csv.png")

# Plot 3: Covariance heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cov_matrix, annot=True, fmt='.2e', cmap='Blues',
            ax=ax, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Covariance'})
ax.set_title("Covariance Matrix - Daily Log Returns (2005-2024)", fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_cov_ief_csv.png', dpi=150, bbox_inches='tight')
print("Plot 3 saved: plot3_cov_ief_csv.png")
