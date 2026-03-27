import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

START_DATE = '2005-01-01'
END_DATE   = '2024-12-31'

# Load Nifty50_USD from CSV (2005-2019)
csv_data = pd.read_csv('data_daily_combined_2005_2019.csv', parse_dates=['Date'])
csv_data = csv_data.set_index('Date')
csv_data.index = pd.to_datetime(csv_data.index).tz_localize(None)
nifty_csv = csv_data['Nifty50_USD']

# Download Nifty50 + USDINR from YF (2020-2024)
def download(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close

nifty_yf  = download('^NSEI',    '2020-01-01', END_DATE)
usdinr_yf = download('USDINR=X', '2020-01-01', END_DATE)
nifty_yf_usd = (nifty_yf / usdinr_yf).rename('Nifty50_USD')

# Stitch Nifty50_USD
nifty_full = pd.concat([nifty_csv, nifty_yf_usd])
nifty_full = nifty_full.sort_index()
nifty_full = nifty_full[~nifty_full.index.duplicated(keep='first')]
nifty_full = nifty_full.loc[START_DATE:END_DATE]

# Download SP500, Gold, IEF
yf_data = pd.DataFrame({
    'SP500'  : download('^GSPC', START_DATE, END_DATE),
    'Gold'   : download('GLD',   START_DATE, END_DATE),
    'USBond' : download('IEF',   START_DATE, END_DATE),
})

# Final dataset
data = yf_data.copy()
data['Nifty50_USD'] = nifty_full
data = data[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]
data.index.name = 'Date'
print(f"Data ready: {data.shape[0]} rows  |  "
      f"{data.index[0].date()} -> {data.index[-1].date()}")

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
        'a': a, 'b': b, 'CAGR': np.exp(b) - 1,
        'R2': r2_score(log_price, log_pred),
        'std_dev': np.std(log_price), 'variance': np.var(log_price),
    }

results = {asset: regress(data[asset], asset) for asset in ASSETS}

print("\n" + "="*75)
print("  REGRESSION RESULTS  -  log(Price) = a + b*t")
print("="*75)
hdr = f"  {'Asset':<14} {'Intercept a':>13} {'Slope b':>10} {'CAGR':>8} {'R2':>8} {'Std Dev':>10} {'Variance':>10}"
print(hdr)
print("  " + "-"*73)
for name, r in results.items():
    print(f"  {name:<14} {r['a']:>13.6f} {r['b']:>10.6f} "
          f"{r['CAGR']*100:>7.2f}% {r['R2']:>8.6f} "
          f"{r['std_dev']:>10.6f} {r['variance']:>10.6f}")

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
plt.suptitle("log(Price) vs Time - Regression Lines (2005-2024)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_regression_lines.png', dpi=150, bbox_inches='tight')
print("\nPlot 1 saved: plot1_regression_lines.png")

# Plot 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Pearson r'})
ax.set_title("Correlation Matrix - Daily Log Returns (2005-2024)", fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("Plot 2 saved: plot2_correlation_heatmap.png")

# Plot 3: Covariance heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cov_matrix, annot=True, fmt='.2e', cmap='Blues',
            ax=ax, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Covariance'})
ax.set_title("Covariance Matrix - Daily Log Returns (2005-2024)", fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_covariance_heatmap.png', dpi=150, bbox_inches='tight')
print("Plot 3 saved: plot3_covariance_heatmap.png")
