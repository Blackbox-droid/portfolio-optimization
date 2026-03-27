import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

START_DATE = '2008-01-01'
END_DATE   = '2025-12-31'

print("Downloading S&P500...")
for attempt in range(3):
    try:
        df = yf.download('^GSPC', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if not df.empty:
            break
    except:
        pass
    time.sleep(2)

close = df['Close']
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
close.index = pd.to_datetime(close.index).tz_localize(None)
close = close.dropna()
close.name = 'SP500'

print(f"Downloaded : {len(close)} rows")
print(f"  Date range : {close.index[0].date()} -> {close.index[-1].date()}")
print(f"  Start Price: ${close.iloc[0]:.2f}")
print(f"  End Price  : ${close.iloc[-1]:.2f}")

t0        = close.index[0]
time_yrs  = ((close.index - t0).days.values / 365.25).reshape(-1, 1)
log_price = np.log(close.values)

model = LinearRegression()
model.fit(time_yrs, log_price)
a = model.intercept_
b = model.coef_[0]

log_pred   = model.predict(time_yrs)
price_pred = np.exp(log_pred)
residuals  = log_price - log_pred

CAGR    = np.exp(b) - 1
RMSE    = np.sqrt(mean_squared_error(log_price, log_pred))
R2      = r2_score(log_price, log_pred)
std_dev = np.std(log_price)

print("\n" + "="*55)
print("  LOG-LINEAR REGRESSION - S&P500")
print("="*55)
print(f"  Model      :  log(Price) = a + b * t")
print(f"  Intercept  :  a = {a:.6f}  ->  e^a = ${np.exp(a):.2f}")
print(f"  Slope      :  b = {b:.6f}  (annual log growth rate)")
print(f"  CAGR       :  e^b - 1 = {CAGR*100:.2f}% per year")
print(f"  Std Dev    :  {std_dev:.6f}  (of log prices)")
print(f"  RMSE       :  {RMSE:.6f}  (log space)")
print(f"  R2         :  {R2:.6f}")
print(f"\n  Period     :  {close.index[0].date()} -> {close.index[-1].date()}")
print(f"  Years      :  {time_yrs[-1][0]:.2f}")
print(f"  Start Price (actual) : ${close.iloc[0]:.2f}")
print(f"  End Price   (actual) : ${close.iloc[-1]:.2f}")
print(f"  End Price   (fitted) : ${price_pred[-1]:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(close.index, log_price, s=0.8, color='#ff7f0e', alpha=0.6, label='log(SP500)')
ax.plot(close.index, log_pred, color='red', linewidth=2, label=f'OLS fit (CAGR={CAGR*100:.2f}%)')
ax.set_title("log(Price) vs Time", fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("log(Price)")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(close.index, close.values, color='#ff7f0e', linewidth=1, label='Actual Price')
ax.plot(close.index, price_pred, color='red', linewidth=2, linestyle='--', label='Fitted: P0 * e^(bt)')
ax.set_title("Actual vs Fitted Price", fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend(fontsize=9)

ax = axes[2]
ax.bar(close.index, residuals, width=2, color='#ff7f0e', alpha=0.6)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title("Residuals (log space)", fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Residual")

plt.suptitle(f"S&P500 - Log-Linear Regression  |  "
             f"CAGR = {CAGR*100:.2f}%  |  R2 = {R2:.4f}  |  "
             f"Std Dev = {std_dev:.4f}",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('sp500_regression.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: sp500_regression.png")
