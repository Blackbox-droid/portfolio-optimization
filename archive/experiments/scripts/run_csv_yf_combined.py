import yfinance as yf
import pandas as pd
import numpy as np
import time as _time

START_DATE = '2005-01-01'
END_DATE   = '2019-12-31'

# ════════════════════════════════════════════════════════════════════
# STEP 1 — Load Nifty50_USD from CSV
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

# ════════════════════════════════════════════════════════════════════
# STEP 2 — Download SP500, Gold, USBond from Yahoo Finance
# ════════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════════
# STEP 3 — Combine Nifty50_USD (CSV) + YF data
# ════════════════════════════════════════════════════════════════════
data_daily = yf_data.copy()
data_daily['Nifty50_USD'] = nifty_usd
data_daily = data_daily[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]

# ════════════════════════════════════════════════════════════════════
# STEP 4 — Resample to Monthly
# ════════════════════════════════════════════════════════════════════
data_monthly = data_daily.resample('ME').last()
data_monthly = data_monthly.ffill()
data_monthly = data_monthly.loc[START_DATE:END_DATE]

print(f"\nMonthly data ready : {data_monthly.shape[0]} months x "
      f"{data_monthly.shape[1]} columns")
print(f"  Date range       : {data_monthly.index[0].date()} -> "
      f"{data_monthly.index[-1].date()}")

# ════════════════════════════════════════════════════════════════════
# STEP 5 — Read the data
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FIRST 5 ROWS")
print("="*65)
print(data_monthly.head().to_string())
print("\n" + "="*65)
print("  LAST 5 ROWS")
print("="*65)
print(data_monthly.tail().to_string())
print("\n" + "="*65)
print("  ALL 2005 ROWS  (verify Nifty50 coverage)")
print("="*65)
print(data_monthly.loc['2005'].to_string())
print("\n" + "="*65)
print("  DESCRIPTIVE STATISTICS")
print("="*65)
print(data_monthly.describe().round(4))
print("\n" + "="*65)
print("  MISSING VALUES")
print("="*65)
missing = data_monthly.isnull().sum()
print(missing.to_string() if missing.sum() > 0 else "  None - dataset complete.")
