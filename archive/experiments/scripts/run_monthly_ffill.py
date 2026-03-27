import yfinance as yf
import pandas as pd
import numpy as np
import time as _time

START_DATE = '2005-01-01'
END_DATE   = '2019-12-31'
TICKERS = {
    'Nifty50' : '^NSEI',
    'SP500'   : '^GSPC',
    'Gold'    : 'GLD',
    'USBond'  : 'TLT',
    'USDINR'  : 'USDINR=X',
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

print("Downloading data...")
data_daily = pd.DataFrame({
    name: download(ticker, START_DATE, END_DATE)
    for name, ticker in TICKERS.items()
})
data_daily.index.name = 'Date'

# Check Nifty50 availability
nifty_first = data_daily['Nifty50'].first_valid_index()
nifty_missing_before = data_daily['Nifty50'].isna().sum()
print(f"\n  Nifty50 first available date : {nifty_first.date()}")
print(f"  Missing rows before first date: {nifty_missing_before}")

# Forward-fill and backfill Nifty50
data_daily['Nifty50'] = data_daily['Nifty50'].ffill()
data_daily['Nifty50'] = data_daily['Nifty50'].bfill()

# Forward-fill USDINR and convert Nifty50 to USD
data_daily['USDINR']      = data_daily['USDINR'].ffill().bfill()
data_daily['Nifty50_USD'] = data_daily['Nifty50'] / data_daily['USDINR']
data_daily = data_daily.drop(columns=['Nifty50'])

# Resample to monthly
data_monthly = data_daily.resample('ME').last()
data_monthly = data_monthly[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]
data_monthly = data_monthly.ffill()

print(f"\nMonthly data ready : {data_monthly.shape[0]} months x "
      f"{data_monthly.shape[1]} columns")
print(f"  Date range       : {data_monthly.index[0].date()} -> "
      f"{data_monthly.index[-1].date()}")

print("\n" + "="*65)
print("  FIRST 5 ROWS  (check 2005 Nifty50 values)")
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
