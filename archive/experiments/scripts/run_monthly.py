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

data_daily['Nifty50_USD'] = data_daily['Nifty50'] / data_daily['USDINR']
data_daily = data_daily.drop(columns=['Nifty50'])

data_monthly = data_daily.resample('ME').last()
data_monthly = data_monthly[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]
print(f"Monthly data ready : {data_monthly.shape[0]} months x {data_monthly.shape[1]} columns")
print(f"  Date range       : {data_monthly.index[0].date()} -> {data_monthly.index[-1].date()}")

print("\n" + "="*65)
print("  FIRST 5 ROWS")
print("="*65)
print(data_monthly.head().to_string())
print("\n" + "="*65)
print("  LAST 5 ROWS")
print("="*65)
print(data_monthly.tail().to_string())
print("\n" + "="*65)
print("  DESCRIPTIVE STATISTICS")
print("="*65)
print(data_monthly.describe().round(4))
print("\n" + "="*65)
print("  MISSING VALUES")
print("="*65)
print(data_monthly.isnull().sum())
