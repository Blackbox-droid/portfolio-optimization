import yfinance as yf
import pandas as pd
import numpy as np

# STEP 1 - Load CSV data (Jan 2005 - Dec 2019)
csv_path = 'data_daily_combined_2005_2019.csv'
csv_data = pd.read_csv(csv_path, parse_dates=['Date'])
csv_data = csv_data.set_index('Date')
csv_data.index = pd.to_datetime(csv_data.index).tz_localize(None)
csv_data = csv_data.sort_index()
print("CSV loaded")
print(f"  Shape      : {csv_data.shape}")
print(f"  Date range : {csv_data.index[0].date()} -> {csv_data.index[-1].date()}")
print(f"  Columns    : {list(csv_data.columns)}")

# STEP 2 - Download Yahoo Finance data (Jan 2020 - Dec 2024)
YF_START = '2020-01-01'
YF_END   = '2024-12-31'
YF_TICKERS = {
    'Nifty50' : '^NSEI',
    'SP500'   : '^GSPC',
    'Gold'    : 'GLD',
    'USBond'  : 'IEF',
    'USDINR'  : 'USDINR=X',
}

def download(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close

print("\nDownloading Yahoo Finance data (2020-2024)...")
yf_raw = pd.DataFrame({
    name: download(ticker, YF_START, YF_END)
    for name, ticker in YF_TICKERS.items()
})
yf_raw.index.name = 'Date'

# Convert Nifty50 INR -> USD
yf_raw['Nifty50_USD'] = yf_raw['Nifty50'] / yf_raw['USDINR']
yf_raw = yf_raw.drop(columns=['Nifty50'])
yf_raw = yf_raw[['Nifty50_USD', 'SP500', 'Gold', 'USBond', 'USDINR']]

for col in ['Nifty50_USD', 'SP500', 'Gold', 'USBond']:
    valid = yf_raw[col].dropna()
    print(f"  {col:12s}  rows={len(valid)}  "
          f"{valid.index[0].date()} -> {valid.index[-1].date()}")

# STEP 3 - Stitch CSV (2005-2019) + YF (2020-2024)
csv_part = csv_data[['Nifty50_USD', 'SP500', 'Gold', 'USBond', 'USDINR']]
yf_part  = yf_raw[['Nifty50_USD', 'SP500', 'Gold', 'USBond', 'USDINR']]

data_full = pd.concat([csv_part, yf_part])
data_full = data_full.sort_index()
data_full = data_full[~data_full.index.duplicated(keep='first')]
data_full = data_full[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]
data_full.index.name = 'Date'

print(f"\nFull dataset stitched")
print(f"  Shape      : {data_full.shape[0]} rows x {data_full.shape[1]} columns")
print(f"  Date range : {data_full.index[0].date()} -> {data_full.index[-1].date()}")

# STEP 4 - Read the data
print("\n" + "="*65)
print("  FIRST 5 ROWS  (Jan 2005)")
print("="*65)
print(data_full.head().to_string())

print("\n" + "="*65)
print("  LAST 5 ROWS  (Dec 2024)")
print("="*65)
print(data_full.tail().to_string())

print("\n" + "="*65)
print("  STITCH POINT CHECK  (Dec 2019 -> Jan 2020)")
print("="*65)
print(data_full.loc['2019-12-25':'2020-01-10'].to_string())

print("\n" + "="*65)
print("  DESCRIPTIVE STATISTICS")
print("="*65)
print(data_full.describe().round(4))

print("\n" + "="*65)
print("  MISSING VALUES")
print("="*65)
print(data_full.isnull().sum())

# STEP 5 - Save to CSV
output_path = 'data_daily_2005_2024.csv'
data_full.to_csv(output_path)
print(f"\nSaved to {output_path}")
