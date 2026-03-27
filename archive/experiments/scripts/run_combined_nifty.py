import yfinance as yf
import pandas as pd
import numpy as np
import time as _time

# ════════════════════════════════════════════════════════════════════
# STEP 1 — Load Nifty50 CSV data (2005–2007)
# ════════════════════════════════════════════════════════════════════
csv_files = [
    r"C:\Users\Lenovo\Downloads\NIFTY 50_Historical_PR_01012005to31122005.csv",
    r"C:\Users\Lenovo\Downloads\NIFTY 50_Historical_PR_01012006to31122006.csv",
    r"C:\Users\Lenovo\Downloads\NIFTY 50_Historical_PR_01012007to31122007.csv",
]

csv_frames = []
for f in csv_files:
    df = pd.read_csv(f)
    df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y')
    df = df[['Date', 'Close']].rename(columns={'Close': 'Nifty50'})
    df = df.set_index('Date').sort_index()
    csv_frames.append(df)
    print(f"  CSV loaded: {f.split(chr(92))[-1]}  -> {len(df)} rows  "
          f"({df.index[0].date()} -> {df.index[-1].date()})")

nifty_csv = pd.concat(csv_frames)
nifty_csv = nifty_csv[~nifty_csv.index.duplicated(keep='first')]
nifty_csv = nifty_csv.sort_index()
print(f"\nNifty50 CSV combined: {len(nifty_csv)} rows  "
      f"({nifty_csv.index[0].date()} -> {nifty_csv.index[-1].date()})")

# ════════════════════════════════════════════════════════════════════
# STEP 2 — Download yfinance data (2008 onwards)
# ════════════════════════════════════════════════════════════════════
START_YF = '2008-01-01'
END_DATE = '2019-12-31'
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

# Also need USDINR for 2005-2007 to convert CSV Nifty to USD
print("\nDownloading yfinance data...")
usdinr_full = download('USDINR=X', '2005-01-01', END_DATE)
print(f"  USDINR: {len(usdinr_full)} rows ({usdinr_full.index[0].date()} -> {usdinr_full.index[-1].date()})")

yf_data = {}
for name, ticker in TICKERS.items():
    if name == 'USDINR':
        yf_data[name] = download(ticker, '2005-01-01', END_DATE)
    elif name == 'Nifty50':
        yf_data[name] = download(ticker, START_YF, END_DATE)
    else:
        yf_data[name] = download(ticker, '2005-01-01', END_DATE)
    print(f"  {name:12s}: {len(yf_data[name])} rows")

# ════════════════════════════════════════════════════════════════════
# STEP 3 — Combine Nifty50: CSV (2005-2007) + yfinance (2008+)
# ════════════════════════════════════════════════════════════════════
nifty_yf = yf_data['Nifty50']
nifty_yf = pd.DataFrame({'Nifty50': nifty_yf})

# Concatenate: CSV first, then yfinance
nifty_combined = pd.concat([nifty_csv, nifty_yf])
nifty_combined = nifty_combined[~nifty_combined.index.duplicated(keep='last')]
nifty_combined = nifty_combined.sort_index()
print(f"\nNifty50 combined: {len(nifty_combined)} rows  "
      f"({nifty_combined.index[0].date()} -> {nifty_combined.index[-1].date()})")

# ════════════════════════════════════════════════════════════════════
# STEP 4 — Build final DataFrame
# ════════════════════════════════════════════════════════════════════
data = pd.DataFrame({
    'Nifty50': nifty_combined['Nifty50'],
    'SP500': yf_data['SP500'],
    'Gold': yf_data['Gold'],
    'USBond': yf_data['USBond'],
    'USDINR': yf_data['USDINR'],
})
data.index.name = 'Date'

# Convert Nifty50 INR -> USD
data['Nifty50_USD'] = data['Nifty50'] / data['USDINR']
data = data.drop(columns=['Nifty50'])
data = data[['Nifty50_USD', 'SP500', 'Gold', 'USBond', 'USDINR']]

# ════════════════════════════════════════════════════════════════════
# STEP 5 — Resample to monthly
# ════════════════════════════════════════════════════════════════════
data_monthly = data.resample('ME').last()
data_monthly = data_monthly[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]

print(f"\nMonthly data ready : {data_monthly.shape[0]} months x {data_monthly.shape[1]} columns")
print(f"  Date range       : {data_monthly.index[0].date()} -> {data_monthly.index[-1].date()}")

print("\n" + "="*65)
print("  FIRST 10 ROWS")
print("="*65)
print(data_monthly.head(10).to_string())
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

# ── Save to CSV ──────────────────────────────────────────────────────
data.to_csv('data_daily_combined_2005_2019.csv')
print("\nSaved: data_daily_combined_2005_2019.csv  (daily)")

data_monthly.to_csv('data_monthly_combined_2005_2019.csv')
print("Saved: data_monthly_combined_2005_2019.csv  (monthly)")
