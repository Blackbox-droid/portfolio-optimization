import yfinance as yf
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────
START_DATE = '2008-01-01'
END_DATE   = '2022-12-31'

TICKERS = {
    'Nifty50' : '^NSEI',
    'SP500'   : '^GSPC',
    'Gold'    : 'GLD',
    'USBond'  : 'TLT',
    'VIX'     : '^VIX',
    'USDINR'  : 'USDINR=X',
}

# ── Download function ────────────────────────────────────────────────
def download(ticker, start, end):
    df    = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = df['Close'].squeeze()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close

# ── Download all tickers ─────────────────────────────────────────────
print("Downloading data...")
data = pd.DataFrame({
    name: download(ticker, START_DATE, END_DATE)
    for name, ticker in TICKERS.items()
})

data.index.name = 'Date'

print(f"✓ Raw data downloaded: {data.shape[0]} rows × {data.shape[1]} columns")

# ── Convert Nifty50 from INR to USD ──────────────────────────────────
data['Nifty50_USD'] = data['Nifty50'] / data['USDINR']
data = data.drop(columns=['Nifty50'])

# ── Final column order ───────────────────────────────────────────────
data = data[['Nifty50_USD', 'SP500', 'Gold', 'USBond', 'VIX', 'USDINR']]

# ── Read the data ────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DATASET OVERVIEW")
print("="*60)
print(f"Shape         : {data.shape[0]} rows × {data.shape[1]} columns")
print(f"Date range    : {data.index[0].date()} → {data.index[-1].date()}")
print(f"Columns       : {list(data.columns)}")

print("\n" + "="*60)
print("  FIRST 5 ROWS")
print("="*60)
print(data.head())

print("\n" + "="*60)
print("  LAST 5 ROWS")
print("="*60)
print(data.tail())

print("\n" + "="*60)
print("  DESCRIPTIVE STATISTICS")
print("="*60)
print(data.describe().round(4))

print("\n" + "="*60)
print("  MISSING VALUES")
print("="*60)
missing = data.isnull().sum()
print(missing)