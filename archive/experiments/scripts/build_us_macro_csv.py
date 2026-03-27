# ============================================================
# BUILD us_macro_data.csv -- US Macro Data (Jan 2005 - Dec 2024)
# ============================================================
# Sources:
#   Fed Rate   : FRED (FEDFUNDS)          -- Fed Funds effective rate
#   US CPI     : FRED (CPIAUCSL)          -- CPI All Urban, converted to YoY %
#   US PMI     : FRED (NAPM)              -- ISM Manufacturing PMI
#   US VIX     : Yahoo Finance (^VIX)     -- CBOE Volatility Index
#   US M2      : FRED (M2SL)             -- US M2 Money Supply (billions USD)
# ============================================================
import sys
import io
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

START = '2004-01-01'   # extra year for CPI YoY
END   = '2024-12-31'

# -- FRED helper (same pattern as build_macro_csv.py) --
def fetch_fred_csv(series_id, start=START, end=END, col_name=None):
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )
    name = col_name or series_id
    try:
        df = pd.read_csv(url, na_values='.')
        date_col = [c for c in df.columns if 'date' in c.lower()][0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        s = df.iloc[:, 0].astype(float)
        s.name = name
        valid = s.dropna()
        print(f"  [OK] {name}: {len(valid)} obs "
              f"({valid.index[0].date()} -> {valid.index[-1].date()})")
        return s
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return pd.Series(dtype=float, name=name)

def to_month_end(s):
    if s.empty:
        return s
    s.index = pd.to_datetime(s.index)
    return s.resample('ME').last()

# -- 1. Fed Funds Rate --
print("[1/5] Fed Funds Rate (FEDFUNDS)...")
fed_rate = fetch_fred_csv('FEDFUNDS', col_name='Fed_Rate')

# -- 2. US CPI (YoY %) --
print("[2/5] US CPI (CPIAUCSL)...")
cpi_index = fetch_fred_csv('CPIAUCSL', start='2004-01-01', col_name='CPI_Index')
us_cpi = cpi_index.pct_change(12) * 100
us_cpi.name = 'US_CPI'

# -- 3. US PMI proxy -- OECD Leading Indicator for US --
# ISM PMI (NAPM) no longer available on FRED.
# Using OECD Composite Leading Indicator as proxy.
print("[3/5] US PMI proxy (USALOLITONOSTSAM -- OECD Leading Indicator)...")
us_pmi = fetch_fred_csv('USALOLITONOSTSAM', col_name='US_PMI')

# -- 4. US VIX -- Yahoo Finance --
print("[4/5] US VIX (^VIX via yfinance)...")
try:
    vix_raw = yf.download('^VIX', start='2005-01-01', end=END,
                          auto_adjust=True, progress=False)
    us_vix = vix_raw['Close'].squeeze().resample('ME').last()
    us_vix.name = 'US_VIX'
    valid_vix = us_vix.dropna()
    print(f"  [OK] US_VIX: {len(valid_vix)} obs "
          f"({valid_vix.index[0].date()} -> {valid_vix.index[-1].date()})")
except Exception as e:
    print(f"  [FAIL] US VIX: {e}")
    us_vix = pd.Series(dtype=float, name='US_VIX')

# -- 5. US M2 Money Supply --
print("[5/5] US M2 (M2SL)...")
us_m2 = fetch_fred_csv('M2SL', col_name='US_M2')

# -- Assemble --
us_macro = pd.DataFrame({
    'Fed_Rate' : to_month_end(fed_rate),
    'US_CPI'   : to_month_end(us_cpi),
    'US_PMI'   : to_month_end(us_pmi),
    'US_VIX'   : to_month_end(us_vix),
    'US_M2'    : to_month_end(us_m2),
})
us_macro = us_macro.loc[pd.Timestamp('2005-01-01'):pd.Timestamp('2024-12-31')]

# Forward-fill gaps up to 3 months (Fed Rate held between meetings)
us_macro = us_macro.ffill(limit=3)

# -- Diagnostics --
print(f"\nShape    : {us_macro.shape}")
print(f"Period   : {us_macro.index[0].date()} -> {us_macro.index[-1].date()}")
print(f"\nMissing values:\n{us_macro.isnull().sum()}")
print(f"\nDescriptive stats:\n{us_macro.describe().round(2).to_string()}")

# -- Save --
us_macro.index.name = 'Date'
us_macro.to_csv('us_macro_data.csv', float_format='%.6f')
print("\n[SAVED] us_macro_data.csv")
