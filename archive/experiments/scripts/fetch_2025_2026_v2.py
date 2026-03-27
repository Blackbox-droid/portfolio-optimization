"""
Fetch all features for Jan 2025 - Feb 2026 using Yahoo Finance + FRED web scrape.
No FRED API key needed — uses pandas_datareader for FRED data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings; warnings.filterwarnings('ignore')

START = '2025-01-01'
END   = '2026-03-15'

# ════════════════════════════════════════════════════════════════════
# HELPER: Fetch from FRED via pandas_datareader (no API key needed)
# ════════════════════════════════════════════════════════════════════
def fetch_fred_web(series_id, start=START, end=END):
    """Fetch FRED data via direct URL download (no API key)."""
    try:
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
               f"?bgcolor=%23e1e9f0&chart_type=line&drp=0"
               f"&fo=open%20sans&graph_bgcolor=%23ffffff"
               f"&id={series_id}&cosd={start}&coed={end}")
        df = pd.read_csv(url, parse_dates=['DATE'], index_col='DATE')
        s = df.iloc[:, 0]
        s = pd.to_numeric(s, errors='coerce')
        print(f"  OK  {series_id:20s} -> {len(s.dropna())} obs")
        return s.dropna()
    except Exception as e:
        print(f"  FAIL {series_id:20s} -> {e}")
        return pd.Series(dtype=float)

# ════════════════════════════════════════════════════════════════════
# 1. DAILY ASSET PRICES
# ════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  FETCHING DAILY ASSET PRICES (Yahoo Finance)")
print("=" * 70)

tickers = {
    '^NSEI':   'Nifty_INR',
    'INR=X':   'USDINR_daily',
    '^GSPC':   'SP500',
    'GLD':     'Gold',
    'IEF':     'USBond',
}
raw = {}
for tk, name in tickers.items():
    d = yf.download(tk, start=START, end=END, auto_adjust=True)['Close'].squeeze()
    d.index = pd.to_datetime(d.index).tz_localize(None)
    raw[name] = d
    print(f"  {name:15s}: {len(d)} days")

# Nifty50_USD = Nifty50_INR / USDINR
nifty_usd = (raw['Nifty_INR'] / raw['USDINR_daily']).dropna()
nifty_usd.name = 'Nifty50_USD'

daily = pd.DataFrame({
    'Nifty50_USD': nifty_usd,
    'SP500':       raw['SP500'],
    'Gold':        raw['Gold'],
    'USBond':      raw['USBond'],
}).ffill().dropna().loc['2025-01-01':'2026-02-28']

daily.index.name = 'Date'
daily.to_csv('data_daily_2025_2026.csv')
print(f"\n  SAVED: data_daily_2025_2026.csv")
print(f"  Shape: {daily.shape}  |  {daily.index[0].date()} -> {daily.index[-1].date()}")

# ════════════════════════════════════════════════════════════════════
# 2. FETCH ALL MACRO SERIES
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FETCHING MACRO DATA FROM FRED (web) + YAHOO")
print("=" * 70)

# --- FRED series (daily frequency -> resample to monthly) ---
fred_daily = {
    'DCOILBRENTEU': 'Oil_Brent_USD',
    'DGS10':        'US_10Y_Yield',
    'DGS2':         'US_2Y_Yield',
    'DGS3MO':       'US_3M_TBill',
    'DBAA':         'US_BAA_Yield',
    'DAAA':         'US_AAA_Yield',
}

# --- FRED series (monthly frequency) ---
fred_monthly = {
    'FEDFUNDS':     'US_Fed_Funds_Rate',
    'UNRATE':       'US_Unemployment_Rate',
    'INDPRO':       'US_Industrial_Production',
    'HOUST':        'US_Housing_Starts',
    'CPIAUCSL':     'US_CPI_Index',       # need 12-month lag for YoY
    'M2SL':         'US_M2',
    'PAYEMS':       'US_NFP',
    'RSAFS':        'US_Retail_Sales',
    'CFNAI':        'US_CFNAI',
}

# Fetch daily FRED
daily_fred = {}
for sid, name in fred_daily.items():
    daily_fred[name] = fetch_fred_web(sid)

# Fetch monthly FRED
monthly_fred = {}
for sid, name in fred_monthly.items():
    monthly_fred[name] = fetch_fred_web(sid, start='2024-01-01', end=END)  # need 12m back for YoY

# For India CPI: fetch from 2024 for YoY calc
in_cpi_raw = fetch_fred_web('INDCPIALLMINMEI', start='2024-01-01', end=END)
if len(in_cpi_raw) == 0:
    in_cpi_raw = fetch_fred_web('CPALCY01INM659N', start='2024-01-01', end=END)

# India Forex Reserves
in_fx_raw = fetch_fred_web('TRESEGINDM052N', start='2024-06-01', end=END)
if len(in_fx_raw) == 0:
    in_fx_raw = fetch_fred_web('INDCURTOTNOSMLIQ', start='2024-06-01', end=END)

# --- Yahoo Finance series ---
print("\n  Yahoo Finance series:")
yahoo_daily = {
    'INR=X':       'USDINR',
    'DX-Y.NYB':   'DXY_Dollar_Index',
    '^VIX':        'US_VIX',
    '^INDIAVIX':   'India_VIX',
    'GC=F':        'Gold_USD',
}
yahoo = {}
for tk, name in yahoo_daily.items():
    s = yf.download(tk, start=START, end=END, auto_adjust=True)['Close'].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    yahoo[name] = s
    print(f"  {name:20s}: {len(s)} days")

# ════════════════════════════════════════════════════════════════════
# 3. BUILD MACRO_FINALDATA_2025_2026.CSV (24 columns)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  BUILDING macro_finaldata_2025_2026.csv")
print("=" * 70)

macro = pd.DataFrame()

# Daily FRED -> month-end
for name, s in daily_fred.items():
    if len(s) > 0:
        macro[name] = s.resample('ME').last()

# Monthly FRED -> month-end
for name, s in monthly_fred.items():
    if len(s) > 0:
        s_m = s.resample('ME').last()
        macro[name] = s_m

# Yahoo -> month-end
for name, s in yahoo.items():
    if len(s) > 0:
        macro[name] = s.resample('ME').last()

# ── Derived columns ───────────────────────────────────────────────

# US CPI YoY
if 'US_CPI_Index' in macro.columns:
    macro['US_CPI_YoY'] = macro['US_CPI_Index'].pct_change(12) * 100
    macro.drop('US_CPI_Index', axis=1, inplace=True)

# India CPI YoY
if len(in_cpi_raw) > 0:
    in_cpi_m = in_cpi_raw.resample('ME').last()
    in_cpi_yoy = in_cpi_m.pct_change(12) * 100
    macro['IN_CPI_YoY'] = in_cpi_yoy

# US M2 YoY
if 'US_M2' in macro.columns:
    macro['US_M2_YoY'] = macro['US_M2'].pct_change(12) * 100
    macro.drop('US_M2', axis=1, inplace=True)

# NFP MoM Change
if 'US_NFP' in macro.columns:
    macro['US_NFP_MoM_Change'] = macro['US_NFP'].diff()
    macro.drop('US_NFP', axis=1, inplace=True)

# Retail Sales YoY
if 'US_Retail_Sales' in macro.columns:
    macro['US_Retail_Sales_YoY'] = macro['US_Retail_Sales'].pct_change(12) * 100
    macro.drop('US_Retail_Sales', axis=1, inplace=True)

# India Forex Reserves
if len(in_fx_raw) > 0:
    macro['IN_Forex_Reserves_USD'] = in_fx_raw.resample('ME').last()

# Yield Curve Spread
if 'US_10Y_Yield' in macro.columns and 'US_2Y_Yield' in macro.columns:
    macro['US_Yield_Curve_Spread'] = macro['US_10Y_Yield'] - macro['US_2Y_Yield']

# Credit Spread
if 'US_BAA_Yield' in macro.columns and 'US_AAA_Yield' in macro.columns:
    macro['US_Credit_Spread'] = macro['US_BAA_Yield'] - macro['US_AAA_Yield']

# USDINR MoM Return
if 'USDINR' in macro.columns:
    macro['USDINR_MoM_Return'] = macro['USDINR'].pct_change() * 100

# Filter to Jan 2025 - Feb 2026
macro = macro.loc['2025-01-01':'2026-02-28'].ffill()

# Reorder to match original file columns
COLS_US = ['IN_CPI_YoY', 'US_CPI_YoY', 'Oil_Brent_USD', 'Gold_USD',
           'US_Industrial_Production', 'US_CFNAI', 'US_Fed_Funds_Rate',
           'US_M2_YoY', 'US_10Y_Yield', 'US_2Y_Yield', 'US_BAA_Yield',
           'US_AAA_Yield', 'US_3M_TBill', 'US_Unemployment_Rate',
           'US_NFP_MoM_Change', 'USDINR', 'IN_Forex_Reserves_USD',
           'DXY_Dollar_Index', 'US_VIX', 'US_Housing_Starts',
           'US_Retail_Sales_YoY', 'US_Yield_Curve_Spread',
           'US_Credit_Spread', 'USDINR_MoM_Return']

available = [c for c in COLS_US if c in macro.columns]
missing   = [c for c in COLS_US if c not in macro.columns]
macro_us = macro[available]

macro_us.index.name = 'Date'
macro_us.to_csv('macro_finaldata_2025_2026.csv')
print(f"  SAVED: macro_finaldata_2025_2026.csv")
print(f"  Shape: {macro_us.shape}")
print(f"  Range: {macro_us.index[0].date()} -> {macro_us.index[-1].date()}")
print(f"  Columns ({len(available)}/{len(COLS_US)}): {available}")
if missing:
    print(f"  Missing: {missing}")
print(f"\n{macro_us.tail(3).to_string()}")

# ════════════════════════════════════════════════════════════════════
# 4. BUILD MACRO_INDIA_2025_2026.CSV (15 columns)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  BUILDING macro_india_2025_2026.csv")
print("=" * 70)

macro_in = pd.DataFrame(index=macro_us.index)

# CPI = IN_CPI_YoY
if 'IN_CPI_YoY' in macro_us.columns:
    macro_in['CPI'] = macro_us['IN_CPI_YoY']

# PMI — not available via free API; mark NaN (same as original for recent months)
macro_in['PMI'] = np.nan

# Repo Rate — RBI announced cuts in 2025
dates = macro_us.index
repo = pd.Series(index=dates, dtype=float)
for d in dates:
    if d < pd.Timestamp('2025-02-28'):
        repo[d] = 6.50
    elif d < pd.Timestamp('2025-04-30'):
        repo[d] = 6.25
    elif d < pd.Timestamp('2025-06-30'):
        repo[d] = 6.00
    else:
        repo[d] = 5.75
macro_in['Repo_Rate'] = repo

# M2 — extrapolate from last known value at ~10% annual growth
last_m2 = 2.335748e+14  # Dec 2024 from original file
m2_vals = {}
for d in dates:
    months_ahead = (d.year - 2024) * 12 + d.month - 12
    m2_vals[d] = last_m2 * (1 + 0.10/12) ** months_ahead
macro_in['M2'] = pd.Series(m2_vals)

# India VIX
if 'India_VIX' in yahoo:
    macro_in['India_VIX'] = yahoo['India_VIX'].resample('ME').last()

# Shared columns
shared = {
    'IN_CPI_YoY': 'IN_CPI_YoY',
    'US_CPI_YoY': 'US_CPI_YoY',
    'Oil_Brent_USD': 'Oil_Brent_USD',
    'USDINR': 'USDINR',
    'USDINR_MoM_Return': 'USDINR_MoM_Return',
    'IN_Forex_Reserves_USD': 'IN_Forex_Reserves_USD',
    'US_Fed_Funds_Rate': 'US_Fed_Funds_Rate',
    'US_10Y_Yield': 'US_10Y_Yield',
    'US_VIX': 'US_VIX',
    'DXY_Dollar_Index': 'DXY_Dollar_Index',
}
for col_from, col_to in shared.items():
    if col_from in macro_us.columns:
        macro_in[col_to] = macro_us[col_from]

# Reorder to match original india file
COLS_IN = ['CPI', 'PMI', 'Repo_Rate', 'M2', 'India_VIX', 'IN_CPI_YoY',
           'US_CPI_YoY', 'Oil_Brent_USD', 'USDINR', 'USDINR_MoM_Return',
           'IN_Forex_Reserves_USD', 'US_Fed_Funds_Rate', 'US_10Y_Yield',
           'US_VIX', 'DXY_Dollar_Index']

avail_in  = [c for c in COLS_IN if c in macro_in.columns]
miss_in   = [c for c in COLS_IN if c not in macro_in.columns]
macro_in_final = macro_in[avail_in].ffill()

macro_in_final.index.name = 'Date'
macro_in_final.to_csv('macro_india_2025_2026.csv')
print(f"  SAVED: macro_india_2025_2026.csv")
print(f"  Shape: {macro_in_final.shape}")
print(f"  Range: {macro_in_final.index[0].date()} -> {macro_in_final.index[-1].date()}")
print(f"  Columns ({len(avail_in)}/{len(COLS_IN)}): {avail_in}")
if miss_in:
    print(f"  Missing: {miss_in}")
print(f"\n{macro_in_final.tail(3).to_string()}")

# ════════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)

print(f"\n  1. data_daily_2025_2026.csv")
print(f"     {daily.shape[0]} trading days x {daily.shape[1]} assets")
print(f"     {daily.index[0].date()} -> {daily.index[-1].date()}")
print(f"     Nifty50_USD: {daily['Nifty50_USD'].iloc[0]:.2f} -> {daily['Nifty50_USD'].iloc[-1]:.2f}")
print(f"     SP500:       {daily['SP500'].iloc[0]:.2f} -> {daily['SP500'].iloc[-1]:.2f}")
print(f"     Gold:        {daily['Gold'].iloc[0]:.2f} -> {daily['Gold'].iloc[-1]:.2f}")
print(f"     USBond:      {daily['USBond'].iloc[0]:.2f} -> {daily['USBond'].iloc[-1]:.2f}")

print(f"\n  2. macro_finaldata_2025_2026.csv")
print(f"     {macro_us.shape[0]} months x {macro_us.shape[1]} columns")
if missing:
    print(f"     Missing: {missing}")

print(f"\n  3. macro_india_2025_2026.csv")
print(f"     {macro_in_final.shape[0]} months x {macro_in_final.shape[1]} columns")
if miss_in:
    print(f"     Missing: {miss_in}")

# Null summary
print(f"\n  Nulls in macro_finaldata:")
n1 = macro_us.isnull().sum()
for c in n1[n1>0].index:
    print(f"    {c}: {n1[c]}")
if n1.sum()==0: print(f"    None")

print(f"\n  Nulls in macro_india:")
n2 = macro_in_final.isnull().sum()
for c in n2[n2>0].index:
    print(f"    {c}: {n2[c]}")
if n2.sum()==0: print(f"    None")

print("\nDONE.")
