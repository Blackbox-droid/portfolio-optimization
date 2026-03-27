"""
Fetch all features for Jan 2025 – Feb 2026 to extend the three dataset files.
Outputs:
  - data_daily_2025_2026.csv        (Nifty50_USD, SP500, Gold, USBond — daily)
  - macro_finaldata_2025_2026.csv   (24 US/global macro cols — monthly)
  - macro_india_2025_2026.csv       (15 India+global macro cols — monthly)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from fredapi import Fred
import warnings; warnings.filterwarnings('ignore')

START = '2025-01-01'
END   = '2026-03-01'   # fetch through Mar 1 to ensure Feb 2026 month-end is captured

# ════════════════════════════════════════════════════════════════════
# 1. DAILY ASSET PRICES  (Nifty50_USD, SP500, Gold, USBond)
# ════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  FETCHING DAILY ASSET PRICES")
print("=" * 70)

# Nifty50 in INR (^NSEI) + USDINR to compute Nifty50_USD
nifty_inr = yf.download('^NSEI', start=START, end=END, auto_adjust=True)['Close']
usdinr_d  = yf.download('INR=X', start=START, end=END, auto_adjust=True)['Close']

# SP500 via SPY ETF (matches original file using index-like proxy)
sp500 = yf.download('^GSPC', start=START, end=END, auto_adjust=True)['Close']

# Gold via GLD ETF
gold = yf.download('GLD', start=START, end=END, auto_adjust=True)['Close']

# USBond via IEF ETF (7-10yr Treasury)
usbond = yf.download('IEF', start=START, end=END, auto_adjust=True)['Close']

# Build daily dataframe
daily = pd.DataFrame({
    'SP500': sp500.squeeze(),
    'Gold': gold.squeeze(),
    'USBond': usbond.squeeze(),
})
daily.index = pd.to_datetime(daily.index).tz_localize(None)

# Nifty USD = Nifty INR / USDINR
nifty_inr.index = pd.to_datetime(nifty_inr.index).tz_localize(None)
usdinr_d.index  = pd.to_datetime(usdinr_d.index).tz_localize(None)
nifty_usd = (nifty_inr.squeeze() / usdinr_d.squeeze()).dropna()
nifty_usd.name = 'Nifty50_USD'

daily = daily.join(nifty_usd, how='outer').sort_index()
daily = daily[['Nifty50_USD', 'SP500', 'Gold', 'USBond']]
daily = daily.ffill().dropna()

# Filter to valid range
daily = daily.loc['2025-01-01':'2026-02-28']

daily.index.name = 'Date'
daily.to_csv('data_daily_2025_2026.csv')
print(f"\n  data_daily_2025_2026.csv saved")
print(f"  Shape: {daily.shape}")
print(f"  Range: {daily.index[0].date()} -> {daily.index[-1].date()}")
print(f"\n  First row:\n{daily.head(1).to_string()}")
print(f"\n  Last row:\n{daily.tail(1).to_string()}")

# ════════════════════════════════════════════════════════════════════
# 2. FRED MACRO DATA (monthly)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FETCHING MACRO DATA FROM FRED + YAHOO")
print("=" * 70)

fred = Fred(api_key='d20de516f9c7e8eb3539db9e6bb0cedb')

def get_fred(series_id, name):
    try:
        s = fred.get_series(series_id, observation_start=START, observation_end=END)
        s.name = name
        return s
    except Exception as e:
        print(f"  WARNING: Failed to fetch {name} ({series_id}): {e}")
        return pd.Series(dtype=float, name=name)

# FRED series mapping
fred_series = {
    # US CPI YoY (calculated from CPI index)
    'CPIAUCSL':        'US_CPI_Index',
    # Oil
    'DCOILBRENTEU':    'Oil_Brent_USD',
    # US Industrial Production
    'INDPRO':          'US_Industrial_Production',
    # Fed Funds Rate
    'FEDFUNDS':        'US_Fed_Funds_Rate',
    # M2
    'M2SL':            'US_M2',
    # Treasury yields
    'DGS10':           'US_10Y_Yield',
    'DGS2':            'US_2Y_Yield',
    'DGS3MO':          'US_3M_TBill',
    # Corporate bond yields
    'DBAA':            'US_BAA_Yield',
    'DAAA':            'US_AAA_Yield',
    # Unemployment
    'UNRATE':          'US_Unemployment_Rate',
    # NFP
    'PAYEMS':          'US_NFP',
    # Housing starts
    'HOUST':           'US_Housing_Starts',
    # Retail sales
    'RSAFS':           'US_Retail_Sales',
}

fred_data = {}
for sid, name in fred_series.items():
    fred_data[name] = get_fred(sid, name)
    print(f"  Fetched {name}: {len(fred_data[name])} obs")

# ── Yahoo Finance monthly series ──────────────────────────────────
# USDINR
usdinr_m = yf.download('INR=X', start=START, end=END, auto_adjust=True)['Close'].squeeze()
usdinr_m.index = pd.to_datetime(usdinr_m.index).tz_localize(None)

# DXY
dxy = yf.download('DX-Y.NYB', start=START, end=END, auto_adjust=True)['Close'].squeeze()
dxy.index = pd.to_datetime(dxy.index).tz_localize(None)

# VIX
vix = yf.download('^VIX', start=START, end=END, auto_adjust=True)['Close'].squeeze()
vix.index = pd.to_datetime(vix.index).tz_localize(None)

# India VIX
india_vix = yf.download('^INDIAVIX', start=START, end=END, auto_adjust=True)['Close'].squeeze()
india_vix.index = pd.to_datetime(india_vix.index).tz_localize(None)

# Gold (for macro file)
gold_usd = yf.download('GC=F', start=START, end=END, auto_adjust=True)['Close'].squeeze()
gold_usd.index = pd.to_datetime(gold_usd.index).tz_localize(None)

print(f"  Fetched USDINR: {len(usdinr_m)} obs")
print(f"  Fetched DXY: {len(dxy)} obs")
print(f"  Fetched US_VIX: {len(vix)} obs")
print(f"  Fetched India_VIX: {len(india_vix)} obs")
print(f"  Fetched Gold_USD: {len(gold_usd)} obs")

# ════════════════════════════════════════════════════════════════════
# 3. BUILD MACRO_FINALDATA_2025_2026.CSV  (24 columns)
# ════════════════════════════════════════════════════════════════════
# Resample everything to month-end
macro_us = pd.DataFrame()

# Yields (daily -> month-end)
for col in ['US_10Y_Yield', 'US_2Y_Yield', 'US_3M_TBill', 'US_BAA_Yield', 'US_AAA_Yield']:
    s = fred_data.get(col, pd.Series(dtype=float))
    if len(s) > 0:
        s.index = pd.to_datetime(s.index)
        macro_us[col] = s.resample('ME').last()

# Monthly series (already monthly)
for col in ['US_Fed_Funds_Rate', 'US_Unemployment_Rate', 'US_Industrial_Production',
            'US_Housing_Starts']:
    s = fred_data.get(col, pd.Series(dtype=float))
    if len(s) > 0:
        s.index = pd.to_datetime(s.index)
        macro_us[col] = s.resample('ME').last()

# Oil (daily -> month-end)
oil_s = fred_data.get('Oil_Brent_USD', pd.Series(dtype=float))
if len(oil_s) > 0:
    oil_s.index = pd.to_datetime(oil_s.index)
    macro_us['Oil_Brent_USD'] = oil_s.resample('ME').last()

# Gold USD (daily -> month-end)
macro_us['Gold_USD'] = gold_usd.resample('ME').last()

# US CPI YoY
cpi_idx = fred_data.get('US_CPI_Index', pd.Series(dtype=float))
if len(cpi_idx) > 0:
    cpi_idx.index = pd.to_datetime(cpi_idx.index)
    cpi_m = cpi_idx.resample('ME').last()
    macro_us['US_CPI_YoY'] = cpi_m.pct_change(12) * 100

# IN CPI YoY — use India CPI from FRED or approximate
in_cpi = get_fred('INDCPIALLMINMEI', 'IN_CPI_Index')
if len(in_cpi) > 0:
    in_cpi.index = pd.to_datetime(in_cpi.index)
    in_cpi_m = in_cpi.resample('ME').last()
    macro_us['IN_CPI_YoY'] = in_cpi_m.pct_change(12) * 100
else:
    print("  WARNING: India CPI not available from FRED, trying alternate series")
    in_cpi = get_fred('CPALCY01INM659N', 'IN_CPI_Index_alt')
    if len(in_cpi) > 0:
        in_cpi.index = pd.to_datetime(in_cpi.index)
        in_cpi_m = in_cpi.resample('ME').last()
        macro_us['IN_CPI_YoY'] = in_cpi_m.pct_change(12) * 100

# US M2 YoY
m2 = fred_data.get('US_M2', pd.Series(dtype=float))
if len(m2) > 0:
    m2.index = pd.to_datetime(m2.index)
    m2_m = m2.resample('ME').last()
    macro_us['US_M2_YoY'] = m2_m.pct_change(12) * 100

# NFP MoM Change
nfp = fred_data.get('US_NFP', pd.Series(dtype=float))
if len(nfp) > 0:
    nfp.index = pd.to_datetime(nfp.index)
    nfp_m = nfp.resample('ME').last()
    macro_us['US_NFP_MoM_Change'] = nfp_m.diff()

# Retail Sales YoY
rsales = fred_data.get('US_Retail_Sales', pd.Series(dtype=float))
if len(rsales) > 0:
    rsales.index = pd.to_datetime(rsales.index)
    rsales_m = rsales.resample('ME').last()
    macro_us['US_Retail_Sales_YoY'] = rsales_m.pct_change(12) * 100

# CFNAI
cfnai = get_fred('CFNAI', 'US_CFNAI')
if len(cfnai) > 0:
    cfnai.index = pd.to_datetime(cfnai.index)
    macro_us['US_CFNAI'] = cfnai.resample('ME').last()

# Yahoo-sourced monthly
macro_us['USDINR']           = usdinr_m.resample('ME').last()
macro_us['DXY_Dollar_Index'] = dxy.resample('ME').last()
macro_us['US_VIX']           = vix.resample('ME').last()

# IN Forex Reserves from FRED
in_fx = get_fred('INDCURTOTNOSMLIQ', 'IN_Forex_Reserves_USD')
if len(in_fx) > 0:
    in_fx.index = pd.to_datetime(in_fx.index)
    macro_us['IN_Forex_Reserves_USD'] = in_fx.resample('ME').last()

# Derived columns
if 'US_10Y_Yield' in macro_us.columns and 'US_2Y_Yield' in macro_us.columns:
    macro_us['US_Yield_Curve_Spread'] = macro_us['US_10Y_Yield'] - macro_us['US_2Y_Yield']
if 'US_BAA_Yield' in macro_us.columns and 'US_AAA_Yield' in macro_us.columns:
    macro_us['US_Credit_Spread'] = macro_us['US_BAA_Yield'] - macro_us['US_AAA_Yield']

# USDINR MoM Return
if 'USDINR' in macro_us.columns:
    macro_us['USDINR_MoM_Return'] = macro_us['USDINR'].pct_change() * 100

# Filter & reorder to match original file
MACRO_US_COLS = ['IN_CPI_YoY', 'US_CPI_YoY', 'Oil_Brent_USD', 'Gold_USD',
                 'US_Industrial_Production', 'US_CFNAI', 'US_Fed_Funds_Rate',
                 'US_M2_YoY', 'US_10Y_Yield', 'US_2Y_Yield', 'US_BAA_Yield',
                 'US_AAA_Yield', 'US_3M_TBill', 'US_Unemployment_Rate',
                 'US_NFP_MoM_Change', 'USDINR', 'IN_Forex_Reserves_USD',
                 'DXY_Dollar_Index', 'US_VIX', 'US_Housing_Starts',
                 'US_Retail_Sales_YoY', 'US_Yield_Curve_Spread',
                 'US_Credit_Spread', 'USDINR_MoM_Return']

# Keep only available columns in order
available_us = [c for c in MACRO_US_COLS if c in macro_us.columns]
macro_us_final = macro_us[available_us].loc['2025-01-01':'2026-02-28'].ffill()

macro_us_final.index.name = 'Date'
macro_us_final.to_csv('macro_finaldata_2025_2026.csv')
print(f"\n  macro_finaldata_2025_2026.csv saved")
print(f"  Shape: {macro_us_final.shape}")
print(f"  Columns: {macro_us_final.columns.tolist()}")
print(f"  Range: {macro_us_final.index[0].date()} -> {macro_us_final.index[-1].date()}")
print(f"\n  Last 3 rows:")
print(macro_us_final.tail(3).to_string())

# ════════════════════════════════════════════════════════════════════
# 4. BUILD MACRO_INDIA_2025_2026.CSV  (15 columns)
# ════════════════════════════════════════════════════════════════════
macro_in = pd.DataFrame()

# CPI = IN_CPI_YoY (same column, different name in india file)
if 'IN_CPI_YoY' in macro_us_final.columns:
    macro_in['CPI'] = macro_us_final['IN_CPI_YoY']
    macro_in['IN_CPI_YoY'] = macro_us_final['IN_CPI_YoY']

# PMI — fetch S&P Global India Manufacturing PMI proxy
# (not on FRED, we'll try to get from available data)
# Use NaN for now — same as original file has NaN for recent months
macro_in['PMI'] = np.nan

# Repo Rate — RBI repo rate (manual: 6.50% until Feb 2025, 6.25% from Feb 7 2025, 6.00% from Apr 9 2025)
dates = macro_us_final.index
repo = pd.Series(index=dates, dtype=float)
for d in dates:
    if d < pd.Timestamp('2025-02-07'):
        repo[d] = 6.50
    elif d < pd.Timestamp('2025-04-09'):
        repo[d] = 6.25
    elif d < pd.Timestamp('2025-06-06'):
        repo[d] = 6.00
    else:
        repo[d] = 5.75  # Jun 2025 cut
macro_in['Repo_Rate'] = repo

# M2 — India M2 money supply (use FRED or extrapolate)
# Try FRED India M2 (broad money)
in_m2 = get_fred('MYAGM2INM189S', 'IN_M2')
if len(in_m2) > 0:
    in_m2.index = pd.to_datetime(in_m2.index)
    macro_in['M2'] = in_m2.resample('ME').last()
else:
    # Use last known value from original file and grow at ~10% annual rate
    last_m2 = 2.335748e+14
    m2_vals = []
    for i, d in enumerate(dates):
        months_from_dec24 = (d.year - 2024) * 12 + d.month - 12
        m2_vals.append(last_m2 * (1 + 0.10/12) ** months_from_dec24)
    macro_in['M2'] = m2_vals

# India VIX
macro_in['India_VIX'] = india_vix.resample('ME').last()

# Shared columns from macro_us
shared_cols = ['US_CPI_YoY', 'Oil_Brent_USD', 'USDINR', 'USDINR_MoM_Return',
               'IN_Forex_Reserves_USD', 'US_Fed_Funds_Rate', 'US_10Y_Yield',
               'US_VIX', 'DXY_Dollar_Index']
for col in shared_cols:
    if col in macro_us_final.columns:
        macro_in[col] = macro_us_final[col]

# Reorder to match original india file
INDIA_COLS = ['CPI', 'PMI', 'Repo_Rate', 'M2', 'India_VIX', 'IN_CPI_YoY',
              'US_CPI_YoY', 'Oil_Brent_USD', 'USDINR', 'USDINR_MoM_Return',
              'IN_Forex_Reserves_USD', 'US_Fed_Funds_Rate', 'US_10Y_Yield',
              'US_VIX', 'DXY_Dollar_Index']

available_in = [c for c in INDIA_COLS if c in macro_in.columns]
macro_in_final = macro_in[available_in].loc['2025-01-01':'2026-02-28'].ffill()

macro_in_final.index.name = 'Date'
macro_in_final.to_csv('macro_india_2025_2026.csv')
print(f"\n  macro_india_2025_2026.csv saved")
print(f"  Shape: {macro_in_final.shape}")
print(f"  Columns: {macro_in_final.columns.tolist()}")
print(f"  Range: {macro_in_final.index[0].date()} -> {macro_in_final.index[-1].date()}")
print(f"\n  Last 3 rows:")
print(macro_in_final.tail(3).to_string())

# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SUMMARY — ALL FILES SAVED")
print("=" * 70)

# Check for missing columns
missing_us = [c for c in MACRO_US_COLS if c not in macro_us_final.columns]
missing_in = [c for c in INDIA_COLS if c not in macro_in_final.columns]

print(f"\n  1. data_daily_2025_2026.csv")
print(f"     {daily.shape[0]} trading days x {daily.shape[1]} assets")
print(f"     {daily.index[0].date()} -> {daily.index[-1].date()}")

print(f"\n  2. macro_finaldata_2025_2026.csv")
print(f"     {macro_us_final.shape[0]} months x {macro_us_final.shape[1]} columns")
if missing_us:
    print(f"     Missing columns: {missing_us}")
else:
    print(f"     All 24 columns present")

print(f"\n  3. macro_india_2025_2026.csv")
print(f"     {macro_in_final.shape[0]} months x {macro_in_final.shape[1]} columns")
if missing_in:
    print(f"     Missing columns: {missing_in}")
else:
    print(f"     All 15 columns present")

# Show null counts
print(f"\n  Null counts — macro_finaldata_2025_2026:")
nulls = macro_us_final.isnull().sum()
for c in nulls[nulls > 0].index:
    print(f"    {c}: {nulls[c]} nulls")
if nulls.sum() == 0:
    print(f"    No nulls")

print(f"\n  Null counts — macro_india_2025_2026:")
nulls_in = macro_in_final.isnull().sum()
for c in nulls_in[nulls_in > 0].index:
    print(f"    {c}: {nulls_in[c]} nulls")
if nulls_in.sum() == 0:
    print(f"    No nulls")

print("\nDONE.")
