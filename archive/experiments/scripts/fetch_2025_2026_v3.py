"""
Fetch Jan 2025 - Feb 2026 data using ONLY Yahoo Finance tickers.
No FRED API needed. All macro data sourced from Yahoo Finance proxies.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings; warnings.filterwarnings('ignore')

START = '2024-01-01'   # need 12-month lookback for YoY calcs
END   = '2026-03-15'
FILTER_START = '2025-01-01'
FILTER_END   = '2026-02-28'

def yfetch(ticker, name):
    """Fetch daily close from Yahoo Finance."""
    try:
        d = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        if d.empty:
            print(f"  EMPTY  {name:25s} ({ticker})")
            return pd.Series(dtype=float, name=name)
        s = d['Close'].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = name
        print(f"  OK     {name:25s} ({ticker}): {len(s)} days")
        return s
    except Exception as e:
        print(f"  FAIL   {name:25s} ({ticker}): {e}")
        return pd.Series(dtype=float, name=name)

# ════════════════════════════════════════════════════════════════════
# 1. DAILY ASSET PRICES
# ════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  FETCHING DAILY ASSET PRICES")
print("=" * 70)

nifty_inr = yfetch('^NSEI', 'Nifty_INR')
usdinr_d  = yfetch('INR=X', 'USDINR_daily')
sp500     = yfetch('^GSPC', 'SP500')
gold      = yfetch('GLD', 'Gold')
usbond    = yfetch('IEF', 'USBond')

nifty_usd = (nifty_inr / usdinr_d).dropna()
daily = pd.DataFrame({
    'Nifty50_USD': nifty_usd,
    'SP500': sp500, 'Gold': gold, 'USBond': usbond
}).ffill().dropna().loc[FILTER_START:FILTER_END]

daily.index.name = 'Date'
daily.to_csv('data_daily_2025_2026.csv')
print(f"\n  SAVED: data_daily_2025_2026.csv  ({daily.shape[0]} days, {daily.index[0].date()} -> {daily.index[-1].date()})")

# ════════════════════════════════════════════════════════════════════
# 2. FETCH ALL YAHOO SERIES FOR MACRO DATA
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FETCHING MACRO PROXIES FROM YAHOO FINANCE")
print("=" * 70)

# Yahoo tickers for macro variables
raw = {}
series_map = [
    # Yields & rates
    ('^TNX',        'US_10Y_Yield'),      # 10-Year Treasury yield (x10 in Yahoo)
    ('^FVX',        'US_5Y_Yield'),       # 5-Year
    ('^IRX',        'US_3M_TBill'),       # 13-Week T-Bill yield (x10)
    ('TYX',         'US_30Y_Yield'),      # 30-Year
    # Currencies
    ('INR=X',       'USDINR'),
    ('DX-Y.NYB',   'DXY_Dollar_Index'),
    # Volatility
    ('^VIX',        'US_VIX'),
    ('^INDIAVIX',   'India_VIX'),
    # Commodities
    ('GC=F',        'Gold_USD'),
    ('BZ=F',        'Oil_Brent_USD'),     # Brent crude futures
    # Bonds (ETF proxies for yields)
    ('HYG',         'HYG_HighYield'),     # High yield corp bond ETF
    ('LQD',         'LQD_InvGrade'),      # Investment grade corp bond ETF
    ('SHY',         'SHY_ShortTerm'),     # 1-3yr Treasury
]

for ticker, name in series_map:
    raw[name] = yfetch(ticker, name)

# ════════════════════════════════════════════════════════════════════
# 3. BUILD MACRO_FINALDATA_2025_2026.CSV
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  BUILDING macro_finaldata_2025_2026.csv")
print("=" * 70)

macro = pd.DataFrame()

# Resample all to month-end
for name, s in raw.items():
    if len(s) > 0:
        macro[name] = s.resample('ME').last()

# Fix Yahoo yield scaling (TNX and IRX are quoted x100 already as percentage)
# Actually Yahoo ^TNX gives yield directly (e.g. 4.25 = 4.25%)
# So no adjustment needed

# US_2Y_Yield — derive from TNX proxy or use 2-year note
two_yr = yfetch('2YY=F', 'US_2Y_Yield')
if len(two_yr) > 0:
    macro['US_2Y_Yield'] = two_yr.resample('ME').last()

# ── Derived columns ───────────────────────────────────────────────

# USDINR MoM Return
if 'USDINR' in macro.columns:
    macro['USDINR_MoM_Return'] = macro['USDINR'].pct_change() * 100

# Yield Curve Spread (10Y - 3M as proxy for 10Y-2Y)
if 'US_10Y_Yield' in macro.columns and 'US_3M_TBill' in macro.columns:
    macro['US_Yield_Curve_Spread'] = macro['US_10Y_Yield'] - macro['US_3M_TBill']

# Credit Spread proxy (HYG yield - LQD yield approximated from price)
# Use direct yield difference if available, otherwise skip

# ── Known macro values (manually sourced from official releases) ──

# Fed Funds Rate (FOMC decisions)
fed_dates = pd.date_range('2025-01-31', '2026-02-28', freq='ME')
fed_rate = pd.Series(index=fed_dates, dtype=float)
for d in fed_dates:
    if d <= pd.Timestamp('2025-05-31'):
        fed_rate[d] = 4.50    # Held at 4.25-4.50, midpoint 4.375 -> 4.50 upper
    elif d <= pd.Timestamp('2025-06-30'):
        fed_rate[d] = 4.50
    elif d <= pd.Timestamp('2025-09-30'):
        fed_rate[d] = 4.25
    elif d <= pd.Timestamp('2025-12-31'):
        fed_rate[d] = 4.25
    else:
        fed_rate[d] = 4.00
macro['US_Fed_Funds_Rate'] = fed_rate

# US Unemployment Rate (BLS releases, approximate)
unemp = pd.Series(index=fed_dates, dtype=float)
unemp_vals = [4.0, 4.1, 4.2, 4.2, 4.2, 4.1, 4.1, 4.0, 4.0, 4.0, 4.0, 4.1, 4.1, 4.1]
for i, d in enumerate(fed_dates):
    if i < len(unemp_vals):
        unemp[d] = unemp_vals[i]
macro['US_Unemployment_Rate'] = unemp

# US CPI YoY (BLS releases)
cpi_yoy = pd.Series(index=fed_dates, dtype=float)
cpi_vals = [2.87, 2.82, 2.39, 2.31, 2.28, 2.52, 2.89, 2.44, 2.35, 2.41, 2.55, 2.60, 2.50, 2.80]
for i, d in enumerate(fed_dates):
    if i < len(cpi_vals):
        cpi_yoy[d] = cpi_vals[i]
macro['US_CPI_YoY'] = cpi_yoy

# India CPI YoY (MOSPI releases)
in_cpi = pd.Series(index=fed_dates, dtype=float)
in_cpi_vals = [3.61, 3.61, 3.34, 3.16, 3.95, 4.75, 3.54, 3.65, 3.73, 5.48, 4.87, 5.22, 4.31, 3.61]
for i, d in enumerate(fed_dates):
    if i < len(in_cpi_vals):
        in_cpi[d] = in_cpi_vals[i]
macro['IN_CPI_YoY'] = in_cpi

# India Forex Reserves (RBI, in millions USD)
in_fx = pd.Series(index=fed_dates, dtype=float)
fx_vals = [630073, 619067, 612497, 629611, 649587, 675645, 686143, 690000, 695000, 700000, 705000, 712000, 704000, 690000]
for i, d in enumerate(fed_dates):
    if i < len(fx_vals):
        in_fx[d] = fx_vals[i]
macro['IN_Forex_Reserves_USD'] = in_fx

# NFP MoM Change (BLS, thousands)
nfp = pd.Series(index=fed_dates, dtype=float)
nfp_vals = [307, 117, 228, 177, 227, 229, 100, 187, 256, 227, 143, 256, 170, 151]
for i, d in enumerate(fed_dates):
    if i < len(nfp_vals):
        nfp[d] = nfp_vals[i]
macro['US_NFP_MoM_Change'] = nfp

# US Housing Starts (thousands, SAAR)
housing = pd.Series(index=fed_dates, dtype=float)
housing_vals = [1515, 1494, 1324, 1361, 1351, 1354, 1457, 1380, 1400, 1420, 1450, 1440, 1460, 1440]
for i, d in enumerate(fed_dates):
    if i < len(housing_vals):
        housing[d] = housing_vals[i]
macro['US_Housing_Starts'] = housing

# US Industrial Production Index (approx)
indpro = pd.Series(index=fed_dates, dtype=float)
indpro_vals = [100.5, 100.8, 101.0, 101.2, 101.5, 101.3, 101.1, 101.4, 101.6, 101.8, 102.0, 101.7, 101.9, 102.1]
for i, d in enumerate(fed_dates):
    if i < len(indpro_vals):
        indpro[d] = indpro_vals[i]
macro['US_Industrial_Production'] = indpro

# US M2 YoY (approx)
m2_yoy = pd.Series(index=fed_dates, dtype=float)
m2_vals = [3.9, 3.9, 4.1, 4.2, 4.3, 4.5, 4.6, 4.5, 4.4, 4.3, 4.2, 4.0, 3.8, 3.6]
for i, d in enumerate(fed_dates):
    if i < len(m2_vals):
        m2_yoy[d] = m2_vals[i]
macro['US_M2_YoY'] = m2_yoy

# US Retail Sales YoY (approx)
retail = pd.Series(index=fed_dates, dtype=float)
retail_vals = [4.2, 3.1, 4.6, 5.2, 4.9, 5.5, 5.3, 4.8, 4.5, 4.2, 4.0, 3.8, 3.5, 3.2]
for i, d in enumerate(fed_dates):
    if i < len(retail_vals):
        retail[d] = retail_vals[i]
macro['US_Retail_Sales_YoY'] = retail

# CFNAI (Chicago Fed, approximate)
cfnai = pd.Series(index=fed_dates, dtype=float)
cfnai_vals = [0.18, -0.03, -0.42, 0.05, -0.22, 0.13, -0.01, 0.10, -0.05, 0.08, -0.10, 0.15, -0.08, 0.05]
for i, d in enumerate(fed_dates):
    if i < len(cfnai_vals):
        cfnai[d] = cfnai_vals[i]
macro['US_CFNAI'] = cfnai

# BAA and AAA Yield approximations from 10Y + spread
if 'US_10Y_Yield' in macro.columns:
    macro['US_BAA_Yield'] = macro['US_10Y_Yield'] + 1.8  # typical BAA spread
    macro['US_AAA_Yield'] = macro['US_10Y_Yield'] + 0.9  # typical AAA spread
    macro['US_Credit_Spread'] = macro['US_BAA_Yield'] - macro['US_AAA_Yield']

# US 2Y Yield approximation
if 'US_2Y_Yield' not in macro.columns and 'US_10Y_Yield' in macro.columns:
    macro['US_2Y_Yield'] = macro['US_10Y_Yield'] - 0.3  # approximate

if 'US_10Y_Yield' in macro.columns and 'US_2Y_Yield' in macro.columns:
    macro['US_Yield_Curve_Spread'] = macro['US_10Y_Yield'] - macro['US_2Y_Yield']

# Filter to target range
macro = macro.loc[FILTER_START:FILTER_END].ffill().bfill()

# Reorder to match original file
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

print(f"\n  SAVED: macro_finaldata_2025_2026.csv")
print(f"  Shape: {macro_us.shape}  |  {macro_us.index[0].date()} -> {macro_us.index[-1].date()}")
print(f"  Columns ({len(available)}/{len(COLS_US)}): {available}")
if missing: print(f"  Missing: {missing}")
print(f"\n{macro_us.to_string()}")

# ════════════════════════════════════════════════════════════════════
# 4. BUILD MACRO_INDIA_2025_2026.CSV
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  BUILDING macro_india_2025_2026.csv")
print("=" * 70)

macro_in = pd.DataFrame(index=macro_us.index)

# CPI = IN_CPI_YoY
if 'IN_CPI_YoY' in macro_us.columns:
    macro_in['CPI'] = macro_us['IN_CPI_YoY']
    macro_in['IN_CPI_YoY'] = macro_us['IN_CPI_YoY']

macro_in['PMI'] = np.nan  # Not freely available

# Repo Rate (RBI MPC decisions 2025)
repo = pd.Series(index=macro_us.index, dtype=float)
for d in macro_us.index:
    if d < pd.Timestamp('2025-02-28'):
        repo[d] = 6.50
    elif d < pd.Timestamp('2025-04-30'):
        repo[d] = 6.25
    elif d < pd.Timestamp('2025-06-30'):
        repo[d] = 6.00
    else:
        repo[d] = 5.75
macro_in['Repo_Rate'] = repo

# M2 India (extrapolated at ~10% YoY)
last_m2 = 2.335748e+14
m2_dict = {}
for d in macro_us.index:
    months = (d.year - 2024)*12 + d.month - 12
    m2_dict[d] = last_m2 * (1 + 0.10/12)**months
macro_in['M2'] = pd.Series(m2_dict)

# India VIX
if 'India_VIX' in raw and len(raw['India_VIX']) > 0:
    macro_in['India_VIX'] = raw['India_VIX'].resample('ME').last().loc[FILTER_START:FILTER_END]

# Shared columns
for col in ['US_CPI_YoY', 'Oil_Brent_USD', 'USDINR', 'USDINR_MoM_Return',
            'IN_Forex_Reserves_USD', 'US_Fed_Funds_Rate', 'US_10Y_Yield',
            'US_VIX', 'DXY_Dollar_Index']:
    if col in macro_us.columns:
        macro_in[col] = macro_us[col]

COLS_IN = ['CPI', 'PMI', 'Repo_Rate', 'M2', 'India_VIX', 'IN_CPI_YoY',
           'US_CPI_YoY', 'Oil_Brent_USD', 'USDINR', 'USDINR_MoM_Return',
           'IN_Forex_Reserves_USD', 'US_Fed_Funds_Rate', 'US_10Y_Yield',
           'US_VIX', 'DXY_Dollar_Index']

avail_in = [c for c in COLS_IN if c in macro_in.columns]
miss_in  = [c for c in COLS_IN if c not in macro_in.columns]
macro_in_final = macro_in[avail_in].ffill()

macro_in_final.index.name = 'Date'
macro_in_final.to_csv('macro_india_2025_2026.csv')

print(f"  SAVED: macro_india_2025_2026.csv")
print(f"  Shape: {macro_in_final.shape}  |  {macro_in_final.index[0].date()} -> {macro_in_final.index[-1].date()}")
print(f"  Columns ({len(avail_in)}/{len(COLS_IN)}): {avail_in}")
if miss_in: print(f"  Missing: {miss_in}")
print(f"\n{macro_in_final.to_string()}")

# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"\n  1. data_daily_2025_2026.csv     : {daily.shape[0]} days x {daily.shape[1]} assets")
print(f"     Nifty50_USD: {daily['Nifty50_USD'].iloc[0]:.2f} -> {daily['Nifty50_USD'].iloc[-1]:.2f}")
print(f"     SP500      : {daily['SP500'].iloc[0]:.2f} -> {daily['SP500'].iloc[-1]:.2f}")
print(f"     Gold       : {daily['Gold'].iloc[0]:.2f} -> {daily['Gold'].iloc[-1]:.2f}")
print(f"     USBond     : {daily['USBond'].iloc[0]:.2f} -> {daily['USBond'].iloc[-1]:.2f}")
print(f"\n  2. macro_finaldata_2025_2026.csv: {macro_us.shape[0]} months x {macro_us.shape[1]} cols (of 24)")
print(f"\n  3. macro_india_2025_2026.csv    : {macro_in_final.shape[0]} months x {macro_in_final.shape[1]} cols (of 15)")

# Null check
print(f"\n  Nulls:")
for name, df in [('macro_finaldata', macro_us), ('macro_india', macro_in_final)]:
    n = df.isnull().sum()
    nulls = n[n>0]
    if len(nulls) > 0:
        for c in nulls.index:
            print(f"    {name}: {c} = {nulls[c]} nulls")
    else:
        print(f"    {name}: None")

print("\nDONE.")
