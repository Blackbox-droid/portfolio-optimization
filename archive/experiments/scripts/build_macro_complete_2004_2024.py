import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

FETCH_START = '2003-01-01'
OUT_START   = '2004-01-31'
OUT_END     = '2024-12-31'

print("=" * 65)
print("  MACRO DATA BUILDER  --  Jan 2004 to Dec 2024")
print("=" * 65)

def fred(series_id, name, start=FETCH_START, end=OUT_END):
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )
    try:
        df  = pd.read_csv(url, na_values=['.', '', 'NA'])
        dc  = [c for c in df.columns if 'date' in c.lower()][0]
        df[dc] = pd.to_datetime(df[dc])
        s   = df.set_index(dc).iloc[:, 0].astype(float)
        s   = s.resample('ME').last()
        s.name = name
        n   = s.dropna().shape[0]
        print(f"  [OK]   {name:<42s} {n:3d} obs  ({series_id})")
        return s
    except Exception as e:
        print(f"  [FAIL] {name:<42s} {e}  ({series_id})")
        return pd.Series(dtype=float, name=name)

def yf_dl(ticker, name, start=FETCH_START, end=OUT_END, field='Close'):
    try:
        raw = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False
        )
        if raw.empty:
            raise ValueError("empty download")
        s   = raw[field].squeeze().resample('ME').last()
        s.name = name
        n   = s.dropna().shape[0]
        print(f"  [OK]   {name:<42s} {n:3d} obs  ({ticker})")
        return s
    except Exception as e:
        print(f"  [FAIL] {name:<42s} {e}  ({ticker})")
        return pd.Series(dtype=float, name=name)

def yoy(level_series, out_name):
    s = level_series.pct_change(12) * 100
    s.name = out_name
    return s

def mom_diff(level_series, out_name):
    s = level_series.diff()
    s.name = out_name
    return s

def mom_ret(level_series, out_name):
    s = level_series.pct_change() * 100
    s.name = out_name
    return s

def to_me(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample('ME').last()

# ================================================================
# SECTION A -- INFLATION & PRICES
# ================================================================
print("\n-- A: INFLATION & PRICES ---")
_in_cpi_idx  = fred('INDCPIALLMINMEI', 'IN_CPI_Index_raw')
IN_CPI_YoY   = yoy(_in_cpi_idx, 'IN_CPI_YoY')

_us_cpi_idx  = fred('CPIAUCSL',        'US_CPI_Index_raw')
US_CPI_YoY   = yoy(_us_cpi_idx, 'US_CPI_YoY')

Oil_Brent    = fred('DCOILBRENTEU',     'Oil_Brent_USD')
# Gold: FRED series removed Jan 2022 -- use Yahoo Finance GC=F
Gold_USD     = yf_dl('GC=F',            'Gold_USD')

# ================================================================
# SECTION B -- GROWTH & ACTIVITY
# ================================================================
print("\n-- B: GROWTH & ACTIVITY ---")
_in_iip_idx  = fred('INDPROINDMISMEI', 'IN_IIP_Index_raw')
IN_IIP_YoY   = yoy(_in_iip_idx, 'IN_IIP_YoY')

IN_CLI       = fred('INDLORSGPNOSTSAM', 'IN_CLI_OECD')
IN_Biz_Conf  = fred('BSCICP03INM665S',  'IN_Business_Confidence')
# IN_Consumer_Confidence: FRED series CSCICP03INM665S does not exist for India
# Use US consumer confidence as proxy, or skip
# Skipping -- not available on FRED for India
print(f"  [SKIP] IN_Consumer_Confidence              not available on FRED for India")
IN_Con_Conf  = pd.Series(dtype=float, name='IN_Consumer_Confidence')

US_IIP       = fred('INDPRO',           'US_Industrial_Production')
US_CFNAI     = fred('CFNAI',            'US_CFNAI')

# ================================================================
# SECTION C -- MONETARY POLICY & CREDIT
# ================================================================
print("\n-- C: MONETARY POLICY & CREDIT ---")
IN_Repo      = fred('IRSTCB01INM156N',  'IN_Repo_Rate')
_in_m3_idx   = fred('MABMM301INM189N',  'IN_M3_Index_raw')
IN_M3_YoY    = yoy(_in_m3_idx, 'IN_M3_YoY')

# India 10Y: try FRED first, fallback to Yahoo
IN_10Y       = fred('INDIRLTLT01STM',   'IN_10Y_Yield')
if IN_10Y.dropna().shape[0] == 0:
    IN_10Y   = yf_dl('IN10YT=X',        'IN_10Y_Yield')

US_FFR       = fred('FEDFUNDS',          'US_Fed_Funds_Rate')
_us_m2_idx   = fred('M2SL',             'US_M2_Index_raw')
US_M2_YoY    = yoy(_us_m2_idx, 'US_M2_YoY')
US_10Y       = fred('DGS10',             'US_10Y_Yield')
US_2Y        = fred('DGS2',              'US_2Y_Yield')
US_BAA       = fred('BAA',               'US_BAA_Yield')
US_AAA       = fred('AAA',               'US_AAA_Yield')
US_TBill     = fred('DTB3',              'US_3M_TBill')

# ================================================================
# SECTION D -- LABOR MARKET
# ================================================================
print("\n-- D: LABOR MARKET ---")
US_Unemp     = fred('UNRATE',            'US_Unemployment_Rate')
_us_nfp      = fred('PAYEMS',            'US_NFP_Level_raw')
US_NFP_Ch    = mom_diff(_us_nfp, 'US_NFP_MoM_Change')

# ================================================================
# SECTION E -- EXTERNAL SECTOR
# ================================================================
print("\n-- E: EXTERNAL SECTOR ---")
USDINR       = yf_dl('INR=X',           'USDINR')
# India Forex Reserves: use TRESEGINM052N (Total Reserves ex Gold, IMF)
IN_Forex     = fred('TRESEGINM052N',     'IN_Forex_Reserves_USD')
DXY          = yf_dl('DX-Y.NYB',        'DXY_Dollar_Index')

# ================================================================
# SECTION F -- FINANCIAL / MARKET-BASED
# ================================================================
print("\n-- F: FINANCIAL / MARKET-BASED ---")
IN_VIX       = yf_dl('^INDIAVIX',       'IN_VIX')
_nifty       = yf_dl('^NSEI',           'Nifty50_Level_raw')
Nifty_Ret    = mom_ret(_nifty,           'Nifty50_MoM_Return')
Nifty_12M    = (_nifty.pct_change(12) * 100)
Nifty_12M.name = 'Nifty50_12M_Momentum'
_sp500       = yf_dl('^GSPC',           'SP500_Level_raw')
SP500_Ret    = mom_ret(_sp500,           'SP500_MoM_Return')
_gold_f      = yf_dl('GC=F',            'Gold_Futures_raw')
Gold_Ret     = mom_ret(_gold_f,          'Gold_MoM_Return')
_bnifty      = yf_dl('^NSEBANK',        'BankNifty_Level_raw')
BNifty_Ret   = mom_ret(_bnifty,          'BankNifty_MoM_Return')

# ================================================================
# SECTION G -- US / GLOBAL CONTEXT
# ================================================================
print("\n-- G: US / GLOBAL CONTEXT ---")
US_VIX       = yf_dl('^VIX',            'US_VIX')
US_Housing   = fred('HOUST',             'US_Housing_Starts')
_us_retail   = fred('RSAFS',             'US_Retail_Sales_raw')
US_Retail_YoY = yoy(_us_retail, 'US_Retail_Sales_YoY')

# ================================================================
# ASSEMBLY
# ================================================================
print("\n-- ASSEMBLING ALL SERIES ---")
primary_series = [
    IN_CPI_YoY, US_CPI_YoY, Oil_Brent, Gold_USD,
    IN_IIP_YoY, IN_CLI, IN_Biz_Conf, IN_Con_Conf,
    US_IIP, US_CFNAI,
    IN_Repo, IN_M3_YoY, IN_10Y, US_FFR, US_M2_YoY,
    US_10Y, US_2Y, US_BAA, US_AAA, US_TBill,
    US_Unemp, US_NFP_Ch,
    USDINR, IN_Forex, DXY,
    IN_VIX, Nifty_Ret, Nifty_12M, SP500_Ret, Gold_Ret, BNifty_Ret,
    US_VIX, US_Housing, US_Retail_YoY,
]

resampled = [to_me(s) for s in primary_series]
macro     = pd.concat(resampled, axis=1)
macro = macro.loc[OUT_START : OUT_END]
print(f"  Base shape after clip: {macro.shape}")

# ================================================================
# DERIVED VARIABLES
# ================================================================
print("\n-- DERIVED VARIABLES ---")
macro['US_Yield_Curve_Spread'] = macro['US_10Y_Yield'] - macro['US_2Y_Yield']
print("  [OK]   US_Yield_Curve_Spread  (10Y - 2Y)")
macro['US_Credit_Spread'] = macro['US_BAA_Yield'] - macro['US_AAA_Yield']
print("  [OK]   US_Credit_Spread  (BAA - AAA)")
macro['IN_Real_Repo_Rate'] = macro['IN_Repo_Rate'] - macro['IN_CPI_YoY']
print("  [OK]   IN_Real_Repo_Rate  (Repo - CPI)")
macro['IN_VIX_MoM_Change'] = macro['IN_VIX'].diff()
print("  [OK]   IN_VIX_MoM_Change")
macro['USDINR_MoM_Return'] = macro['USDINR'].pct_change() * 100
print("  [OK]   USDINR_MoM_Return")
macro['Nifty_vs_SP500_Spread'] = macro['Nifty50_MoM_Return'] - macro['SP500_MoM_Return']
print("  [OK]   Nifty_vs_SP500_Spread")

# ================================================================
# SMART GAP FILLING
# ================================================================
print("\n-- SMART GAP FILLING ---")
ffill_3 = ['IN_Repo_Rate', 'US_Fed_Funds_Rate']
for col in ffill_3:
    if col in macro.columns:
        before = macro[col].isna().sum()
        macro[col] = macro[col].ffill(limit=3)
        after = macro[col].isna().sum()
        if before > after:
            print(f"  ffill(3)  {col}: {before} -> {after} NaNs")

ffill_2 = [
    'IN_CLI_OECD', 'IN_Business_Confidence',
    'IN_Consumer_Confidence', 'IN_IIP_YoY',
    'IN_M3_YoY', 'IN_10Y_Yield',
]
for col in ffill_2:
    if col in macro.columns:
        before = macro[col].isna().sum()
        macro[col] = macro[col].ffill(limit=2)
        after = macro[col].isna().sum()
        if before > after:
            print(f"  ffill(2)  {col}: {before} -> {after} NaNs")

# ================================================================
# COVERAGE REPORT
# ================================================================
print("\n" + "=" * 70)
print("  FINAL COVERAGE REPORT")
print("=" * 70)
TOTAL = len(macro)
print(f"\n  Period     : {macro.index[0].date()}  ->  {macro.index[-1].date()}")
print(f"  Total rows : {TOTAL}  (expected 252 for Jan2004-Dec2024)")
print(f"  Columns    : {macro.shape[1]}")

header = f"\n  {'Variable':<35} {'Valid':>6}  {'NaN':>5}  {'Pct':>7}  {'Start':>8}  {'End':>8}  Status"
print(header)
print("  " + "-" * 75)

full_vars    = []
partial_vars = []
missing_vars = []

for col in macro.columns:
    s      = macro[col].dropna()
    n_val  = len(s)
    n_nan  = TOTAL - n_val
    pct    = n_val / TOTAL * 100
    fv     = s.index[0].strftime('%Y-%m') if n_val > 0 else 'N/A'
    lv     = s.index[-1].strftime('%Y-%m') if n_val > 0 else 'N/A'
    if n_val == 0:
        status = 'X NO DATA'
        missing_vars.append(col)
    elif pct >= 98 and fv <= '2004-03':
        status = 'FULL'
        full_vars.append(col)
    elif pct >= 80:
        status = '~ PARTIAL'
        partial_vars.append(col)
    else:
        status = 'X SPARSE'
        partial_vars.append(col)
    print(f"  {col:<35} {n_val:>6}  {n_nan:>5}  {pct:>6.1f}%  {fv:>8}  {lv:>8}  {status}")

print(f"\n  {'-'*70}")
print(f"  Full coverage  : {len(full_vars)}")
print(f"  Partial        : {len(partial_vars)}")
print(f"  No data        : {len(missing_vars)}")

if partial_vars:
    print(f"\n  PARTIAL variables (still usable with NaN handling):")
    for v in partial_vars:
        s  = macro[v].dropna()
        fv = s.index[0].strftime('%Y-%m') if len(s)>0 else 'N/A'
        lv = s.index[-1].strftime('%Y-%m') if len(s)>0 else 'N/A'
        print(f"    - {v:<40s} {fv} -> {lv}")

# ================================================================
# SAVE
# ================================================================
macro.index.name = 'Date'
OUT_FILE = 'macro_complete_2004_2024.csv'
macro.to_csv(OUT_FILE, float_format='%.6f')

print(f"\n{'='*70}")
print(f"  SAVED: {OUT_FILE}")
print(f"  Shape: {macro.shape[0]} rows  x  {macro.shape[1]} columns")
print(f"  Period: {macro.index[0].date()}  ->  {macro.index[-1].date()}")
print(f"{'='*70}")

# Print column list
print("\n  ALL COLUMNS:")
for i, col in enumerate(macro.columns, 1):
    print(f"  {i:2d}. {col}")
print("\n  DONE")
