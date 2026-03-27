import os
import sys
import io
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

OUTPUT_FOLDER = r"."
START_BUFFER = '2003-01-01'
START        = '2005-01-01'
END          = '2024-12-31'
EXPECTED     = 240

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

log_lines = []
def log(msg=''):
    print(msg)
    log_lines.append(msg)

def fetch_fred(series_id, col_name, start=START_BUFFER, end=END, retries=3):
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), na_values=['.', '', 'NA'])
            date_col = [c for c in df.columns if 'date' in c.lower()]
            if not date_col:
                raise ValueError(f"No date column. Cols: {df.columns.tolist()}")
            df[date_col[0]] = pd.to_datetime(df[date_col[0]])
            df = df.set_index(date_col[0])
            s  = df.iloc[:, 0].astype(float)
            s.name = col_name
            return s
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                log(f"  [FAIL] {col_name:38s} -- {str(e)[:60]}  (FRED:{series_id})")
                return pd.Series(dtype=float, name=col_name)

def fetch_yf(ticker, col_name, field='Close', retries=3):
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker,
                start=START_BUFFER,
                end=END,
                auto_adjust=True,
                progress=False
            )
            if raw.empty:
                raise ValueError("Empty response from Yahoo Finance")
            s = raw[field].squeeze()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.resample('ME').last()
            s.name = col_name
            return s
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                log(f"  [FAIL] {col_name:38s} -- {str(e)[:60]}  (Yahoo:{ticker})")
                return pd.Series(dtype=float, name=col_name)

def to_monthly(s, agg='last'):
    if s is None or (isinstance(s, pd.Series) and s.empty):
        return s
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if agg == 'last':
        s = s.resample('ME').last()
    elif agg == 'mean':
        s = s.resample('ME').mean()
    s = s.loc[START:END]
    return s

def yoy(s):
    result = s.pct_change(12) * 100
    result.name = s.name
    return result

def mom(s):
    result = s.pct_change(1) * 100
    result.name = s.name
    return result

log("=" * 65)
log("  MACRO DATA DOWNLOAD  |  35 Variables  |  Jan 2005 - Dec 2024")
log("=" * 65)

series = {}

log("\n--- GROUP A: INFLATION & PRICES ---")

_s = fetch_fred('INDCPIALLMINMEI', 'IN_CPI_Index')
_s = to_monthly(_s)
series['IN_CPI_YoY'] = yoy(_s)
series['IN_CPI_YoY'].name = 'IN_CPI_YoY'

# IN_Core_CPI and IN_Food_CPI: not available on FRED for India -- skipped

_s = fetch_fred('CPIAUCSL', 'US_CPI_Index')
_s = to_monthly(_s)
series['US_CPI_YoY'] = yoy(_s)
series['US_CPI_YoY'].name = 'US_CPI_YoY'

_s = fetch_fred('DCOILBRENTEU', 'Oil_Brent_USD')
_s = to_monthly(_s, agg='mean')
series['Oil_Brent_USD'] = _s

# Gold USD from Yahoo Finance (FRED series removed Jan 2022)
_s = fetch_yf('GC=F', 'Gold_USD')
_s = to_monthly(_s, agg='mean')
series['Gold_USD'] = _s

log(f"  Variables defined: {[k for k in series]}")

log("\n--- GROUP B: GROWTH & ACTIVITY ---")

_s = fetch_fred('INDPROINDMISMEI', 'IN_IIP_YoY')
_s = to_monthly(_s)
series['IN_IIP_YoY'] = _s

_s = fetch_fred('INDLORSGPNOSTSAM', 'IN_CLI_OECD')
_s = to_monthly(_s)
series['IN_CLI_OECD'] = _s

_s = fetch_fred('BSCICP03INM665S', 'IN_Business_Confidence')
_s = to_monthly(_s)
series['IN_Business_Confidence'] = _s

# IN_Consumer_Confidence: not available on FRED for India -- skipped

_s = fetch_fred('INDPRO', 'US_Industrial_Production')
_s = to_monthly(_s)
series['US_Industrial_Production'] = _s

_s = fetch_fred('CFNAI', 'US_CFNAI')
_s = to_monthly(_s)
series['US_CFNAI'] = _s

_s = fetch_fred('USSLIND', 'US_Leading_Index')
_s = to_monthly(_s)
series['US_Leading_Index'] = _s

log(f"  Variables defined so far: {len(series)}")

log("\n--- GROUP C: MONETARY POLICY & CREDIT ---")

_s = fetch_fred('IRSTCB01INM156N', 'IN_Repo_Rate')
_s = to_monthly(_s)
series['IN_Repo_Rate'] = _s

# India 10Y yield: try FRED first, fallback to Yahoo Finance
_s = fetch_fred('INDIRLTLT01STM', 'IN_10Y_Yield')
if _s is not None and not _s.empty:
    _s = to_monthly(_s)
else:
    _s = fetch_yf('IN10YT=X', 'IN_10Y_Yield')
    _s = to_monthly(_s)
series['IN_10Y_Yield'] = _s

_s = fetch_fred('FEDFUNDS', 'US_Fed_Funds_Rate')
_s = to_monthly(_s)
series['US_Fed_Funds_Rate'] = _s

_s = fetch_fred('M2SL', 'US_M2_Level')
_s = to_monthly(_s)
series['US_M2_YoY'] = yoy(_s)
series['US_M2_YoY'].name = 'US_M2_YoY'

_s = fetch_fred('DGS10', 'US_10Y_Yield')
_s = to_monthly(_s, agg='mean')
series['US_10Y_Yield'] = _s

_s = fetch_fred('DGS2', 'US_2Y_Yield')
_s = to_monthly(_s, agg='mean')
series['US_2Y_Yield'] = _s

_s = fetch_fred('BAA', 'US_BAA_Yield')
_s = to_monthly(_s, agg='mean')
series['US_BAA_Yield'] = _s

_s = fetch_fred('AAA', 'US_AAA_Yield')
_s = to_monthly(_s, agg='mean')
series['US_AAA_Yield'] = _s

_s = fetch_fred('DTB3', 'US_3M_TBill')
_s = to_monthly(_s, agg='mean')
series['US_3M_TBill'] = _s

log(f"  Variables defined so far: {len(series)}")

log("\n--- GROUP D: LABOR MARKET ---")

# IN_Unemployment_Rate: not available on FRED for India -- skipped

_s = fetch_fred('UNRATE', 'US_Unemployment_Rate')
_s = to_monthly(_s)
series['US_Unemployment_Rate'] = _s

_s = fetch_fred('PAYEMS', 'US_NFP_Level')
_s = to_monthly(_s)
series['US_NFP_MoM_Change'] = _s.diff()
series['US_NFP_MoM_Change'].name = 'US_NFP_MoM_Change'

log(f"  Variables defined so far: {len(series)}")

log("\n--- GROUP E: EXTERNAL SECTOR ---")

_s = fetch_yf('INR=X', 'USDINR')
_s = to_monthly(_s)
series['USDINR'] = _s

_s = fetch_fred('TRESEGINM052N', 'IN_Forex_Reserves_USD')
_s = to_monthly(_s)
series['IN_Forex_Reserves_USD'] = _s

_s = fetch_yf('DX-Y.NYB', 'DXY_Dollar_Index')
_s = to_monthly(_s)
series['DXY_Dollar_Index'] = _s

log(f"  Variables defined so far: {len(series)}")

log("\n--- GROUP F: FINANCIAL / MARKET-BASED ---")

_s = fetch_yf('^NSEI', 'Nifty50_Level')
_s = to_monthly(_s)
series['Nifty50_MoM_Return'] = mom(_s)
series['Nifty50_MoM_Return'].name = 'Nifty50_MoM_Return'

_s = fetch_yf('^GSPC', 'SP500_Level')
_s = to_monthly(_s)
series['SP500_MoM_Return'] = mom(_s)
series['SP500_MoM_Return'].name = 'SP500_MoM_Return'

_s = fetch_yf('GC=F', 'Gold_Futures_Level')
_s = to_monthly(_s)
series['Gold_MoM_Return'] = mom(_s)
series['Gold_MoM_Return'].name = 'Gold_MoM_Return'

_s = fetch_yf('^NSEBANK', 'BankNifty_Level')
_s = to_monthly(_s)
series['BankNifty_MoM_Return'] = mom(_s)
series['BankNifty_MoM_Return'].name = 'BankNifty_MoM_Return'

_s = fetch_yf('^VIX', 'US_VIX')
_s = to_monthly(_s)
series['US_VIX'] = _s

log(f"  Variables defined so far: {len(series)}")

log("\n--- GROUP G: US / GLOBAL MACRO CONTEXT ---")

_s = fetch_fred('HOUST', 'US_Housing_Starts')
_s = to_monthly(_s)
series['US_Housing_Starts'] = _s

_s = fetch_fred('RSAFS', 'US_Retail_Sales_Level')
_s = to_monthly(_s)
series['US_Retail_Sales_YoY'] = yoy(_s)
series['US_Retail_Sales_YoY'].name = 'US_Retail_Sales_YoY'

log(f"  Variables defined so far: {len(series)}")

log("\n" + "=" * 65)
log("  ASSEMBLING DATAFRAME")
log("=" * 65)

frames = []
for col_name, s in series.items():
    if s is None or s.empty:
        frames.append(pd.Series(dtype=float, name=col_name,
                                index=pd.date_range(START, END, freq='ME')))
        continue
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s.resample('ME').last()
    s = s.loc[START:END]
    s.name = col_name
    frames.append(s)

macro = pd.concat(frames, axis=1)
macro.index.name = 'Date'

log("\nComputing derived variables...")

if 'US_10Y_Yield' in macro.columns and 'US_2Y_Yield' in macro.columns:
    macro['US_Yield_Curve_Spread'] = macro['US_10Y_Yield'] - macro['US_2Y_Yield']
    log("  [+] US_Yield_Curve_Spread  = US_10Y - US_2Y")

if 'US_BAA_Yield' in macro.columns and 'US_AAA_Yield' in macro.columns:
    macro['US_Credit_Spread'] = macro['US_BAA_Yield'] - macro['US_AAA_Yield']
    log("  [+] US_Credit_Spread       = BAA - AAA")

if 'IN_Repo_Rate' in macro.columns and 'IN_CPI_YoY' in macro.columns:
    macro['IN_Real_Repo_Rate'] = macro['IN_Repo_Rate'] - macro['IN_CPI_YoY']
    log("  [+] IN_Real_Repo_Rate      = Repo - CPI_YoY")

log("\n" + "=" * 65)
log("  MISSING DATA HANDLING")
log("=" * 65)

log(f"\nShape before fill: {macro.shape}")
log(f"Total NaN before fill: {macro.isnull().sum().sum()}")

macro_filled = macro.ffill(limit=3)
log(f"Total NaN after ffill(3): {macro_filled.isnull().sum().sum()}")

log("\n" + "=" * 65)
log("  VARIABLE COVERAGE REPORT")
log("=" * 65)

log(f"\n{'Variable':<35} {'Valid':>6} {'Missing':>8} {'Coverage':>10}  {'First':>8}  {'Last':>8}  {'Status':>8}")
log("-" * 95)

full_count    = 0
partial_count = 0
fail_count    = 0

for col in macro_filled.columns:
    s        = macro_filled[col].dropna()
    valid    = len(s)
    missing  = EXPECTED - valid
    pct      = valid / EXPECTED * 100
    first    = s.index[0].strftime('%Y-%m') if valid > 0 else 'N/A'
    last     = s.index[-1].strftime('%Y-%m') if valid > 0 else 'N/A'
    if valid >= EXPECTED:
        status = 'FULL'
        full_count += 1
    elif valid >= EXPECTED * 0.8:
        status = 'PARTIAL'
        partial_count += 1
    elif valid > 0:
        status = 'LIMITED'
        partial_count += 1
    else:
        status = 'EMPTY'
        fail_count += 1
    log(f"  {col:<33} {valid:>6} {missing:>8} {pct:>9.1f}%  {first:>8}  {last:>8}  {status:>8}")

log("-" * 95)
log(f"\n  FULL     (>=240 months): {full_count}")
log(f"  PARTIAL  (>=192 months): {partial_count}")
log(f"  EMPTY    (0 months)    : {fail_count}")
log(f"  TOTAL    columns       : {len(macro_filled.columns)}")

log("\n" + "=" * 65)
log("  DESCRIPTIVE STATISTICS")
log("=" * 65)

log("\n" + macro_filled.describe().round(3).to_string())

log("\n" + "=" * 65)
log("  SAVING FILES")
log("=" * 65)

path_all = os.path.join(OUTPUT_FOLDER, 'macro_35_variables.csv')
macro_filled.to_csv(path_all, float_format='%.6f')
log(f"\n  [SAVED] {path_all}")
log(f"          Rows    : {len(macro_filled)}")
log(f"          Columns : {len(macro_filled.columns)}")
log(f"          Period  : {macro_filled.index[0].date()} -> {macro_filled.index[-1].date()}")

india_cols = [
    'IN_CPI_YoY',
    'IN_Core_CPI_YoY',
    'IN_IIP_YoY',
    'IN_CLI_OECD',
    'IN_Business_Confidence',
    'IN_Repo_Rate',
    'IN_Real_Repo_Rate',
    'IN_10Y_Yield',
    'US_10Y_Yield',
    'US_Yield_Curve_Spread',
    'US_Credit_Spread',
    'USDINR',
    'Oil_Brent_USD',
    'US_VIX',
    'Nifty50_MoM_Return',
    'SP500_MoM_Return',
]

india_available = [c for c in india_cols if c in macro_filled.columns]
macro_india = macro_filled[india_available].copy()
path_india = os.path.join(OUTPUT_FOLDER, 'macro_india_focused.csv')
macro_india.to_csv(path_india, float_format='%.6f')
log(f"\n  [SAVED] {path_india}")
log(f"          Rows    : {len(macro_india)}")
log(f"          Columns : {len(macro_india.columns)}")

path_log = os.path.join(OUTPUT_FOLDER, 'macro_35_variables_log.txt')
with open(path_log, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"\n  [SAVED] {path_log}")

log("\n" + "=" * 65)
log("  ALL COLUMNS IN macro_35_variables.csv")
log("=" * 65)

for i, col in enumerate(macro_filled.columns, 1):
    log(f"  {i:2d}. {col}")

log("\n" + "=" * 65)
log("  DONE")
log("=" * 65)
