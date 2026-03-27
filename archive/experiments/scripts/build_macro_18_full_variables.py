import os
import sys
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

OUTPUT_FOLDER = r"."
START_BUFFER = '2004-01-01'
START        = '2005-01-01'
END          = '2024-12-31'
EXPECTED     = 240

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def fetch_fred(series_id, col_name, agg='last', retries=3):
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={START_BUFFER}&coed={END}"
    )
    for attempt in range(retries):
        try:
            from io import StringIO
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), na_values=['.', '', 'NA'])
            date_col = [c for c in df.columns if 'date' in c.lower()][0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            s  = df.iloc[:, 0].astype(float)
            s.name = col_name
            if agg == 'mean':
                s = s.resample('ME').mean()
            else:
                s = s.resample('ME').last()
            s = s.loc[START:END]
            valid = s.dropna().shape[0]
            print(f"  [OK]   {col_name:<35}  {valid:3d}/240  (FRED:{series_id})")
            return s
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  [FAIL] {col_name:<35}  -- {str(e)[:55]}  (FRED:{series_id})")
                return pd.Series(
                    dtype=float, name=col_name,
                    index=pd.date_range(START, END, freq='ME')
                )

def fetch_yf(ticker, col_name, agg='last', retries=3):
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
                raise ValueError("Empty response")
            s = raw['Close'].squeeze()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            if agg == 'mean':
                s = s.resample('ME').mean()
            else:
                s = s.resample('ME').last()
            s = s.loc[START:END]
            s.name = col_name
            valid = s.dropna().shape[0]
            print(f"  [OK]   {col_name:<35}  {valid:3d}/240  (Yahoo:{ticker})")
            return s
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  [FAIL] {col_name:<35}  -- {str(e)[:55]}  (Yahoo:{ticker})")
                return pd.Series(
                    dtype=float, name=col_name,
                    index=pd.date_range(START, END, freq='ME')
                )

print("=" * 65)
print("  DOWNLOADING 18 FULLY COVERED MACRO VARIABLES")
print("  Period : Jan 2005 - Dec 2024  (240 months)")
print("=" * 65)

print("\n--- COMMODITIES ---")
oil   = fetch_fred('DCOILBRENTEU',    'Oil_Brent_USD',            agg='mean')
gold  = fetch_yf('GC=F',             'Gold_USD',                  agg='mean')

print("\n--- US ACTIVITY ---")
us_ip = fetch_fred('INDPRO',          'US_Industrial_Production', agg='last')
cfnai = fetch_fred('CFNAI',           'US_CFNAI',                agg='last')

print("\n--- US MONETARY & RATES ---")
ffr   = fetch_fred('FEDFUNDS',        'US_Fed_Funds_Rate',       agg='last')
us10y = fetch_fred('DGS10',           'US_10Y_Yield',            agg='mean')
us2y  = fetch_fred('DGS2',            'US_2Y_Yield',             agg='mean')
baa   = fetch_fred('BAA',             'US_BAA_Yield',            agg='mean')
aaa   = fetch_fred('AAA',             'US_AAA_Yield',            agg='mean')
tbill = fetch_fred('DTB3',            'US_3M_TBill',             agg='mean')

print("\n--- US LABOR ---")
unemp = fetch_fred('UNRATE',          'US_Unemployment_Rate',    agg='last')

print("\n--- INDIA EXTERNAL ---")
forex = fetch_fred('TRESEGINM052N',   'IN_Forex_Reserves_USD',   agg='last')

print("\n--- US MACRO ---")
hous  = fetch_fred('HOUST',           'US_Housing_Starts',       agg='last')

print("\n--- YAHOO FINANCE ---")
usdinr = fetch_yf('INR=X',            'USDINR')
dxy    = fetch_yf('DX-Y.NYB',         'DXY_Dollar_Index')
vix    = fetch_yf('^VIX',             'US_VIX')

print("\n" + "=" * 65)
print("  ASSEMBLING DATAFRAME")
print("=" * 65)

raw_series = [
    oil, gold,
    us_ip, cfnai,
    ffr, us10y, us2y, baa, aaa, tbill,
    unemp,
    forex,
    hous,
    usdinr, dxy, vix
]

frames = []
for s in raw_series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s.resample('ME').last()
    s = s.loc[START:END]
    frames.append(s)

macro = pd.concat(frames, axis=1)
macro.index.name = 'Date'

print("\nComputing derived variables...")
macro['US_Yield_Curve_Spread'] = macro['US_10Y_Yield'] - macro['US_2Y_Yield']
print(f"  [+] US_Yield_Curve_Spread = US_10Y_Yield - US_2Y_Yield")
macro['US_Credit_Spread'] = macro['US_BAA_Yield'] - macro['US_AAA_Yield']
print(f"  [+] US_Credit_Spread      = US_BAA_Yield - US_AAA_Yield")

print("\n" + "=" * 65)
print("  COVERAGE REPORT")
print("=" * 65)
print(f"\n{'#':>3}  {'Variable':<35} {'Valid':>6} {'Missing':>8} {'Coverage':>9}  Status")
print("-" * 72)

full_list = []
fail_list = []

for i, col in enumerate(macro.columns, 1):
    s       = macro[col].dropna()
    valid   = len(s)
    missing = EXPECTED - valid
    pct     = valid / EXPECTED * 100
    status  = 'FULL' if valid >= EXPECTED else f'PARTIAL ({valid})'
    if valid >= EXPECTED:
        full_list.append(col)
    else:
        fail_list.append((col, valid))
    print(f"{i:>3}. {col:<35} {valid:>6} {missing:>8} {pct:>8.1f}%  {status}")

print("-" * 72)
print(f"\n  FULL    : {len(full_list)}/18")
print(f"  PARTIAL : {len(fail_list)}/18")

if fail_list:
    print("\n  [!] Partial variables:")
    for col, v in fail_list:
        print(f"      {col}  ({v}/240 months)")

print("\n" + "=" * 65)
print("  DESCRIPTIVE STATISTICS")
print("=" * 65)
print(macro.describe().round(3).to_string())

print("\n" + "=" * 65)
print("  SAVING")
print("=" * 65)

output_path = os.path.join(OUTPUT_FOLDER, 'macro_18_full_variables.csv')
macro.to_csv(output_path, float_format='%.6f')
print(f"\n  [SAVED] {output_path}")
print(f"          Rows    : {len(macro)}")
print(f"          Columns : {len(macro.columns)}")
print(f"          Period  : {macro.index[0].date()} --> {macro.index[-1].date()}")

print("\n" + "=" * 65)
print("  ALL 18 VARIABLES IN macro_18_full_variables.csv")
print("=" * 65)

descriptions = {
    'Oil_Brent_USD'           : 'Brent Crude Oil Price (USD/barrel)      -- inflation, commodity cycle',
    'Gold_USD'                : 'Gold Price (USD/troy oz)                 -- safe haven, inflation hedge',
    'US_Industrial_Production': 'US Industrial Production Index           -- global growth proxy',
    'US_CFNAI'                : 'Chicago Fed National Activity Index      -- US economic activity (-/+)',
    'US_Fed_Funds_Rate'       : 'US Federal Funds Rate (%)               -- global rate cycle',
    'US_10Y_Yield'            : 'US 10-Year Treasury Yield (%)           -- global risk-free rate',
    'US_2Y_Yield'             : 'US 2-Year Treasury Yield (%)            -- near-term rate expectations',
    'US_BAA_Yield'            : 'US BAA Corporate Bond Yield (%)         -- corporate credit cost',
    'US_AAA_Yield'            : 'US AAA Corporate Bond Yield (%)         -- investment-grade benchmark',
    'US_3M_TBill'             : 'US 3-Month T-Bill Rate (%)              -- short-term risk-free rate',
    'US_Unemployment_Rate'    : 'US Unemployment Rate (%)                -- labor market health',
    'IN_Forex_Reserves_USD'   : 'India Forex Reserves (USD billions)     -- external sector buffer',
    'US_Housing_Starts'       : 'US Housing Starts (thousands)           -- leading economic indicator',
    'USDINR'                  : 'USD/INR Exchange Rate                   -- India currency pressure',
    'DXY_Dollar_Index'        : 'US Dollar Index (DXY)                   -- global dollar strength',
    'US_VIX'                  : 'CBOE Volatility Index                   -- global risk appetite',
    'US_Yield_Curve_Spread'   : 'US 10Y - 2Y Yield Spread (%)  DERIVED  -- recession signal',
    'US_Credit_Spread'        : 'US BAA - AAA Yield Spread (%)  DERIVED  -- corporate stress signal',
}

for i, col in enumerate(macro.columns, 1):
    desc = descriptions.get(col, '')
    print(f"  {i:2d}. {col:<35}  {desc}")

print("\n" + "=" * 65)
print("  DONE")
print("=" * 65)
