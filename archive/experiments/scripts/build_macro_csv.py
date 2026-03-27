# ============================================================
# BUILD macro_data.csv -- India Macro Data (Jan 2005 - Dec 2024)
# ============================================================
# Sources:
#   CPI        : FRED (INDCPIALLMINMEI) -- CPI index, converted to YoY %
#   Repo Rate  : FRED (IRSTCB01INM156N) -- RBI policy rate
#   M2 (M3)    : FRED (MABMM301INM189N) -- Broad money in national currency
#   India VIX  : Yahoo Finance (^INDIAVIX) -- available from Mar 2009
#   PMI        : FRED proxy (OECD Leading Indicator)
#
# FRED data is downloaded via direct CSV URL (no API key needed).
# ============================================================

import sys
import io
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

START = '2004-01-01'   # extra year for CPI YoY calculation
END   = '2024-12-31'

# ----------------------------------------------------------------
# HELPER -- download a FRED series via CSV URL (no API key)
# ----------------------------------------------------------------
def fetch_fred_csv(series_id, start=START, end=END, col_name=None):
    """Download a FRED series as CSV. No API key required."""
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )
    name = col_name or series_id
    try:
        df = pd.read_csv(url, na_values='.')
        # FRED CSV columns vary: 'DATE', 'observation_date', etc.
        date_col = [c for c in df.columns if 'date' in c.lower()]
        if not date_col:
            print(f"  [FAIL] {name}: No date column found. Columns: {df.columns.tolist()}")
            return pd.Series(dtype=float, name=name)

        df[date_col[0]] = pd.to_datetime(df[date_col[0]])
        df = df.set_index(date_col[0])
        s = df.iloc[:, 0].astype(float)
        s.name = name
        valid = s.dropna()
        print(f"  [OK] {name}: {len(valid)} obs "
              f"({valid.index[0].date()} -> {valid.index[-1].date()})")
        return s
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return pd.Series(dtype=float, name=name)


# ============================================================
# 1. FETCH ALL SERIES
# ============================================================
print("=" * 60)
print("  DOWNLOADING MACRO DATA")
print("=" * 60)

# -- 1a. CPI Index -> compute YoY % --
print("\n[1/5] CPI (INDCPIALLMINMEI)...")
cpi_index = fetch_fred_csv('INDCPIALLMINMEI', start='2004-01-01',
                           col_name='CPI_Index')
cpi_yoy = cpi_index.pct_change(12) * 100  # YoY %
cpi_yoy.name = 'CPI'

# -- 1b. RBI Repo Rate --
print("\n[2/5] Repo Rate (IRSTCB01INM156N)...")
repo = fetch_fred_csv('IRSTCB01INM156N', col_name='Repo_Rate')

# -- 1c. M3 Broad Money --
print("\n[3/5] M3 Money Supply (MABMM301INM189N)...")
m3 = fetch_fred_csv('MABMM301INM189N', col_name='M2')

# -- 1d. India VIX (Yahoo Finance) --
print("\n[4/5] India VIX (^INDIAVIX via yfinance)...")
try:
    vix_raw = yf.download('^INDIAVIX', start='2005-01-01', end=END,
                          auto_adjust=True, progress=False)
    vix_monthly = vix_raw['Close'].resample('ME').last().squeeze()
    vix_monthly.name = 'India_VIX'
    valid_vix = vix_monthly.dropna()
    print(f"  [OK] India_VIX: {len(valid_vix)} obs "
          f"({valid_vix.index[0].date()} -> {valid_vix.index[-1].date()})")
except Exception as e:
    print(f"  [FAIL] India VIX: {e}")
    vix_monthly = pd.Series(dtype=float, name='India_VIX')

# -- 1e. PMI -- FRED proxy (OECD Leading Indicator) --
print("\n[5/5] PMI proxy (INDLORSGPNOSTSAM -- OECD Leading Indicator)...")
pmi_proxy = fetch_fred_csv('INDLORSGPNOSTSAM', col_name='PMI')

# Fallback: Business Confidence Index
if pmi_proxy.dropna().empty:
    print("  -> Trying BSCICP03INM665S (Business Confidence)...")
    pmi_proxy = fetch_fred_csv('BSCICP03INM665S', col_name='PMI')

# Second fallback: Consumer Confidence
if pmi_proxy.dropna().empty:
    print("  -> Trying CSCICP03INM665S (Consumer Confidence)...")
    pmi_proxy = fetch_fred_csv('CSCICP03INM665S', col_name='PMI')

# ============================================================
# 2. ASSEMBLE INTO SINGLE DATAFRAME
# ============================================================
print("\n" + "=" * 60)
print("  ASSEMBLING MACRO DATA")
print("=" * 60)

# Resample everything to month-end
def to_month_end(s):
    if s.empty:
        return s
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample('ME').last()

cpi_m   = to_month_end(cpi_yoy)
repo_m  = to_month_end(repo)
m3_m    = to_month_end(m3)
vix_m   = to_month_end(vix_monthly) if len(vix_monthly) > 0 else vix_monthly
pmi_m   = to_month_end(pmi_proxy)

# Build DataFrame
macro = pd.DataFrame({
    'CPI':       cpi_m,
    'PMI':       pmi_m,
    'Repo_Rate': repo_m,
    'M2':        m3_m,
    'India_VIX': vix_m,
})

# Filter to target period
macro = macro.loc[pd.Timestamp('2005-01-01'):pd.Timestamp('2024-12-31')]

# ============================================================
# 3. HANDLE MISSING DATA
# ============================================================
print(f"\nDate range: {macro.index[0].date()} -> {macro.index[-1].date()}")
print(f"Shape: {macro.shape}")
print(f"\nMissing values BEFORE fill:")
print(macro.isnull().sum())
print(f"\nMissing % :")
print((macro.isnull().sum() / len(macro) * 100).round(1))

# Forward-fill gaps of <= 3 months (policy rates often unchanged)
macro = macro.ffill(limit=3)

# For India VIX (starts Mar 2009), backfill is inappropriate -- leave NaN
# For M3 (may end mid-2023), forward-fill last known value with a warning
if macro['M2'].isnull().sum() > 0:
    last_m2_date = macro['M2'].last_valid_index()
    if last_m2_date is not None:
        print(f"\n[!] M2 data ends at {last_m2_date.date()}. "
              f"Forward-filling remaining months.")
        macro['M2'] = macro['M2'].ffill()

print(f"\nMissing values AFTER fill:")
print(macro.isnull().sum())

# ============================================================
# 4. DATA QUALITY CHECKS
# ============================================================
print("\n" + "=" * 60)
print("  DATA QUALITY CHECKS")
print("=" * 60)

print("\nDescriptive Statistics:")
print(macro.describe().round(2).to_string())

# Sanity checks
checks = {
    'CPI':       (0, 20),
    'PMI':       (90, 105),
    'Repo_Rate': (3, 10),
    'India_VIX': (5, 90),
}
print("\nRange checks:")
for col, (lo, hi) in checks.items():
    if col in macro.columns and macro[col].dropna().any():
        mn, mx = macro[col].min(), macro[col].max()
        ok = '[OK]' if lo <= mn and mx <= hi * 1.5 else '[!]'
        print(f"  {ok} {col}: [{mn:.2f}, {mx:.2f}] (expected ~[{lo}, {hi}])")

# ============================================================
# 5. SAVE CSV
# ============================================================
macro.index.name = 'Date'
output_path = 'macro_data.csv'
macro.to_csv(output_path, float_format='%.6f')

print(f"\n{'=' * 60}")
print(f"  SAVED: {output_path}")
print(f"    Rows:    {len(macro)}")
print(f"    Columns: {macro.columns.tolist()}")
print(f"    Period:  {macro.index[0].date()} -> {macro.index[-1].date()}")
print(f"{'=' * 60}")

# ============================================================
# 6. NOTES
# ============================================================
print("""
NOTES:
  1. PMI: No free API for India PMI. Script uses OECD Leading
     Indicator as proxy. For actual PMI, download manually from
     tradingeconomics.com/india/manufacturing-pmi and replace
     the 'PMI' column in macro_data.csv.

  2. India VIX: Data starts ~Mar 2009. Earlier months are NaN.

  3. M2: Uses India M3 (broad money) from OECD/FRED.
     Series may end before Dec 2024 -- gap is forward-filled.

  4. CPI: Computed as YoY % from CPI index level.
""")
