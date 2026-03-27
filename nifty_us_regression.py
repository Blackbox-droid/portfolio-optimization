import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════
asset_raw = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date']).set_index('Date').sort_index()
asset_raw.index = pd.to_datetime(asset_raw.index).tz_localize(None)

macro_raw = pd.read_csv('macro_finaldata_india.csv', parse_dates=['Date']).set_index('Date').sort_index()
macro_raw.index = pd.to_datetime(macro_raw.index).tz_localize(None)

TARGET = 'Nifty50_USD'
MACRO_COLS = list(macro_raw.columns)

# ════════════════════════════════════════════════════════════════════
# PREPARE ML DATA
# ════════════════════════════════════════════════════════════════════
monthly_price   = asset_raw[[TARGET]].resample('ME').last().ffill()
monthly_returns = np.log(monthly_price / monthly_price.shift(1)).dropna()

macro_monthly = macro_raw[MACRO_COLS].resample('ME').last().ffill().bfill()
macro_monthly['CPI_Change']   = macro_monthly['CPI'].pct_change()
macro_monthly['M2_Change']    = macro_monthly['M2'].pct_change()
macro_monthly['VIX_Change']   = macro_monthly['India_VIX'].pct_change()
macro_monthly['Oil_Change']   = macro_monthly['Oil_Brent_USD'].pct_change()
macro_monthly['Forex_Change'] = macro_monthly['IN_Forex_Reserves_USD'].pct_change()
macro_monthly['DXY_Change']   = macro_monthly['DXY_Dollar_Index'].pct_change()

FEATURE_COLS = list(macro_monthly.columns)

features_lagged = macro_monthly[FEATURE_COLS].shift(1)
ml_data = pd.concat([monthly_returns, features_lagged], axis=1).dropna()

TRAIN_END  = '2019-12-31'
TEST_START = '2020-01-01'
train = ml_data.loc[:TRAIN_END]
test  = ml_data.loc[TEST_START:]

X_train = train[FEATURE_COLS].values
X_test  = test[FEATURE_COLS].values
y_train = train[TARGET].values
y_test  = test[TARGET].values

# ════════════════════════════════════════════════════════════════════
# LINEAR REGRESSION
# ════════════════════════════════════════════════════════════════════
model = LinearRegression()
model.fit(X_train, y_train)

yp_train = model.predict(X_train)
yp_test  = model.predict(X_test)

# ════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  LINEAR REGRESSION  --  Nifty50_USD Monthly Return")
print("  Features: ALL macro variables (21)")
print("  Train: 2005-2019 ({} months)  |  Test: 2020-2024 ({} months)".format(len(train), len(test)))
print("=" * 70)

tr_rmse = np.sqrt(mean_squared_error(y_train, yp_train))
tr_mae  = mean_absolute_error(y_train, yp_train)
tr_r2   = r2_score(y_train, yp_train)
tr_dir  = np.mean(np.sign(yp_train) == np.sign(y_train))

te_rmse = np.sqrt(mean_squared_error(y_test, yp_test))
te_mae  = mean_absolute_error(y_test, yp_test)
te_r2   = r2_score(y_test, yp_test)
te_dir  = np.mean(np.sign(yp_test) == np.sign(y_test))

print("\n  TRAIN SET METRICS")
print("  -----------------")
print(f"  RMSE          : {tr_rmse:.6f}")
print(f"  MAE           : {tr_mae:.6f}")
print(f"  R-squared     : {tr_r2:.4f}")
print(f"  Dir. Accuracy : {tr_dir:.1%}")

print("\n  TEST SET METRICS")
print("  ----------------")
print(f"  RMSE          : {te_rmse:.6f}")
print(f"  MAE           : {te_mae:.6f}")
print(f"  R-squared     : {te_r2:.4f}")
print(f"  Dir. Accuracy : {te_dir:.1%}")

# ── Coefficients table ────────────────────────────────────────────
print("\n  REGRESSION COEFFICIENTS")
print("  {:<28} {:>14} {:>12}".format("Feature", "Coefficient", "Abs Value"))
print("  " + "-" * 54)
coefs = sorted(zip(FEATURE_COLS, model.coef_), key=lambda x: abs(x[1]), reverse=True)
for feat, c in coefs:
    print("  {:<28} {:>14.6f} {:>12.6f}".format(feat, c, abs(c)))
print("  {:<28} {:>14.6f}".format("Intercept", model.intercept_))

# ── Regression equation ───────────────────────────────────────────
print("\n  REGRESSION EQUATION:")
print(f"  Nifty50_USD_return = {model.intercept_:.6f}")
for feat, c in coefs:
    sign = "+" if c >= 0 else "-"
    print(f"    {sign} {abs(c):.6f} * {feat}")

# ════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════

# Plot 1: Predicted vs Actual (Train + Test side by side)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].bar(train.index, y_train, width=20, color='#555555', alpha=0.35, label='Actual')
axes[0].plot(train.index, yp_train, color='blue', lw=1.2, marker='o', ms=2, label='Predicted')
axes[0].axhline(0, color='black', lw=0.7, ls=':')
axes[0].set_title(f'TRAIN (2005-2019)  R2={tr_r2:.4f}', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Monthly Log Return')
axes[0].legend()

axes[1].bar(test.index, y_test, width=20, color='#555555', alpha=0.35, label='Actual')
axes[1].plot(test.index, yp_test, color='red', lw=1.5, marker='o', ms=3, label='Predicted')
axes[1].axhline(0, color='black', lw=0.7, ls=':')
axes[1].set_title(f'TEST (2020-2024)  R2={te_r2:.4f}', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Monthly Log Return')
axes[1].legend()

plt.suptitle('Linear Regression -- Nifty50_USD  Predicted vs Actual',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_regression_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Coefficient bar chart (sorted by absolute value)
fig, ax = plt.subplots(figsize=(10, 8))
feats = [f for f, c in coefs]
vals  = [c for f, c in coefs]
colors_bar = ['#d62728' if v < 0 else '#2ca02c' for v in vals]
ax.barh(range(len(feats)), vals, color=colors_bar, edgecolor='white', alpha=0.85)
ax.set_yticks(range(len(feats)))
ax.set_yticklabels(feats, fontsize=9)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Regression Coefficients -- Nifty50_USD\n(sorted by absolute value, green=+ve, red=-ve)',
             fontweight='bold')
ax.set_xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('nifty_us_regression_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Residual histograms (Train + Test)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

res_train = y_train - yp_train
res_test  = y_test - yp_test

axes[0].hist(res_train, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', lw=1.5, ls='--')
axes[0].set_title(f'Train Residuals (mean={np.mean(res_train):.4f}, std={np.std(res_train):.4f})',
                  fontweight='bold')
axes[0].set_xlabel('Residual')
axes[0].set_ylabel('Frequency')

axes[1].hist(res_test, bins=15, color='#d62728', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='blue', lw=1.5, ls='--')
axes[1].set_title(f'Test Residuals (mean={np.mean(res_test):.4f}, std={np.std(res_test):.4f})',
                  fontweight='bold')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Frequency')

plt.suptitle('Linear Regression -- Nifty50_USD Residual Distributions',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_regression_residuals.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Scatter — Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, yact, ypred, label, color in [(axes[0], y_train, yp_train, 'Train', 'blue'),
                                       (axes[1], y_test, yp_test, 'Test', 'red')]:
    ax.scatter(yact, ypred, alpha=0.6, color=color, edgecolors='white', s=40)
    mn = min(yact.min(), ypred.min())
    mx = max(yact.max(), ypred.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1, label='Perfect prediction')
    ax.set_title(f'{label} (R2={r2_score(yact, ypred):.4f})', fontweight='bold')
    ax.set_xlabel('Actual Return')
    ax.set_ylabel('Predicted Return')
    ax.legend()

plt.suptitle('Linear Regression -- Nifty50_USD  Actual vs Predicted Scatter',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_regression_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[Plots Saved]")
print("  nifty_us_regression_pred_vs_actual.png")
print("  nifty_us_regression_coefficients.png")
print("  nifty_us_regression_residuals.png")
print("  nifty_us_regression_scatter.png")
print("\nDONE.")
