import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import clone

# ════════════════════════════════════════════════════════════════════
# LOAD TRAINING DATA (2005-2024)
# ════════════════════════════════════════════════════════════════════
asset_train = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date']).set_index('Date').sort_index()
asset_train.index = pd.to_datetime(asset_train.index).tz_localize(None)

macro_train = pd.read_csv('macro_finaldata_india.csv', parse_dates=['Date']).set_index('Date').sort_index()
macro_train.index = pd.to_datetime(macro_train.index).tz_localize(None)

# ════════════════════════════════════════════════════════════════════
# LOAD TEST DATA (Jan 2025 - Feb 2026)
# ════════════════════════════════════════════════════════════════════
asset_test = pd.read_csv('data_daily_2025_2026.csv', parse_dates=['Date']).set_index('Date').sort_index()
asset_test.index = pd.to_datetime(asset_test.index).tz_localize(None)

macro_test = pd.read_csv('macro_india_2025_2026.csv', parse_dates=['Date']).set_index('Date').sort_index()
macro_test.index = pd.to_datetime(macro_test.index).tz_localize(None)

TARGET = 'Nifty50_USD'
MACRO_COLS = list(macro_train.columns)

print("=" * 75)
print("  NIFTY50_USD — TRAIN ON 2005-2024, TEST ON JAN 2025 - FEB 2026")
print("=" * 75)

# ════════════════════════════════════════════════════════════════════
# PREPARE TRAINING SET (2005-2024)
# ════════════════════════════════════════════════════════════════════
monthly_price_tr = asset_train[[TARGET]].resample('ME').last().ffill()
monthly_ret_tr   = np.log(monthly_price_tr / monthly_price_tr.shift(1)).dropna()

macro_m_tr = macro_train[MACRO_COLS].resample('ME').last().ffill().bfill()
macro_m_tr['CPI_Change']   = macro_m_tr['CPI'].pct_change()
macro_m_tr['M2_Change']    = macro_m_tr['M2'].pct_change()
macro_m_tr['VIX_Change']   = macro_m_tr['India_VIX'].pct_change()
macro_m_tr['Oil_Change']   = macro_m_tr['Oil_Brent_USD'].pct_change()
macro_m_tr['Forex_Change'] = macro_m_tr['IN_Forex_Reserves_USD'].pct_change()
macro_m_tr['DXY_Change']   = macro_m_tr['DXY_Dollar_Index'].pct_change()

FEATURE_COLS = list(macro_m_tr.columns)

features_lag_tr = macro_m_tr[FEATURE_COLS].shift(1)
train_data = pd.concat([monthly_ret_tr, features_lag_tr], axis=1).dropna()

X_train = train_data[FEATURE_COLS].values
y_train = train_data[TARGET].values

# ════════════════════════════════════════════════════════════════════
# PREPARE TEST SET (Jan 2025 - Feb 2026)
# ════════════════════════════════════════════════════════════════════
monthly_price_te = asset_test[[TARGET]].resample('ME').last().ffill()
monthly_ret_te   = np.log(monthly_price_te / monthly_price_te.shift(1)).dropna()

# Use same columns from macro_india test file
macro_m_te = macro_test[MACRO_COLS].resample('ME').last().ffill().bfill()
macro_m_te = macro_m_te.fillna(0)  # PMI is NaN in test period — fill with 0 for compatibility
macro_m_te['CPI_Change']   = macro_m_te['CPI'].pct_change()
macro_m_te['M2_Change']    = macro_m_te['M2'].pct_change()
macro_m_te['VIX_Change']   = macro_m_te['India_VIX'].pct_change()
macro_m_te['Oil_Change']   = macro_m_te['Oil_Brent_USD'].pct_change()
macro_m_te['Forex_Change'] = macro_m_te['IN_Forex_Reserves_USD'].pct_change()
macro_m_te['DXY_Change']   = macro_m_te['DXY_Dollar_Index'].pct_change()

features_lag_te = macro_m_te[FEATURE_COLS].shift(1)
test_data = pd.concat([monthly_ret_te, features_lag_te], axis=1).dropna()

X_test = test_data[FEATURE_COLS].values
y_test = test_data[TARGET].values

# Scale features
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

print(f"\n  Train : {len(train_data)} months ({train_data.index[0].date()} -> {train_data.index[-1].date()})")
print(f"  Test  : {len(test_data)} months ({test_data.index[0].date()} -> {test_data.index[-1].date()})")
print(f"  Features: {len(FEATURE_COLS)}")

# ════════════════════════════════════════════════════════════════════
# TRAIN & EVALUATE ALL 4 MODELS
# ════════════════════════════════════════════════════════════════════
MODELS = {
    'Linear Regression': (LinearRegression(), False),
    'SVR (RBF)'        : (SVR(kernel='rbf', C=1.0, epsilon=0.005, gamma='scale'), True),
    'Random Forest'    : (RandomForestRegressor(n_estimators=200, max_depth=5,
                                                min_samples_leaf=3,
                                                random_state=42, n_jobs=-1), False),
    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                     learning_rate=0.05,
                                                     subsample=0.8,
                                                     random_state=42), False),
}

results = {}
for mname, (model, scaled) in MODELS.items():
    mdl = clone(model)
    Xtr = Xs_train if scaled else X_train
    Xte = Xs_test  if scaled else X_test
    mdl.fit(Xtr, y_train)
    yp_tr = mdl.predict(Xtr)
    yp_te = mdl.predict(Xte)
    results[mname] = {
        'model'        : mdl,
        'y_pred_train' : yp_tr,
        'y_pred_test'  : yp_te,
        'Train_RMSE'   : np.sqrt(mean_squared_error(y_train, yp_tr)),
        'Test_RMSE'    : np.sqrt(mean_squared_error(y_test, yp_te)),
        'Train_MAE'    : mean_absolute_error(y_train, yp_tr),
        'Test_MAE'     : mean_absolute_error(y_test, yp_te),
        'Train_R2'     : r2_score(y_train, yp_tr),
        'Test_R2'      : r2_score(y_test, yp_te),
        'Dir_Accuracy' : np.mean(np.sign(yp_te) == np.sign(y_test)),
    }

# ════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("  MODEL RESULTS — Train: 2005-2024 | Test: Jan 2025 - Feb 2026 ({} months)".format(len(test_data)))
print("=" * 95)
print("  {:<22} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}".format(
    "Model", "Tr RMSE", "Te RMSE", "Tr MAE", "Te MAE", "Tr R2", "Te R2", "Dir Acc"))
print("  " + "-" * 90)
for mname, r in results.items():
    print("  {:<22} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>8.4f} {:>8.4f} {:>7.1%}".format(
        mname, r['Train_RMSE'], r['Test_RMSE'], r['Train_MAE'], r['Test_MAE'],
        r['Train_R2'], r['Test_R2'], r['Dir_Accuracy']))

best_name = min(results, key=lambda m: results[m]['Test_RMSE'])
print("\n  >>> Best Model (lowest Test RMSE): {}".format(best_name))
print("      Test RMSE={:.6f}  R2={:.4f}  Dir.Acc={:.1%}".format(
    results[best_name]['Test_RMSE'], results[best_name]['Test_R2'],
    results[best_name]['Dir_Accuracy']))

# ── Actual vs Predicted table ─────────────────────────────────────
print("\n" + "=" * 95)
print("  MONTH-BY-MONTH PREDICTIONS vs ACTUAL (Test: Jan 2025 - Feb 2026)")
print("=" * 95)
header = "  {:<12} {:>10}".format("Month", "Actual")
for mname in results:
    header += " {:>12}".format(mname[:10])
print(header)
print("  " + "-" * 70)
for i, date in enumerate(test_data.index):
    row = "  {:<12} {:>10.4f}".format(date.strftime('%Y-%m'), y_test[i])
    for mname, r in results.items():
        pred = r['y_pred_test'][i]
        # Mark correct direction with *
        marker = "*" if np.sign(pred) == np.sign(y_test[i]) else " "
        row += " {:>11.4f}{}".format(pred, marker)
    print(row)
print("  (* = correct direction)")

# ── Linear Regression Coefficients ────────────────────────────────
lr = results['Linear Regression']['model']
coefs = sorted(zip(FEATURE_COLS, lr.coef_), key=lambda x: abs(x[1]), reverse=True)
print("\n" + "=" * 60)
print("  LINEAR REGRESSION COEFFICIENTS (top 10)")
print("=" * 60)
print("  {:<28} {:>14}".format("Feature", "Coefficient"))
print("  " + "-" * 42)
for feat, c in coefs[:10]:
    print("  {:<28} {:>14.6f}".format(feat, c))
print("  {:<28} {:>14.6f}".format("Intercept", lr.intercept_))

# ── Feature Importance (RF & GB) ──────────────────────────────────
print("\n" + "=" * 70)
print("  FEATURE IMPORTANCE — Top 10")
print("=" * 70)
rf_imp = results['Random Forest']['model'].feature_importances_
gb_imp = results['Gradient Boosting']['model'].feature_importances_
rf_sorted = sorted(zip(FEATURE_COLS, rf_imp), key=lambda x: x[1], reverse=True)
gb_sorted = sorted(zip(FEATURE_COLS, gb_imp), key=lambda x: x[1], reverse=True)

print("  {:<5} {:<25} {:>10}   {:<25} {:>10}".format(
    "Rank", "Random Forest", "Import.", "Gradient Boosting", "Import."))
print("  " + "-" * 78)
for i in range(10):
    print("  {:<5} {:<25} {:>10.4f}   {:<25} {:>10.4f}".format(
        i+1, rf_sorted[i][0], rf_sorted[i][1], gb_sorted[i][0], gb_sorted[i][1]))

# ════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════

# Plot 1: Predicted vs Actual — all 4 models (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (mname, r) in zip(axes.flatten(), results.items()):
    ax.bar(test_data.index, y_test, width=20, color='#555555', alpha=0.4,
           label='Actual', zorder=2)
    ax.plot(test_data.index, r['y_pred_test'], color='red', lw=2,
            marker='o', ms=5, label='Predicted', zorder=3)
    ax.axhline(0, color='black', lw=0.7, ls=':')
    ax.set_title("{} (R2={:.4f}, RMSE={:.4f}, Dir={:.0%})".format(
        mname, r['Test_R2'], r['Test_RMSE'], r['Dir_Accuracy']),
        fontweight='bold', fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Log Return')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Predicted vs Actual -- Nifty50_USD\nTrain: 2005-2024 | Test: Jan 2025 - Feb 2026',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_test2025_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: RMSE & R2 comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
names = list(results.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# RMSE
rmses = [results[m]['Test_RMSE'] for m in names]
bars = axes[0].bar(names, rmses, color=colors, alpha=0.85, edgecolor='white')
for bar, v in zip(bars, rmses):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                 "{:.4f}".format(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_title('Test RMSE', fontweight='bold')
axes[0].set_ylabel('RMSE')
axes[0].tick_params(axis='x', rotation=25)

# R2
r2s = [results[m]['Test_R2'] for m in names]
bars = axes[1].bar(names, r2s, color=colors, alpha=0.85, edgecolor='white')
for bar, v in zip(bars, r2s):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 "{:.4f}".format(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_title('Test R-squared', fontweight='bold')
axes[1].set_ylabel('R2')
axes[1].tick_params(axis='x', rotation=25)

# Dir Accuracy
daccs = [results[m]['Dir_Accuracy']*100 for m in names]
bars = axes[2].bar(names, daccs, color=colors, alpha=0.85, edgecolor='white')
axes[2].axhline(50, color='red', lw=1.2, ls='--', label='Random (50%)')
for bar, v in zip(bars, daccs):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 "{:.1f}%".format(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[2].set_title('Directional Accuracy', fontweight='bold')
axes[2].set_ylabel('%')
axes[2].legend()
axes[2].tick_params(axis='x', rotation=25)

plt.suptitle('Model Comparison -- Nifty50_USD Test: Jan 2025 - Feb 2026',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_test2025_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Scatter Actual vs Predicted (2x2)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (mname, r) in zip(axes.flatten(), results.items()):
    ax.scatter(y_test, r['y_pred_test'], alpha=0.7, color='steelblue',
               edgecolors='white', s=60, zorder=3)
    mn = min(y_test.min(), r['y_pred_test'].min())
    mx = max(y_test.max(), r['y_pred_test'].max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect')
    ax.axhline(0, color='grey', lw=0.5, ls=':')
    ax.axvline(0, color='grey', lw=0.5, ls=':')
    ax.set_title("{} (R2={:.4f})".format(mname, r['Test_R2']), fontweight='bold')
    ax.set_xlabel('Actual Return')
    ax.set_ylabel('Predicted Return')
    ax.legend(fontsize=8)
plt.suptitle('Actual vs Predicted Scatter -- Nifty50_USD Test: Jan 2025 - Feb 2026',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_test2025_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Residual distributions (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (mname, r) in zip(axes.flatten(), results.items()):
    residuals = y_test - r['y_pred_test']
    ax.hist(residuals, bins=10, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', lw=1.5, ls='--')
    ax.set_title("{} (mean={:.4f}, std={:.4f})".format(
        mname, np.mean(residuals), np.std(residuals)), fontweight='bold')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
plt.suptitle('Residual Distributions -- Nifty50_USD Test: Jan 2025 - Feb 2026',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_test2025_residuals.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Feature Importance (RF & GB)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
n_feat = len(FEATURE_COLS)

rf_idx = np.argsort(rf_imp)
axes[0].barh(range(n_feat), rf_imp[rf_idx], color='#2ca02c', edgecolor='white', alpha=0.85)
axes[0].set_yticks(range(n_feat))
axes[0].set_yticklabels([FEATURE_COLS[i] for i in rf_idx], fontsize=9)
axes[0].set_title('Random Forest', fontweight='bold')
axes[0].set_xlabel('Feature Importance')

gb_idx = np.argsort(gb_imp)
axes[1].barh(range(n_feat), gb_imp[gb_idx], color='#d62728', edgecolor='white', alpha=0.85)
axes[1].set_yticks(range(n_feat))
axes[1].set_yticklabels([FEATURE_COLS[i] for i in gb_idx], fontsize=9)
axes[1].set_title('Gradient Boosting', fontweight='bold')
axes[1].set_xlabel('Feature Importance')

plt.suptitle('Feature Importance -- Nifty50_USD (Trained on 2005-2024)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_test2025_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: Cumulative return — Actual vs Best model
fig, ax = plt.subplots(figsize=(12, 5))
cum_actual = np.cumsum(y_test)
for mname, r in results.items():
    cum_pred = np.cumsum(r['y_pred_test'])
    ax.plot(test_data.index, cum_pred, lw=1.5, marker='o', ms=4, label=mname)
ax.plot(test_data.index, cum_actual, lw=2.5, color='black', marker='s', ms=5,
        label='Actual', zorder=5)
ax.axhline(0, color='grey', lw=0.7, ls=':')
ax.set_title('Cumulative Log Return -- Nifty50_USD (Jan 2025 - Feb 2026)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Log Return')
ax.legend()
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('nifty_us_test2025_cumulative.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[6 Plots Saved]")
print("  nifty_us_test2025_pred_vs_actual.png")
print("  nifty_us_test2025_comparison.png")
print("  nifty_us_test2025_scatter.png")
print("  nifty_us_test2025_residuals.png")
print("  nifty_us_test2025_feature_importance.png")
print("  nifty_us_test2025_cumulative.png")
print("\nDONE.")
