import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import clone

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
# PREPARE DATA — USE ALL DATA FOR TRAINING (2005-2024)
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
n_features = len(FEATURE_COLS)

# Lag features by 1 month (no look-ahead bias)
features_lagged = macro_monthly[FEATURE_COLS].shift(1)
ml_data = pd.concat([monthly_returns, features_lagged], axis=1).dropna()

X = ml_data[FEATURE_COLS].values
y = ml_data[TARGET].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

print("=" * 75)
print("  NIFTY50_USD RETURN PREDICTION — ALL DATA TRAINING (2005-2024)")
print("=" * 75)
print(f"  Months     : {len(ml_data)}  ({ml_data.index[0].date()} -> {ml_data.index[-1].date()})")
print(f"  Features   : {n_features}")
print(f"  Target     : Monthly log return of {TARGET}")
print(f"  Features   : {FEATURE_COLS}")

# ════════════════════════════════════════════════════════════════════
# DEFINE & TRAIN ALL MODELS
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
    Xtr = Xs if scaled else X
    mdl.fit(Xtr, y)
    yp = mdl.predict(Xtr)
    results[mname] = {
        'model'        : mdl,
        'y_pred'       : yp,
        'RMSE'         : np.sqrt(mean_squared_error(y, yp)),
        'MAE'          : mean_absolute_error(y, yp),
        'R2'           : r2_score(y, yp),
        'Dir_Accuracy' : np.mean(np.sign(yp) == np.sign(y)),
        'Residual_Mean': np.mean(y - yp),
        'Residual_Std' : np.std(y - yp),
    }

# ════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 85)
print("  MODEL RESULTS — Full Dataset Training (2005-2024, {} months)".format(len(ml_data)))
print("=" * 85)
print("  {:<22} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
    "Model", "RMSE", "MAE", "R2", "Dir Acc", "Resid Mean", "Resid Std"))
print("  " + "-" * 82)
for mname, r in results.items():
    print("  {:<22} {:>10.6f} {:>10.6f} {:>10.4f} {:>9.1%} {:>12.6f} {:>10.6f}".format(
        mname, r['RMSE'], r['MAE'], r['R2'], r['Dir_Accuracy'],
        r['Residual_Mean'], r['Residual_Std']))

best_name = min(results, key=lambda m: results[m]['RMSE'])
print("\n  >>> Best Model (lowest RMSE): {}".format(best_name))
print("      RMSE={:.6f}  R2={:.4f}  Dir.Acc={:.1%}".format(
    results[best_name]['RMSE'], results[best_name]['R2'], results[best_name]['Dir_Accuracy']))

# ════════════════════════════════════════════════════════════════════
# LINEAR REGRESSION — COEFFICIENTS & EQUATION
# ════════════════════════════════════════════════════════════════════
lr_model = results['Linear Regression']['model']
coefs = sorted(zip(FEATURE_COLS, lr_model.coef_), key=lambda x: abs(x[1]), reverse=True)

print("\n" + "=" * 60)
print("  LINEAR REGRESSION — COEFFICIENTS")
print("=" * 60)
print("  {:<28} {:>14} {:>12}".format("Feature", "Coefficient", "Abs Value"))
print("  " + "-" * 54)
for feat, c in coefs:
    print("  {:<28} {:>14.6f} {:>12.6f}".format(feat, c, abs(c)))
print("  {:<28} {:>14.6f}".format("Intercept", lr_model.intercept_))

print("\n  REGRESSION EQUATION:")
print("  Nifty50_USD_return = {:.6f}".format(lr_model.intercept_))
for feat, c in coefs:
    sign = "+" if c >= 0 else "-"
    print("    {} {:.6f} * {}".format(sign, abs(c), feat))

# ════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — RF & GB
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FEATURE IMPORTANCE — Random Forest & Gradient Boosting")
print("=" * 70)

rf_imp = results['Random Forest']['model'].feature_importances_
gb_imp = results['Gradient Boosting']['model'].feature_importances_

rf_sorted = sorted(zip(FEATURE_COLS, rf_imp), key=lambda x: x[1], reverse=True)
gb_sorted = sorted(zip(FEATURE_COLS, gb_imp), key=lambda x: x[1], reverse=True)

print("\n  {:<5} {:<28} {:>12}   {:<28} {:>12}".format(
    "Rank", "RF Feature", "Importance", "GB Feature", "Importance"))
print("  " + "-" * 88)
for i in range(n_features):
    print("  {:<5} {:<28} {:>12.4f}   {:<28} {:>12.4f}".format(
        i+1, rf_sorted[i][0], rf_sorted[i][1], gb_sorted[i][0], gb_sorted[i][1]))

# ════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════

# Plot 1: Predicted vs Actual — all 4 models (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (mname, r) in zip(axes.flatten(), results.items()):
    ax.bar(ml_data.index, y, width=20, color='#555555', alpha=0.3, label='Actual', zorder=2)
    ax.plot(ml_data.index, r['y_pred'], color='red', lw=1.2, marker='o', ms=1.5,
            label='Predicted', zorder=3)
    ax.axhline(0, color='black', lw=0.7, ls=':')
    ax.set_title("{} (R2={:.4f}, RMSE={:.4f})".format(mname, r['R2'], r['RMSE']),
                 fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Log Return')
    ax.legend(fontsize=8)
plt.suptitle('Predicted vs Actual -- Nifty50_USD  All Models (Full Data 2005-2024)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_all_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: RMSE & R2 comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names = list(results.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

rmses = [results[m]['RMSE'] for m in names]
bars = axes[0].bar(names, rmses, color=colors, alpha=0.85, edgecolor='white')
for bar, v in zip(bars, rmses):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 "{:.4f}".format(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_title('RMSE Comparison', fontweight='bold')
axes[0].set_ylabel('RMSE')

r2s = [results[m]['R2'] for m in names]
bars = axes[1].bar(names, r2s, color=colors, alpha=0.85, edgecolor='white')
for bar, v in zip(bars, r2s):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 "{:.4f}".format(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_title('R-squared Comparison', fontweight='bold')
axes[1].set_ylabel('R2')

plt.suptitle('Model Performance -- Nifty50_USD (Full Data 2005-2024)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_all_rmse_r2.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Directional Accuracy
fig, ax = plt.subplots(figsize=(10, 5))
daccs = [results[m]['Dir_Accuracy']*100 for m in names]
bars = ax.bar(names, daccs, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(50, color='red', lw=1.2, ls='--', label='Random baseline (50%)')
for bar, v in zip(bars, daccs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            "{:.1f}%".format(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Directional Accuracy -- Nifty50_USD (Full Data 2005-2024)', fontweight='bold')
ax.set_ylabel('Directional Accuracy (%)')
ax.legend()
plt.tight_layout()
plt.savefig('nifty_us_all_dir_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Actual vs Predicted scatter (2x2)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (mname, r) in zip(axes.flatten(), results.items()):
    ax.scatter(y, r['y_pred'], alpha=0.5, color='steelblue', edgecolors='white', s=35)
    mn = min(y.min(), r['y_pred'].min())
    mx = max(y.max(), r['y_pred'].max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_title("{} (R2={:.4f})".format(mname, r['R2']), fontweight='bold')
    ax.set_xlabel('Actual Return')
    ax.set_ylabel('Predicted Return')
    ax.legend(fontsize=8)
plt.suptitle('Actual vs Predicted Scatter -- Nifty50_USD (Full Data 2005-2024)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_all_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Residual distributions (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (mname, r) in zip(axes.flatten(), results.items()):
    residuals = y - r['y_pred']
    ax.hist(residuals, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', lw=1.5, ls='--')
    ax.set_title("{} (mean={:.4f}, std={:.4f})".format(mname, np.mean(residuals), np.std(residuals)),
                 fontweight='bold')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
plt.suptitle('Residual Distributions -- Nifty50_USD (Full Data 2005-2024)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_all_residuals.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: Feature Importance — RF & GB side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest
rf_idx = np.argsort(rf_imp)
axes[0].barh(range(n_features), rf_imp[rf_idx], color='#2ca02c', edgecolor='white', alpha=0.85)
axes[0].set_yticks(range(n_features))
axes[0].set_yticklabels([FEATURE_COLS[i] for i in rf_idx], fontsize=9)
axes[0].set_title('Random Forest', fontweight='bold')
axes[0].set_xlabel('Feature Importance')

# Gradient Boosting
gb_idx = np.argsort(gb_imp)
axes[1].barh(range(n_features), gb_imp[gb_idx], color='#d62728', edgecolor='white', alpha=0.85)
axes[1].set_yticks(range(n_features))
axes[1].set_yticklabels([FEATURE_COLS[i] for i in gb_idx], fontsize=9)
axes[1].set_title('Gradient Boosting', fontweight='bold')
axes[1].set_xlabel('Feature Importance')

plt.suptitle('Feature Importance -- Nifty50_USD (Full Data 2005-2024)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_us_all_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 7: Regression Coefficients
fig, ax = plt.subplots(figsize=(10, 8))
feats = [f for f, c in coefs]
vals  = [c for f, c in coefs]
colors_bar = ['#d62728' if v < 0 else '#2ca02c' for v in vals]
ax.barh(range(len(feats)), vals, color=colors_bar, edgecolor='white', alpha=0.85)
ax.set_yticks(range(len(feats)))
ax.set_yticklabels(feats, fontsize=9)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Linear Regression Coefficients -- Nifty50_USD\n(sorted by |coef|, green=+ve, red=-ve)',
             fontweight='bold')
ax.set_xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('nifty_us_all_lr_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[7 Plots Saved]")
print("  nifty_us_all_pred_vs_actual.png    - Predicted vs Actual (all models)")
print("  nifty_us_all_rmse_r2.png           - RMSE & R2 comparison")
print("  nifty_us_all_dir_accuracy.png      - Directional accuracy")
print("  nifty_us_all_scatter.png           - Actual vs Predicted scatter")
print("  nifty_us_all_residuals.png         - Residual distributions")
print("  nifty_us_all_feature_importance.png - Feature importance (RF & GB)")
print("  nifty_us_all_lr_coefficients.png   - Linear Regression coefficients")
print("\nDONE.")
