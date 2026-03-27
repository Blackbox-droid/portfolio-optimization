import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model    import LinearRegression
from sklearn.svm             import SVR
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base            import clone

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════════
asset_raw = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date'])
asset_raw = asset_raw.set_index('Date').sort_index()
asset_raw.index = pd.to_datetime(asset_raw.index).tz_localize(None)

macro_raw = pd.read_csv('macro_finaldata_india.csv', parse_dates=['Date'])
macro_raw = macro_raw.set_index('Date').sort_index()
macro_raw.index = pd.to_datetime(macro_raw.index).tz_localize(None)

TARGET = 'Nifty50_USD'
MACRO_COLS = [c for c in macro_raw.columns]  # use ALL macro features

print("=" * 70)
print("  NIFTY50_USD RETURN PREDICTION USING INDIAN MACRO FEATURES")
print("=" * 70)
print(f"\n[Data Loaded]")
print(f"  Asset daily  : {asset_raw.shape}  |  {asset_raw.index[0].date()} -> {asset_raw.index[-1].date()}")
print(f"  Macro monthly: {macro_raw.shape}  |  {macro_raw.index[0].date()} -> {macro_raw.index[-1].date()}")

# ════════════════════════════════════════════════════════════════════
# SECTION 2 — OLS LOG-LINEAR REGRESSION (log(Price) = a + b*t)
# ════════════════════════════════════════════════════════════════════
s  = asset_raw[TARGET].dropna()
t0 = s.index[0]
t  = ((s.index - t0).days.values / 365.25).reshape(-1, 1)
log_price = np.log(s.values)

ols = LinearRegression().fit(t, log_price)
a, b = ols.intercept_, ols.coef_[0]
log_pred = ols.predict(t)

ols_cagr = np.exp(b) - 1
ols_r2   = r2_score(log_price, log_pred)
ols_rmse = np.sqrt(mean_squared_error(log_price, log_pred))

print(f"\n{'='*70}")
print(f"  OLS LOG-LINEAR REGRESSION  --  log(Price) = a + b*t")
print(f"{'='*70}")
print(f"  Intercept (a)  : {a:.6f}")
print(f"  Slope (b)      : {b:.6f}")
print(f"  CAGR           : {ols_cagr*100:.2f}%")
print(f"  R-squared      : {ols_r2:.4f}")
print(f"  RMSE           : {ols_rmse:.4f}")
print(f"  Std Dev        : {np.std(log_price):.6f}")
print(f"  Variance       : {np.var(log_price):.6f}")

# ════════════════════════════════════════════════════════════════════
# SECTION 3 — PREPARE ML DATA
#   Monthly returns + lagged macro features
#   Train: 2005-2019 | Test: 2020-2024
# ════════════════════════════════════════════════════════════════════
TRAIN_END  = '2019-12-31'
TEST_START = '2020-01-01'

# Monthly price -> log returns
monthly_price   = asset_raw[[TARGET]].resample('ME').last().ffill()
monthly_returns = np.log(monthly_price / monthly_price.shift(1)).dropna()

# Macro: fill NaN then derive change features
macro_monthly = macro_raw[MACRO_COLS].resample('ME').last().ffill().bfill()

# Add derived momentum features for key series
macro_monthly['CPI_Change']    = macro_monthly['CPI'].pct_change()
macro_monthly['M2_Change']     = macro_monthly['M2'].pct_change()
macro_monthly['VIX_Change']    = macro_monthly['India_VIX'].pct_change()
macro_monthly['Oil_Change']    = macro_monthly['Oil_Brent_USD'].pct_change()
macro_monthly['Forex_Change']  = macro_monthly['IN_Forex_Reserves_USD'].pct_change()
macro_monthly['DXY_Change']    = macro_monthly['DXY_Dollar_Index'].pct_change()

FEATURE_COLS = [c for c in macro_monthly.columns]

# Lag features by 1 month (no look-ahead bias)
features_lagged = macro_monthly[FEATURE_COLS].shift(1)

# Merge
ml_data = pd.concat([monthly_returns, features_lagged], axis=1).dropna()

train = ml_data.loc[:TRAIN_END]
test  = ml_data.loc[TEST_START:]

X_train = train[FEATURE_COLS].values
X_test  = test[FEATURE_COLS].values
y_train = train[TARGET].values
y_test  = test[TARGET].values

scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

print(f"\n{'='*70}")
print(f"  ML DATASET SUMMARY")
print(f"{'='*70}")
print(f"  Total months : {len(ml_data)}  ({ml_data.index[0].date()} -> {ml_data.index[-1].date()})")
print(f"  Train        : {len(train)} months  ({train.index[0].date()} -> {train.index[-1].date()})")
print(f"  Test         : {len(test)} months  ({test.index[0].date()} -> {test.index[-1].date()})")
print(f"  Features (8) : {FEATURE_COLS}")

# ════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAIN & EVALUATE MODELS
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
        'Train_RMSE'   : np.sqrt(mean_squared_error(y_train, yp_tr)),
        'Test_RMSE'    : np.sqrt(mean_squared_error(y_test, yp_te)),
        'Train_MAE'    : mean_absolute_error(y_train, yp_tr),
        'Test_MAE'     : mean_absolute_error(y_test, yp_te),
        'Train_R2'     : r2_score(y_train, yp_tr),
        'Test_R2'      : r2_score(y_test, yp_te),
        'Dir_Accuracy' : np.mean(np.sign(yp_te) == np.sign(y_test)),
        'y_pred_train' : yp_tr,
        'y_pred_test'  : yp_te,
    }

# ════════════════════════════════════════════════════════════════════
# SECTION 5 — RESULTS TABLES
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  MODEL RESULTS  --  Nifty50_USD Monthly Return Prediction")
print(f"{'='*90}")
print(f"  {'Model':<22} {'Tr RMSE':>10} {'Te RMSE':>10} {'Tr MAE':>10} {'Te MAE':>10} "
      f"{'Tr R2':>8} {'Te R2':>8} {'Dir Acc':>8}")
print(f"  {'-'*86}")
for mname, r in results.items():
    print(f"  {mname:<22} {r['Train_RMSE']:>10.6f} {r['Test_RMSE']:>10.6f} "
          f"{r['Train_MAE']:>10.6f} {r['Test_MAE']:>10.6f} "
          f"{r['Train_R2']:>8.4f} {r['Test_R2']:>8.4f} {r['Dir_Accuracy']:>7.1%}")

best_name = min(results, key=lambda m: results[m]['Test_RMSE'])
best = results[best_name]
print(f"\n  >>> Best Model (lowest Test RMSE): {best_name}")
print(f"      Test RMSE={best['Test_RMSE']:.6f}  R2={best['Test_R2']:.4f}  Dir.Acc={best['Dir_Accuracy']:.1%}")

# ════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTS
# ════════════════════════════════════════════════════════════════════

# --- Plot 1: OLS log-linear regression ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(s.index, log_price, s=0.4, color='#1f77b4', alpha=0.5, label='log(Price)')
ax.plot(s.index, log_pred, color='red', lw=2,
        label=f'OLS: CAGR={ols_cagr*100:.2f}%  R2={ols_r2:.4f}')
ax.set_title('Nifty50_USD  --  OLS Log-Linear Regression (2005-2024)', fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('log(Price)')
ax.legend(); plt.tight_layout()
plt.savefig('nifty_plot1_ols.png', dpi=150, bbox_inches='tight'); plt.close()

# --- Plot 2: Test RMSE bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
rmses = [results[m]['Test_RMSE'] for m in names]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(names, rmses, color=colors, alpha=0.85, edgecolor='white')
ax.set_title('Test RMSE Comparison -- Nifty50_USD (2020-2024)', fontweight='bold')
ax.set_ylabel('Test RMSE')
for bar, v in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('nifty_plot2_rmse.png', dpi=150, bbox_inches='tight'); plt.close()

# --- Plot 3: Directional Accuracy bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
daccs = [results[m]['Dir_Accuracy']*100 for m in names]
bars = ax.bar(names, daccs, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(50, color='red', lw=1.2, ls='--', label='Random baseline (50%)')
ax.set_title('Directional Accuracy -- Nifty50_USD (2020-2024)', fontweight='bold')
ax.set_ylabel('Directional Accuracy (%)')
for bar, v in zip(bars, daccs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig('nifty_plot3_dir_accuracy.png', dpi=150, bbox_inches='tight'); plt.close()

# --- Plot 4: Predicted vs Actual (all 4 models, 2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (mname, r) in zip(axes.flatten(), results.items()):
    ax.bar(test.index, y_test, width=20, color='#555555', alpha=0.35,
           label='Actual', zorder=2)
    ax.plot(test.index, r['y_pred_test'], color='red', lw=1.5,
            marker='o', ms=3, label=f'Predicted', zorder=3)
    ax.axhline(0, color='black', lw=0.7, ls=':')
    ax.set_title(f'{mname}  (R2={r["Test_R2"]:.4f})', fontweight='bold')
    ax.set_xlabel('Date'); ax.set_ylabel('Monthly Log Return')
    ax.legend(fontsize=8)
plt.suptitle('Predicted vs Actual -- Nifty50_USD (Test: 2020-2024)',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('nifty_plot4_pred_vs_actual.png', dpi=150, bbox_inches='tight'); plt.close()

# --- Plot 5: Feature Importance (RF & GB) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, mname in zip(axes, ['Random Forest', 'Gradient Boosting']):
    mdl = clone(MODELS[mname][0]).fit(X_train, y_train)
    imp = mdl.feature_importances_
    idx = np.argsort(imp)
    ax.barh(range(len(FEATURE_COLS)), imp[idx], color='steelblue',
            edgecolor='white', alpha=0.85)
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in idx], fontsize=10)
    ax.set_title(f'{mname}', fontweight='bold')
    ax.set_xlabel('Feature Importance')
plt.suptitle('Feature Importance -- Nifty50_USD (Train: 2005-2019)',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('nifty_plot5_feature_importance.png', dpi=150, bbox_inches='tight'); plt.close()

# --- Plot 6: Residual distribution (all models) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (mname, r) in zip(axes.flatten(), results.items()):
    residuals = y_test - r['y_pred_test']
    ax.hist(residuals, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', lw=1.5, ls='--')
    ax.set_title(f'{mname}  (mean={np.mean(residuals):.4f})', fontweight='bold')
    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Frequency')
plt.suptitle('Residual Distribution -- Nifty50_USD (Test: 2020-2024)',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('nifty_plot6_residuals.png', dpi=150, bbox_inches='tight'); plt.close()

print("\n[Plots saved]")
print("  nifty_plot1_ols.png             - OLS log-linear regression")
print("  nifty_plot2_rmse.png            - Test RMSE comparison")
print("  nifty_plot3_dir_accuracy.png    - Directional accuracy")
print("  nifty_plot4_pred_vs_actual.png  - Predicted vs Actual (all models)")
print("  nifty_plot5_feature_importance.png - Feature importance (RF & GB)")
print("  nifty_plot6_residuals.png       - Residual distributions")
print("\nDONE.")
