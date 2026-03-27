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

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════════
asset_raw = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date'])
asset_raw  = asset_raw.set_index('Date')
asset_raw.index = pd.to_datetime(asset_raw.index).tz_localize(None)
asset_raw  = asset_raw.sort_index()

macro_raw = pd.read_csv('macro_finaldata_2005_2024.csv', parse_dates=['Date'])
macro_raw  = macro_raw.set_index('Date')
macro_raw.index = pd.to_datetime(macro_raw.index).tz_localize(None)
macro_raw  = macro_raw.sort_index()

ASSETS       = ['Nifty50_USD', 'SP500', 'Gold', 'USBond']
MACRO_COLS   = ['CPI', 'PMI', 'Repo_Rate', 'M2', 'India_VIX']

print("✓ Asset data loaded")
print(f"  Shape      : {asset_raw.shape}")
print(f"  Date range : {asset_raw.index[0].date()} → {asset_raw.index[-1].date()}")
print(f"\n✓ Macro data loaded")
print(f"  Shape      : {macro_raw.shape}")
print(f"  Date range : {macro_raw.index[0].date()} → {macro_raw.index[-1].date()}")

# ════════════════════════════════════════════════════════════════════
# SECTION 2 — LOG-LINEAR REGRESSION
# ════════════════════════════════════════════════════════════════════
COLORS = {
    'Nifty50_USD' : '#1f77b4',
    'SP500'       : '#ff7f0e',
    'Gold'        : '#FFD700',
    'USBond'      : '#2ca02c',
}

def ols_regression(series, name):
    s         = series.dropna()
    t0        = s.index[0]
    t         = ((s.index - t0).days.values / 365.25).reshape(-1, 1)
    log_price = np.log(s.values)
    model    = LinearRegression()
    model.fit(t, log_price)
    a        = model.intercept_
    b        = model.coef_[0]
    log_pred = model.predict(t)
    return {
        'series'    : s,
        't'         : t,
        'log_price' : log_price,
        'log_pred'  : log_pred,
        'a'         : a,
        'b'         : b,
        'CAGR'      : np.exp(b) - 1,
        'R2'        : r2_score(log_price, log_pred),
        'RMSE'      : np.sqrt(mean_squared_error(log_price, log_pred)),
        'std_dev'   : np.std(log_price),
        'variance'  : np.var(log_price),
    }

ols_results = {a: ols_regression(asset_raw[a], a) for a in ASSETS}

print("\n" + "="*80)
print("  OLS LOG-LINEAR REGRESSION  —  log(Price) = a + b×t  (2005–2024)")
print("="*80)
header = f"  {'Asset':<14} {'Intercept a':>13} {'Slope b':>10} {'CAGR':>8} {'R²':>8} {'RMSE':>8} {'Std Dev':>10} {'Variance':>10}"
print(header)
print("  " + "-"*78)
for name, r in ols_results.items():
    print(f"  {name:<14} {r['a']:>13.6f} {r['b']:>10.6f} "
          f"{r['CAGR']*100:>7.2f}% {r['R2']:>8.4f} "
          f"{r['RMSE']:>8.4f} {r['std_dev']:>10.6f} {r['variance']:>10.6f}")

# ════════════════════════════════════════════════════════════════════
# SECTION 3 — ML RETURN PREDICTION
# ════════════════════════════════════════════════════════════════════
TRAIN_END  = '2019-12-31'
TEST_START = '2020-01-01'

monthly_prices  = asset_raw[ASSETS].resample('ME').last().ffill()
monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

macro_monthly = macro_raw[MACRO_COLS].resample('ME').last().ffill()
macro_monthly['CPI_Change']  = macro_monthly['CPI'].pct_change()
macro_monthly['M2_Change']   = macro_monthly['M2'].pct_change()
macro_monthly['VIX_Change']  = macro_monthly['India_VIX'].pct_change()
FEATURE_COLS = ['CPI', 'PMI', 'Repo_Rate', 'M2', 'India_VIX',
                'CPI_Change', 'M2_Change', 'VIX_Change']

features_lagged = macro_monthly[FEATURE_COLS].shift(1)
ml_data = pd.concat([monthly_returns, features_lagged], axis=1).dropna()

print(f"\n✓ ML dataset : {ml_data.shape[0]} months × {ml_data.shape[1]} columns")
print(f"  Date range : {ml_data.index[0].date()} → {ml_data.index[-1].date()}")

train = ml_data.loc[:TRAIN_END]
test  = ml_data.loc[TEST_START:]

X_train = train[FEATURE_COLS].values
X_test  = test[FEATURE_COLS].values

scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

print(f"\n  Train : {len(train)} months  ({train.index[0].date()} → {train.index[-1].date()})")
print(f"  Test  : {len(test)}  months  ({test.index[0].date()} → {test.index[-1].date()})")
print(f"  Features : {len(FEATURE_COLS)}")

MODELS = {
    'SVR'              : (SVR(kernel='rbf', C=1.0, epsilon=0.005, gamma='scale'), True),
    'RandomForest'     : (RandomForestRegressor(n_estimators=200, max_depth=5,
                                                min_samples_leaf=3,
                                                random_state=42, n_jobs=-1), False),
    'GradientBoosting' : (GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                     learning_rate=0.05,
                                                     subsample=0.8,
                                                     random_state=42), False),
}

def evaluate(model, X_tr, y_tr, X_te, y_te, scaled):
    Xtr = Xs_train if scaled else X_tr
    Xte = Xs_test  if scaled else X_te
    model.fit(Xtr, y_tr)
    yp_tr = model.predict(Xtr)
    yp_te = model.predict(Xte)
    return {
        'Train_RMSE'   : np.sqrt(mean_squared_error(y_tr, yp_tr)),
        'Test_RMSE'    : np.sqrt(mean_squared_error(y_te, yp_te)),
        'Train_R2'     : r2_score(y_tr, yp_tr),
        'Test_R2'      : r2_score(y_te, yp_te),
        'Test_MAE'     : mean_absolute_error(y_te, yp_te),
        'Dir_Accuracy' : np.mean(np.sign(yp_te) == np.sign(y_te)),
        'y_pred_test'  : yp_te,
    }

all_results = {}
for asset in ASSETS:
    y_train = train[asset].values
    y_test  = test[asset].values
    all_results[asset] = {}
    for mname, (model, scaled) in MODELS.items():
        all_results[asset][mname] = evaluate(
            model, X_train, y_train, X_test, y_test, scaled)

# ════════════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS TABLES
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  ML MODEL RESULTS — TEST SET  (2020–2024)")
print("="*80)
for asset in ASSETS:
    print(f"\n{'─'*70}")
    print(f"  {asset}")
    print(f"{'─'*70}")
    print(f"  {'Model':<20} {'Train RMSE':>12} {'Test RMSE':>12} "
          f"{'Train R²':>10} {'Test R²':>10} {'MAE':>10} {'Dir.Acc':>10}")
    print(f"  {'-'*68}")
    for mname, res in all_results[asset].items():
        print(f"  {mname:<20} {res['Train_RMSE']:>12.6f} {res['Test_RMSE']:>12.6f} "
              f"{res['Train_R2']:>10.4f} {res['Test_R2']:>10.4f} "
              f"{res['Test_MAE']:>10.6f} {res['Dir_Accuracy']:>9.1%}")

print("\n" + "="*65)
print("  BEST MODEL PER ASSET  (lowest Test RMSE)")
print("="*65)
print(f"  {'Asset':<14} {'Best Model':<22} {'Test RMSE':>10} "
      f"{'Test R²':>10} {'Dir.Acc':>10}")
print("  " + "-"*63)
for asset in ASSETS:
    best_name = min(all_results[asset],
                    key=lambda m: all_results[asset][m]['Test_RMSE'])
    best = all_results[asset][best_name]
    print(f"  {asset:<14} {best_name:<22} {best['Test_RMSE']:>10.6f} "
          f"{best['Test_R2']:>10.4f} {best['Dir_Accuracy']:>9.1%}")

# ════════════════════════════════════════════════════════════════════
# SECTION 5 — PLOTS
# ════════════════════════════════════════════════════════════════════

# ── Plot 1: OLS Regression lines ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, name in zip(axes, ASSETS):
    r = ols_results[name]
    ax.scatter(r['series'].index, r['log_price'],
               s=0.5, color=COLORS[name], alpha=0.5, label='log(Price)')
    ax.plot(r['series'].index, r['log_pred'],
            color='red', linewidth=2,
            label=f"OLS: CAGR={r['CAGR']*100:.2f}%  R²={r['R2']:.4f}")
    ax.set_title(f"{name}", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("log(Price)")
    ax.legend(fontsize=8)
plt.suptitle("OLS Log-Linear Regression  (2005–2024)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_ols_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved plot1_ols_regression.png")

# ── Plot 2: Test RMSE comparison across models ────────────────────────
model_names = list(MODELS.keys())
x           = np.arange(len(ASSETS))
width       = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
for i, mname in enumerate(model_names):
    rmses = [all_results[a][mname]['Test_RMSE'] for a in ASSETS]
    ax.bar(x + i*width, rmses, width, label=mname, alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(ASSETS)
ax.set_title("Test RMSE Comparison — All Models × All Assets  (2020–2024)",
             fontweight='bold')
ax.set_ylabel("Test RMSE")
ax.legend()
plt.tight_layout()
plt.savefig('plot2_test_rmse.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot2_test_rmse.png")

# ── Plot 3: Directional Accuracy ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for i, mname in enumerate(model_names):
    daccs = [all_results[a][mname]['Dir_Accuracy']*100 for a in ASSETS]
    ax.bar(x + i*width, daccs, width, label=mname, alpha=0.85)
ax.axhline(50, color='red', linewidth=1.2, linestyle='--',
           label='Random baseline (50%)')
ax.set_xticks(x + width)
ax.set_xticklabels(ASSETS)
ax.set_title("Directional Accuracy — All Models × All Assets  (2020–2024)",
             fontweight='bold')
ax.set_ylabel("Directional Accuracy (%)")
ax.legend()
plt.tight_layout()
plt.savefig('plot3_dir_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot3_dir_accuracy.png")

# ── Plot 4: Predicted vs Actual — best model per asset ───────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, asset in zip(axes, ASSETS):
    best_name = min(all_results[asset],
                    key=lambda m: all_results[asset][m]['Test_RMSE'])
    y_actual  = test[asset].values
    y_pred    = all_results[asset][best_name]['y_pred_test']
    ax.bar(test.index, y_actual, width=20,
           color='#555555', alpha=0.35, label='Actual', zorder=2)
    ax.plot(test.index, y_pred, color='red',
            linewidth=1.5, marker='o', markersize=3,
            label=f'Predicted ({best_name})', zorder=3)
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(f"{asset}  —  Best: {best_name}", fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Log Return")
    ax.legend(fontsize=8)
plt.suptitle("Predicted vs Actual — Best Model per Asset  (Test: 2020–2024)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot4_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot4_pred_vs_actual.png")

# ── Plot 5: Feature Importance (RF & GB) ─────────────────────────────
tree_models = ['RandomForest', 'GradientBoosting']
fig, axes = plt.subplots(len(ASSETS), 2, figsize=(14, 4*len(ASSETS)))
for row, asset in enumerate(ASSETS):
    y_train = train[asset].values
    for col, mname in enumerate(tree_models):
        ax  = axes[row][col]
        mdl_template = MODELS[mname][0]
        # Clone the model to avoid interference
        from sklearn.base import clone
        mdl = clone(mdl_template)
        mdl.fit(X_train, y_train)
        imp = mdl.feature_importances_
        idx = np.argsort(imp)
        ax.barh(range(len(FEATURE_COLS)),
                imp[idx], color='steelblue',
                edgecolor='white', alpha=0.85)
        ax.set_yticks(range(len(FEATURE_COLS)))
        ax.set_yticklabels([FEATURE_COLS[i] for i in idx], fontsize=9)
        ax.set_title(f"{asset} — {mname}", fontweight='bold', fontsize=10)
        ax.set_xlabel("Importance")
plt.suptitle("Feature Importance — Tree Models  (2005–2019 Training)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot5_feature_importance.png")

# ── Plot 6: Correlation heatmap ───────────────────────────────────────
log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
corr_matrix = log_returns.corr()
cov_matrix  = log_returns.cov()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=axes[0],
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Pearson r'})
axes[0].set_title("Correlation Matrix\nMonthly Log Returns",
                  fontweight='bold')
sns.heatmap(cov_matrix, annot=True, fmt='.2e', cmap='Blues',
            ax=axes[1], linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Covariance'})
axes[1].set_title("Covariance Matrix\nMonthly Log Returns",
                  fontweight='bold')
plt.suptitle("Asset Correlation & Covariance  (2005–2024)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot6_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot6_correlation.png")

print("\n✓ ALL DONE — 6 plots saved to working directory")
