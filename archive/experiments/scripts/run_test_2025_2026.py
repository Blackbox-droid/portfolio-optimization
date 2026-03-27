import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════
# Training data (2005-2024)
train_asset = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date']).set_index('Date')
train_asset.index = pd.to_datetime(train_asset.index).tz_localize(None)
train_asset = train_asset.sort_index()

train_macro = pd.read_csv('macro_finaldata_2005_2024.csv', parse_dates=['Date']).set_index('Date')
train_macro.index = pd.to_datetime(train_macro.index).tz_localize(None)
train_macro = train_macro.sort_index()

# Test data (2025-2026)
test_asset = pd.read_csv('data_daily_2025_2026.csv', parse_dates=['Date']).set_index('Date')
test_asset.index = pd.to_datetime(test_asset.index).tz_localize(None)
test_asset = test_asset.sort_index()

test_macro = pd.read_csv('macro_finaldata_2025_2026.csv', parse_dates=['Date']).set_index('Date')
test_macro.index = pd.to_datetime(test_macro.index).tz_localize(None)
test_macro = test_macro.sort_index()

ASSETS = ['Nifty50_USD', 'SP500', 'Gold', 'USBond']
COLORS = {'Nifty50_USD': '#1f77b4', 'SP500': '#ff7f0e', 'Gold': '#FFD700', 'USBond': '#2ca02c'}

# Use common features between both macro files
train_features = set(train_macro.columns)
test_features = set(test_macro.columns)
FEATURES = sorted(list(train_features & test_features))

print("=" * 70)
print("  DATA SUMMARY")
print("=" * 70)
print(f"  Train assets : {train_asset.shape}  |  {train_asset.index[0].date()} -> {train_asset.index[-1].date()}")
print(f"  Train macro  : {train_macro.shape}  |  {train_macro.index[0].date()} -> {train_macro.index[-1].date()}")
print(f"  Test assets  : {test_asset.shape}   |  {test_asset.index[0].date()} -> {test_asset.index[-1].date()}")
print(f"  Test macro   : {test_macro.shape}   |  {test_macro.index[0].date()} -> {test_macro.index[-1].date()}")
print(f"  Common features : {len(FEATURES)}")

# ════════════════════════════════════════════════════════════════════
# PREPARE DATASETS
# ════════════════════════════════════════════════════════════════════
# Training: monthly returns + lagged macro (2005-2024)
train_monthly_prices = train_asset[ASSETS].resample('ME').last().ffill()
train_monthly_returns = np.log(train_monthly_prices / train_monthly_prices.shift(1)).dropna()
train_macro_lagged = train_macro[FEATURES].shift(1)
train_data = pd.concat([train_monthly_returns, train_macro_lagged], axis=1).dropna()

# Test: monthly returns + lagged macro (2025-2026)
test_monthly_prices = test_asset[ASSETS].resample('ME').last().ffill()

# For first test month, we need last month of training macro as lagged feature
# Combine macro data and then lag
all_macro = pd.concat([train_macro[FEATURES], test_macro[FEATURES]]).sort_index()
all_macro_lagged = all_macro.shift(1)

# Monthly returns for test period
# We need Dec 2024 price to compute Jan 2025 return
all_monthly_prices = pd.concat([train_monthly_prices, test_monthly_prices]).sort_index()
all_monthly_prices = all_monthly_prices[~all_monthly_prices.index.duplicated(keep='last')]
all_monthly_returns = np.log(all_monthly_prices / all_monthly_prices.shift(1)).dropna()

# Test data: only 2025+ months
test_data = pd.concat([all_monthly_returns, all_macro_lagged], axis=1).dropna()
test_data = test_data.loc['2025-01-01':]

print(f"\n  Train dataset : {len(train_data)} months ({train_data.index[0].date()} -> {train_data.index[-1].date()})")
print(f"  Test dataset  : {len(test_data)} months ({test_data.index[0].date()} -> {test_data.index[-1].date()})")

# ════════════════════════════════════════════════════════════════════
# TRAIN & TEST MODELS
# ════════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_train = train_data[FEATURES].values
X_test = test_data[FEATURES].values
Xs_train = scaler.fit_transform(X_train)
Xs_test = scaler.transform(X_test)

lr_results = {}
rf_results = {}

print("\n" + "=" * 80)
print("  RESULTS  --  Train: 2005-2024  |  Test: 2025-2026")
print("=" * 80)

for asset in ASSETS:
    y_train = train_data[asset].values
    y_test = test_data[asset].values

    # Linear Regression
    lr = LinearRegression().fit(Xs_train, y_train)
    lr_pred_train = lr.predict(Xs_train)
    lr_pred_test = lr.predict(Xs_test)

    lr_results[asset] = {
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, lr_pred_train)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, lr_pred_test)),
        'Train_R2': r2_score(y_train, lr_pred_train),
        'Test_R2': r2_score(y_test, lr_pred_test),
        'Test_MAE': mean_absolute_error(y_test, lr_pred_test),
        'Dir_Acc': np.mean(np.sign(lr_pred_test) == np.sign(y_test)),
        'pred': lr_pred_test,
        'actual': y_test,
        'coef': pd.Series(lr.coef_, index=FEATURES),
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                               min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred_train = rf.predict(X_train)
    rf_pred_test = rf.predict(X_test)

    rf_results[asset] = {
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, rf_pred_train)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, rf_pred_test)),
        'Train_R2': r2_score(y_train, rf_pred_train),
        'Test_R2': r2_score(y_test, rf_pred_test),
        'Test_MAE': mean_absolute_error(y_test, rf_pred_test),
        'Dir_Acc': np.mean(np.sign(rf_pred_test) == np.sign(y_test)),
        'pred': rf_pred_test,
        'actual': y_test,
        'importance': pd.Series(rf.feature_importances_, index=FEATURES),
    }

# ════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ════════════════════════════════════════════════════════════════════
print(f"\n  {'':=<78}")
print(f"  LINEAR REGRESSION")
print(f"  {'':=<78}")
print(f"  {'Asset':<14} {'Train RMSE':>12} {'Test RMSE':>12} {'Train R2':>10} {'Test R2':>10} {'MAE':>10} {'Dir.Acc':>10}")
print(f"  {'-'*76}")
for asset in ASSETS:
    r = lr_results[asset]
    print(f"  {asset:<14} {r['Train_RMSE']:>12.6f} {r['Test_RMSE']:>12.6f} "
          f"{r['Train_R2']:>10.4f} {r['Test_R2']:>10.4f} "
          f"{r['Test_MAE']:>10.6f} {r['Dir_Acc']:>9.1%}")

print(f"\n  {'':=<78}")
print(f"  RANDOM FOREST  (n_estimators=200, max_depth=5, min_samples_leaf=3)")
print(f"  {'':=<78}")
print(f"  {'Asset':<14} {'Train RMSE':>12} {'Test RMSE':>12} {'Train R2':>10} {'Test R2':>10} {'MAE':>10} {'Dir.Acc':>10}")
print(f"  {'-'*76}")
for asset in ASSETS:
    r = rf_results[asset]
    print(f"  {asset:<14} {r['Train_RMSE']:>12.6f} {r['Test_RMSE']:>12.6f} "
          f"{r['Train_R2']:>10.4f} {r['Test_R2']:>10.4f} "
          f"{r['Test_MAE']:>10.6f} {r['Dir_Acc']:>9.1%}")

# Feature importance
print(f"\n  TOP 5 RF FEATURES PER ASSET:")
for asset in ASSETS:
    imp = rf_results[asset]['importance']
    top5 = imp.nlargest(5)
    print(f"\n  {asset}:")
    for f, v in top5.items():
        bar = '#' * int(v / imp.max() * 20)
        print(f"    {f:<35} {v:.4f}  {bar}")

# ════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  MODEL COMPARISON  --  Test: 2025-2026")
print("=" * 70)
print(f"  {'Asset':<14} {'Metric':<14} {'Lin.Reg':>10} {'Rand.Forest':>13} {'Winner':>8}")
print(f"  {'-'*62}")
for asset in ASSETS:
    lr = lr_results[asset]; rf = rf_results[asset]
    rw = 'RF' if rf['Test_RMSE'] < lr['Test_RMSE'] else 'LR'
    r2w = 'RF' if rf['Test_R2'] > lr['Test_R2'] else 'LR'
    dw = 'RF' if rf['Dir_Acc'] > lr['Dir_Acc'] else 'LR'
    print(f"  {asset:<14} {'Test RMSE':<14} {lr['Test_RMSE']:>10.6f} {rf['Test_RMSE']:>13.6f} {rw:>8}")
    print(f"  {'':<14} {'Test R2':<14} {lr['Test_R2']:>10.4f} {rf['Test_R2']:>13.4f} {r2w:>8}")
    print(f"  {'':<14} {'Dir. Acc':<14} {lr['Dir_Acc']:>9.1%} {rf['Dir_Acc']:>12.1%} {dw:>8}")
    print()

# ════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════
test_index = test_data.index

# Plot 1: LR Predicted vs Actual
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, asset in zip(axes.flatten(), ASSETS):
    r = lr_results[asset]
    ax.bar(test_index, r['actual'], width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(test_index, r['pred'], color='red', linewidth=1.5, marker='o', markersize=4, label='Predicted (LR)')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(f"{asset}\nDirAcc={r['Dir_Acc']:.1%}  R2={r['Test_R2']:.4f}  RMSE={r['Test_RMSE']:.4f}", fontweight='bold')
    ax.set_ylabel("Monthly Log Return"); ax.legend(fontsize=8)
plt.suptitle("Linear Regression -- Predicted vs Actual (Test: 2025-2026)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_test2526_lr.png', dpi=150); plt.close()

# Plot 2: RF Predicted vs Actual
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, asset in zip(axes.flatten(), ASSETS):
    r = rf_results[asset]
    ax.bar(test_index, r['actual'], width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(test_index, r['pred'], color='blue', linewidth=1.5, marker='o', markersize=4, label='Predicted (RF)')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(f"{asset}\nDirAcc={r['Dir_Acc']:.1%}  R2={r['Test_R2']:.4f}  RMSE={r['Test_RMSE']:.4f}", fontweight='bold')
    ax.set_ylabel("Monthly Log Return"); ax.legend(fontsize=8)
plt.suptitle("Random Forest -- Predicted vs Actual (Test: 2025-2026)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_test2526_rf.png', dpi=150); plt.close()

# Plot 3: Feature Importance RF
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, asset in zip(axes.flatten(), ASSETS):
    imp = rf_results[asset]['importance'].sort_values()
    ax.barh(imp.index, imp.values, color=COLORS[asset], edgecolor='white', alpha=0.85)
    ax.set_title(f"{asset}", fontweight='bold'); ax.set_xlabel("Importance")
plt.suptitle("Feature Importance -- Random Forest (Trained: 2005-2024)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_test2526_rf_importance.png', dpi=150); plt.close()

# Plot 4: RMSE & Dir Acc comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(ASSETS)); width = 0.35

axes[0].bar(x - width/2, [lr_results[a]['Test_RMSE'] for a in ASSETS], width, label='Linear Regression', color='#1f77b4', alpha=0.85)
axes[0].bar(x + width/2, [rf_results[a]['Test_RMSE'] for a in ASSETS], width, label='Random Forest', color='#d62728', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(ASSETS, rotation=15)
axes[0].set_title("Test RMSE (2025-2026)", fontweight='bold'); axes[0].set_ylabel("RMSE"); axes[0].legend()

axes[1].bar(x - width/2, [lr_results[a]['Dir_Acc']*100 for a in ASSETS], width, label='Linear Regression', color='#1f77b4', alpha=0.85)
axes[1].bar(x + width/2, [rf_results[a]['Dir_Acc']*100 for a in ASSETS], width, label='Random Forest', color='#d62728', alpha=0.85)
axes[1].axhline(50, color='black', linestyle='--', linewidth=1, label='Baseline (50%)')
axes[1].set_xticks(x); axes[1].set_xticklabels(ASSETS, rotation=15)
axes[1].set_title("Directional Accuracy (2025-2026)", fontweight='bold'); axes[1].set_ylabel("Accuracy (%)"); axes[1].legend()
plt.suptitle("Model Comparison -- Test: 2025-2026 (Trained: 2005-2024)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_test2526_comparison.png', dpi=150); plt.close()

# Plot 5: Monthly prediction detail table
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, asset in zip(axes.flatten(), ASSETS):
    lr_p = lr_results[asset]['pred']
    rf_p = rf_results[asset]['pred']
    actual = lr_results[asset]['actual']
    months = [d.strftime('%Y-%m') for d in test_index]
    x_pos = np.arange(len(months))
    w = 0.25
    ax.bar(x_pos - w, actual, w, color='#555', alpha=0.5, label='Actual')
    ax.bar(x_pos, lr_p, w, color='red', alpha=0.7, label='LR Pred')
    ax.bar(x_pos + w, rf_p, w, color='blue', alpha=0.7, label='RF Pred')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_xticks(x_pos); ax.set_xticklabels(months, rotation=45, fontsize=7)
    ax.set_title(f"{asset}", fontweight='bold')
    ax.set_ylabel("Monthly Log Return"); ax.legend(fontsize=7)
plt.suptitle("Monthly Predictions: Actual vs LR vs RF (2025-2026)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_test2526_monthly.png', dpi=150); plt.close()

print("\nAll 5 plots saved.")
