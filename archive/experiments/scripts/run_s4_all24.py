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

# Load data
asset_df = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date'])
asset_df = asset_df.set_index('Date')
asset_df.index = pd.to_datetime(asset_df.index).tz_localize(None)
asset_df = asset_df.sort_index()

macro_df = pd.read_csv('macro_finaldata_2005_2024.csv', parse_dates=['Date'])
macro_df = macro_df.set_index('Date')
macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
macro_df = macro_df.sort_index()

ASSETS = ['SP500', 'Gold', 'USBond']
FEATURES = list(macro_df.columns)
COLORS = {'SP500': '#ff7f0e', 'Gold': '#FFD700', 'USBond': '#2ca02c'}

print("Data loaded")
print(f"  Asset data  : {asset_df.shape}  |  {asset_df.index[0].date()} -> {asset_df.index[-1].date()}")
print(f"  Macro data  : {macro_df.shape}  |  {macro_df.index[0].date()} -> {macro_df.index[-1].date()}")
print(f"  Features    : {len(FEATURES)}")
print(f"  Features    : {FEATURES}")

# Prepare dataset
TRAIN_END = '2019-12-31'
TEST_START = '2020-01-01'

monthly_prices = asset_df[ASSETS].resample('ME').last().ffill()
monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
macro_lagged = macro_df[FEATURES].shift(1)
ml_data = pd.concat([monthly_returns, macro_lagged], axis=1).dropna()

train = ml_data.loc[:TRAIN_END]
test = ml_data.loc[TEST_START:]
X_train = train[FEATURES].values
X_test = test[FEATURES].values

scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test = scaler.transform(X_test)

print(f"\nML dataset ready")
print(f"  Total months : {len(ml_data)}")
print(f"  Train        : {len(train)} months ({train.index[0].date()} -> {train.index[-1].date()})")
print(f"  Test         : {len(test)} months ({test.index[0].date()} -> {test.index[-1].date()})")

# LINEAR REGRESSION
lr_results = {}
lr_preds = {}
print("\n" + "=" * 80)
print("  LINEAR REGRESSION  --  All 24 Macro Features -> Monthly Returns")
print("=" * 80)
print(f"  {'Asset':<10} {'Train RMSE':>12} {'Test RMSE':>12} {'Train R2':>10} {'Test R2':>10} {'MAE':>10} {'Dir.Acc':>10}")
print("  " + "-" * 75)

for asset in ASSETS:
    y_train = train[asset].values
    y_test = test[asset].values
    model = LinearRegression()
    model.fit(Xs_train, y_train)
    yp_tr = model.predict(Xs_train)
    yp_te = model.predict(Xs_test)
    res = {
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, yp_tr)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, yp_te)),
        'Train_R2': r2_score(y_train, yp_tr),
        'Test_R2': r2_score(y_test, yp_te),
        'Test_MAE': mean_absolute_error(y_test, yp_te),
        'Dir_Acc': np.mean(np.sign(yp_te) == np.sign(y_test)),
        'coef': pd.Series(model.coef_, index=FEATURES),
        'y_pred': yp_te,
        'y_actual': y_test,
    }
    lr_results[asset] = res
    lr_preds[asset] = yp_te
    print(f"  {asset:<10} {res['Train_RMSE']:>12.6f} {res['Test_RMSE']:>12.6f} "
          f"{res['Train_R2']:>10.4f} {res['Test_R2']:>10.4f} "
          f"{res['Test_MAE']:>10.6f} {res['Dir_Acc']:>9.1%}")

print("\n  TOP 5 FEATURES BY COEFFICIENT MAGNITUDE (Linear Regression):")
for asset in ASSETS:
    coef = lr_results[asset]['coef']
    top5 = coef.abs().nlargest(5).index
    print(f"\n  {asset}:")
    for f in top5:
        print(f"    {f:<35} {coef[f]:>+8.4f}")

# RANDOM FOREST
rf_results = {}
rf_preds = {}
print("\n" + "=" * 80)
print("  RANDOM FOREST  --  All 24 Macro Features -> Monthly Returns")
print("  (n_estimators=200, max_depth=5, min_samples_leaf=3)")
print("=" * 80)
print(f"  {'Asset':<10} {'Train RMSE':>12} {'Test RMSE':>12} {'Train R2':>10} {'Test R2':>10} {'MAE':>10} {'Dir.Acc':>10}")
print("  " + "-" * 75)

for asset in ASSETS:
    y_train = train[asset].values
    y_test = test[asset].values
    model = RandomForestRegressor(n_estimators=200, max_depth=5,
                                  min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    yp_tr = model.predict(X_train)
    yp_te = model.predict(X_test)
    res = {
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, yp_tr)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, yp_te)),
        'Train_R2': r2_score(y_train, yp_tr),
        'Test_R2': r2_score(y_test, yp_te),
        'Test_MAE': mean_absolute_error(y_test, yp_te),
        'Dir_Acc': np.mean(np.sign(yp_te) == np.sign(y_test)),
        'importance': pd.Series(model.feature_importances_, index=FEATURES),
        'y_pred': yp_te,
        'y_actual': y_test,
    }
    rf_results[asset] = res
    rf_preds[asset] = yp_te
    print(f"  {asset:<10} {res['Train_RMSE']:>12.6f} {res['Test_RMSE']:>12.6f} "
          f"{res['Train_R2']:>10.4f} {res['Test_R2']:>10.4f} "
          f"{res['Test_MAE']:>10.6f} {res['Dir_Acc']:>9.1%}")

print("\n  TOP 5 FEATURES BY IMPORTANCE (Random Forest):")
for asset in ASSETS:
    imp = rf_results[asset]['importance']
    top5 = imp.nlargest(5)
    print(f"\n  {asset}:")
    for f, v in top5.items():
        bar = '#' * int(v / imp.max() * 20)
        print(f"    {f:<35} {v:.4f}  {bar}")

# MODEL COMPARISON
print("\n" + "=" * 70)
print("  MODEL COMPARISON SUMMARY  --  Test Set (2020-2024)")
print("=" * 70)
print(f"  {'Asset':<10} {'Metric':<14} {'Lin.Reg':>10} {'Rand.Forest':>13} {'Winner':>10}")
print("  " + "-" * 58)
for asset in ASSETS:
    lr = lr_results[asset]
    rf = rf_results[asset]
    print(f"  {asset:<10} {'Test RMSE':<14} {lr['Test_RMSE']:>10.6f} "
          f"{rf['Test_RMSE']:>13.6f} "
          f"{'RF' if rf['Test_RMSE'] < lr['Test_RMSE'] else 'LR':>10}")
    print(f"  {'':<10} {'Test R2':<14} {lr['Test_R2']:>10.4f} "
          f"{rf['Test_R2']:>13.4f} "
          f"{'RF' if rf['Test_R2'] > lr['Test_R2'] else 'LR':>10}")
    print(f"  {'':<10} {'Dir. Acc':<14} {lr['Dir_Acc']:>9.1%} "
          f"{rf['Dir_Acc']:>12.1%} "
          f"{'RF' if rf['Dir_Acc'] > lr['Dir_Acc'] else 'LR':>10}")
    print()

# PLOTS
test_index = test.index

# Plot 1: LR pred vs actual
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, asset in zip(axes, ASSETS):
    ax.bar(test_index, lr_results[asset]['y_actual'], width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(test_index, lr_results[asset]['y_pred'], color='red', linewidth=1.5, marker='o', markersize=3, label='Predicted')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(f"{asset}\nDirAcc={lr_results[asset]['Dir_Acc']:.1%}  R2={lr_results[asset]['Test_R2']:.4f}", fontweight='bold')
    ax.set_ylabel("Monthly Log Return")
    ax.legend(fontsize=8)
plt.suptitle("Linear Regression - Predicted vs Actual (Test: 2020-2024)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s4_lr_pred.png', dpi=150)
plt.close()

# Plot 2: RF pred vs actual
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, asset in zip(axes, ASSETS):
    ax.bar(test_index, rf_results[asset]['y_actual'], width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(test_index, rf_results[asset]['y_pred'], color='blue', linewidth=1.5, marker='o', markersize=3, label='Predicted')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(f"{asset}\nDirAcc={rf_results[asset]['Dir_Acc']:.1%}  R2={rf_results[asset]['Test_R2']:.4f}", fontweight='bold')
    ax.set_ylabel("Monthly Log Return")
    ax.legend(fontsize=8)
plt.suptitle("Random Forest - Predicted vs Actual (Test: 2020-2024)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s4_rf_pred.png', dpi=150)
plt.close()

# Plot 3: Feature Importance RF
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for ax, asset in zip(axes, ASSETS):
    imp = rf_results[asset]['importance'].sort_values()
    ax.barh(imp.index, imp.values, color=COLORS[asset], edgecolor='white', alpha=0.85)
    ax.set_title(f"{asset}", fontweight='bold')
    ax.set_xlabel("Importance")
plt.suptitle("Feature Importance - Random Forest (All 24 Features)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s4_rf_importance.png', dpi=150)
plt.close()

# Plot 4: Top 5 Coefficients LR
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, asset in zip(axes, ASSETS):
    coef = lr_results[asset]['coef']
    top5 = coef.abs().nlargest(5)
    vals = [coef[f] for f in top5.index]
    clrs = ['red' if v < 0 else 'steelblue' for v in vals]
    ax.barh(top5.index, vals, color=clrs, edgecolor='white', alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f"{asset}", fontweight='bold')
    ax.set_xlabel("Coefficient (standardised)")
plt.suptitle("Top 5 Feature Coefficients - Linear Regression", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s4_lr_coef.png', dpi=150)
plt.close()

# Plot 5: RMSE & Dir Acc comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(ASSETS))
width = 0.35
axes[0].bar(x - width/2, [lr_results[a]['Test_RMSE'] for a in ASSETS], width, label='Linear Regression', color='#1f77b4', alpha=0.85)
axes[0].bar(x + width/2, [rf_results[a]['Test_RMSE'] for a in ASSETS], width, label='Random Forest', color='#d62728', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(ASSETS)
axes[0].set_title("Test RMSE Comparison", fontweight='bold')
axes[0].set_ylabel("RMSE")
axes[0].legend()

axes[1].bar(x - width/2, [lr_results[a]['Dir_Acc']*100 for a in ASSETS], width, label='Linear Regression', color='#1f77b4', alpha=0.85)
axes[1].bar(x + width/2, [rf_results[a]['Dir_Acc']*100 for a in ASSETS], width, label='Random Forest', color='#d62728', alpha=0.85)
axes[1].axhline(50, color='black', linestyle='--', linewidth=1, label='Baseline (50%)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(ASSETS)
axes[1].set_title("Directional Accuracy (%)", fontweight='bold')
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
plt.suptitle("Model Comparison - Test Set (2020-2024)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s4_comparison.png', dpi=150)
plt.close()

# Plot 6: Correlation heatmap
corr = monthly_returns.corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, fmt='.4f', cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, linecolor='white', cbar_kws={'label': 'Pearson r'})
ax.set_title("Correlation - Monthly Log Returns (2005-2024)", fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s4_corr.png', dpi=150)
plt.close()

print("\nAll 6 plots saved.")
