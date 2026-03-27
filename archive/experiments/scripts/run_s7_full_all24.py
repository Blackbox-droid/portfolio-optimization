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

# Prepare full dataset
monthly_prices = asset_df[ASSETS].resample('ME').last().ffill()
monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
macro_lagged = macro_df[FEATURES].shift(1)
ml_data = pd.concat([monthly_returns, macro_lagged], axis=1).dropna()

X = ml_data[FEATURES].values
sc = StandardScaler()
Xs = sc.fit_transform(X)

print(f"\nFull dataset ready")
print(f"  Total months : {len(ml_data)}")
print(f"  Date range   : {ml_data.index[0].date()} -> {ml_data.index[-1].date()}")
print(f"  Features     : {len(FEATURES)}")

# LINEAR REGRESSION
lr_results = {}
print("\n" + "=" * 75)
print("  LINEAR REGRESSION  --  All 24 Macro Features (Full: 2005-2024)")
print("=" * 75)
print(f"  {'Asset':<10} {'RMSE':>12} {'R2':>10} {'MAE':>10} {'Dir.Acc':>10}")
print("  " + "-" * 48)

for asset in ASSETS:
    y = ml_data[asset].values
    model = LinearRegression().fit(Xs, y)
    pred = model.predict(Xs)
    res = {
        'RMSE': np.sqrt(mean_squared_error(y, pred)),
        'R2': r2_score(y, pred),
        'MAE': mean_absolute_error(y, pred),
        'Dir_Acc': np.mean(np.sign(pred) == np.sign(y)),
        'coef': pd.Series(model.coef_, index=FEATURES),
        'pred': pred, 'actual': y,
    }
    lr_results[asset] = res
    print(f"  {asset:<10} {res['RMSE']:>12.6f} {res['R2']:>10.4f} {res['MAE']:>10.6f} {res['Dir_Acc']:>9.1%}")

print("\n  TOP 5 FEATURES BY COEFFICIENT MAGNITUDE (Linear Regression):")
for asset in ASSETS:
    coef = lr_results[asset]['coef']
    top5 = coef.abs().nlargest(5).index
    print(f"\n  {asset}:")
    for f in top5:
        val = coef[f]
        d = 'positive' if val > 0 else 'negative'
        print(f"    {f:<35} {val:>+8.4f}  {d}")

# RANDOM FOREST
rf_results = {}
print("\n" + "=" * 75)
print("  RANDOM FOREST  --  All 24 Macro Features (Full: 2005-2024)")
print("  (n_estimators=200, max_depth=5, min_samples_leaf=3)")
print("=" * 75)
print(f"  {'Asset':<10} {'RMSE':>12} {'R2':>10} {'MAE':>10} {'Dir.Acc':>10}")
print("  " + "-" * 48)

for asset in ASSETS:
    y = ml_data[asset].values
    model = RandomForestRegressor(n_estimators=200, max_depth=5,
                                  min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X, y)
    pred = model.predict(X)
    res = {
        'RMSE': np.sqrt(mean_squared_error(y, pred)),
        'R2': r2_score(y, pred),
        'MAE': mean_absolute_error(y, pred),
        'Dir_Acc': np.mean(np.sign(pred) == np.sign(y)),
        'importance': pd.Series(model.feature_importances_, index=FEATURES),
        'pred': pred, 'actual': y,
    }
    rf_results[asset] = res
    print(f"  {asset:<10} {res['RMSE']:>12.6f} {res['R2']:>10.4f} {res['MAE']:>10.6f} {res['Dir_Acc']:>9.1%}")

print("\n  TOP 5 FEATURES BY IMPORTANCE (Random Forest):")
for asset in ASSETS:
    imp = rf_results[asset]['importance']
    top5 = imp.nlargest(5)
    print(f"\n  {asset}:")
    for f, v in top5.items():
        bar = '#' * int(v / imp.max() * 20)
        print(f"    {f:<35} {v:.4f}  {bar}")

# COMPARISON
print("\n" + "=" * 65)
print("  MODEL COMPARISON SUMMARY  --  Full Data (2005-2024)")
print("=" * 65)
print(f"  {'Asset':<10} {'Metric':<12} {'Lin.Reg':>10} {'Rand.Forest':>13} {'Winner':>8}")
print("  " + "-" * 55)
for asset in ASSETS:
    lr = lr_results[asset]; rf = rf_results[asset]
    print(f"  {asset:<10} {'RMSE':<12} {lr['RMSE']:>10.6f} {rf['RMSE']:>13.6f} {'RF' if rf['RMSE'] < lr['RMSE'] else 'LR':>8}")
    print(f"  {'':<10} {'R2':<12} {lr['R2']:>10.4f} {rf['R2']:>13.4f} {'RF' if rf['R2'] > lr['R2'] else 'LR':>8}")
    print(f"  {'':<10} {'Dir. Acc':<12} {lr['Dir_Acc']:>9.1%} {rf['Dir_Acc']:>12.1%} {'RF' if rf['Dir_Acc'] > lr['Dir_Acc'] else 'LR':>8}")
    print()

# PLOTS
# Plot 1: LR pred vs actual
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, asset in zip(axes, ASSETS):
    r = lr_results[asset]
    ax.bar(ml_data.index, r['actual'], width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(ml_data.index, r['pred'], color='red', linewidth=1.2, label='Predicted')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(f"{asset}\nDirAcc={r['Dir_Acc']:.1%}  R2={r['R2']:.4f}", fontweight='bold')
    ax.set_ylabel("Monthly Log Return"); ax.legend(fontsize=8)
plt.suptitle("Linear Regression -- Predicted vs Actual (Full: 2005-2024)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_s7_lr_pred.png', dpi=150); plt.close()

# Plot 2: RF pred vs actual
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, asset in zip(axes, ASSETS):
    r = rf_results[asset]
    ax.bar(ml_data.index, r['actual'], width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(ml_data.index, r['pred'], color='blue', linewidth=1.2, label='Predicted')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(f"{asset}\nDirAcc={r['Dir_Acc']:.1%}  R2={r['R2']:.4f}", fontweight='bold')
    ax.set_ylabel("Monthly Log Return"); ax.legend(fontsize=8)
plt.suptitle("Random Forest -- Predicted vs Actual (Full: 2005-2024)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_s7_rf_pred.png', dpi=150); plt.close()

# Plot 3: Feature Importance RF
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
for ax, asset in zip(axes, ASSETS):
    imp = rf_results[asset]['importance'].sort_values()
    ax.barh(imp.index, imp.values, color=COLORS[asset], edgecolor='white', alpha=0.85)
    ax.set_title(f"{asset}", fontweight='bold'); ax.set_xlabel("Importance")
plt.suptitle("Feature Importance -- Random Forest (All 24 Features)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_s7_rf_importance.png', dpi=150); plt.close()

# Plot 4: Top 5 Coefficients LR
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, asset in zip(axes, ASSETS):
    coef = lr_results[asset]['coef']
    top5 = coef.abs().nlargest(5)
    vals = [coef[f] for f in top5.index]
    clrs = ['red' if v < 0 else 'steelblue' for v in vals]
    ax.barh(top5.index, vals, color=clrs, edgecolor='white', alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f"{asset}", fontweight='bold'); ax.set_xlabel("Coefficient (standardised)")
plt.suptitle("Top 5 Feature Coefficients -- Linear Regression", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_s7_lr_coef.png', dpi=150); plt.close()

# Plot 5: RMSE & Dir Acc comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(ASSETS)); width = 0.35
axes[0].bar(x - width/2, [lr_results[a]['RMSE'] for a in ASSETS], width, label='Linear Regression', color='#1f77b4', alpha=0.85)
axes[0].bar(x + width/2, [rf_results[a]['RMSE'] for a in ASSETS], width, label='Random Forest', color='#d62728', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(ASSETS)
axes[0].set_title("RMSE Comparison", fontweight='bold'); axes[0].set_ylabel("RMSE"); axes[0].legend()

axes[1].bar(x - width/2, [lr_results[a]['Dir_Acc']*100 for a in ASSETS], width, label='Linear Regression', color='#1f77b4', alpha=0.85)
axes[1].bar(x + width/2, [rf_results[a]['Dir_Acc']*100 for a in ASSETS], width, label='Random Forest', color='#d62728', alpha=0.85)
axes[1].axhline(50, color='black', linestyle='--', linewidth=1, label='Baseline (50%)')
axes[1].set_xticks(x); axes[1].set_xticklabels(ASSETS)
axes[1].set_title("Directional Accuracy (%)", fontweight='bold'); axes[1].set_ylabel("Accuracy (%)"); axes[1].legend()
plt.suptitle("Model Comparison -- Full Data (2005-2024)", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('plot_s7_comparison.png', dpi=150); plt.close()

# Plot 6: Correlation heatmap
corr = monthly_returns.corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, fmt='.4f', cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, linecolor='white', cbar_kws={'label': 'Pearson r'})
ax.set_title("Correlation -- Monthly Log Returns (2005-2024)", fontweight='bold')
plt.tight_layout(); plt.savefig('plot_s7_corr.png', dpi=150); plt.close()

print("\nAll 6 plots saved.")
