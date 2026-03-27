import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

macro_df = pd.read_csv('macro_finaldata_india.csv', parse_dates=['Date'])
macro_df = macro_df.set_index('Date')
macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
macro_df = macro_df.sort_index()

FEATURES = list(macro_df.columns)

print("Data loaded")
print(f"  Asset data  : {asset_df.shape}  |  {asset_df.index[0].date()} -> {asset_df.index[-1].date()}")
print(f"  Macro data  : {macro_df.shape}  |  {macro_df.index[0].date()} -> {macro_df.index[-1].date()}")
print(f"  Features    : {FEATURES}")

# Prepare full dataset
monthly_price = asset_df[['Nifty50_USD']].resample('ME').last().ffill()
monthly_return = np.log(monthly_price / monthly_price.shift(1)).dropna()
macro_lagged = macro_df[FEATURES].shift(1)
ml_data = pd.concat([monthly_return, macro_lagged], axis=1).dropna()

X = ml_data[FEATURES].values
y = ml_data['Nifty50_USD'].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

print(f"\nFull dataset ready")
print(f"  Months    : {len(ml_data)}")
print(f"  Date range: {ml_data.index[0].date()} -> {ml_data.index[-1].date()}")
print(f"  Features  : {len(FEATURES)}")

# LINEAR REGRESSION
lr_model = LinearRegression()
lr_model.fit(Xs, y)
lr_pred = lr_model.predict(Xs)

lr_res = {
    'RMSE': np.sqrt(mean_squared_error(y, lr_pred)),
    'R2': r2_score(y, lr_pred),
    'MAE': mean_absolute_error(y, lr_pred),
    'Dir_Acc': np.mean(np.sign(lr_pred) == np.sign(y)),
    'coef': pd.Series(lr_model.coef_, index=FEATURES),
}

print("\n" + "=" * 60)
print("  LINEAR REGRESSION -- Nifty50_USD (Full: 2005-2024)")
print("=" * 60)
print(f"  RMSE          : {lr_res['RMSE']:.6f}")
print(f"  R2            : {lr_res['R2']:.6f}")
print(f"  MAE           : {lr_res['MAE']:.6f}")
print(f"  Dir. Accuracy : {lr_res['Dir_Acc']:.1%}")

print("\n  FEATURE COEFFICIENTS (sorted by magnitude):")
print(f"  {'Feature':<25} {'Coefficient':>12}  Direction")
print("  " + "-" * 55)
coef_sorted = lr_res['coef'].abs().sort_values(ascending=False)
for f in coef_sorted.index:
    val = lr_res['coef'][f]
    direction = 'positive' if val > 0 else 'negative'
    print(f"  {f:<25} {val:>+12.4f}  {direction}")

# RANDOM FOREST
rf_model = RandomForestRegressor(n_estimators=200, max_depth=5,
                                 min_samples_leaf=3, random_state=42, n_jobs=-1)
rf_model.fit(X, y)
rf_pred = rf_model.predict(X)

rf_res = {
    'RMSE': np.sqrt(mean_squared_error(y, rf_pred)),
    'R2': r2_score(y, rf_pred),
    'MAE': mean_absolute_error(y, rf_pred),
    'Dir_Acc': np.mean(np.sign(rf_pred) == np.sign(y)),
    'importance': pd.Series(rf_model.feature_importances_, index=FEATURES),
}

print("\n" + "=" * 60)
print("  RANDOM FOREST -- Nifty50_USD (Full: 2005-2024)")
print("  (n_estimators=200, max_depth=5, min_samples_leaf=3)")
print("=" * 60)
print(f"  RMSE          : {rf_res['RMSE']:.6f}")
print(f"  R2            : {rf_res['R2']:.6f}")
print(f"  MAE           : {rf_res['MAE']:.6f}")
print(f"  Dir. Accuracy : {rf_res['Dir_Acc']:.1%}")

print("\n  FEATURE IMPORTANCES (sorted):")
print(f"  {'Feature':<25} {'Importance':>12}  Bar")
print("  " + "-" * 60)
imp_sorted = rf_res['importance'].sort_values(ascending=False)
for f, v in imp_sorted.items():
    bar = '#' * int(v / imp_sorted.max() * 25)
    print(f"  {f:<25} {v:>12.4f}  {bar}")

# COMPARISON
print("\n" + "=" * 55)
print("  MODEL COMPARISON -- Nifty50_USD (Full: 2005-2024)")
print("=" * 55)
print(f"  {'Metric':<16} {'Lin.Reg':>12} {'Rand.Forest':>13} {'Winner':>8}")
print("  " + "-" * 52)
metrics = [
    ('RMSE', 'RMSE', False),
    ('R2', 'R2', True),
    ('MAE', 'MAE', False),
    ('Dir. Acc', 'Dir_Acc', True),
]
for label, key, higher_better in metrics:
    lv = lr_res[key]
    rv = rf_res[key]
    winner = 'RF' if (rv > lv if higher_better else rv < lv) else 'LR'
    if key == 'Dir_Acc':
        print(f"  {label:<16} {lv:>11.1%} {rv:>12.1%} {winner:>8}")
    else:
        print(f"  {label:<16} {lv:>12.6f} {rv:>13.6f} {winner:>8}")

# PLOTS
# Plot 1: Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (label, pred, color, res) in zip(axes, [
        ('Linear Regression', lr_pred, 'red', lr_res),
        ('Random Forest', rf_pred, 'blue', rf_res)]):
    ax.bar(ml_data.index, y, width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(ml_data.index, pred, color=color, linewidth=1.2, label='Predicted')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(f"Nifty50_USD -- {label}\nDirAcc={res['Dir_Acc']:.1%}  R2={res['R2']:.4f}  RMSE={res['RMSE']:.4f}",
                 fontweight='bold')
    ax.set_ylabel("Monthly Log Return")
    ax.legend(fontsize=9)
plt.suptitle("Predicted vs Actual -- Nifty50_USD (Full: 2005-2024)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s6_pred.png', dpi=150)
plt.close()

# Plot 2: Feature Importance RF
fig, ax = plt.subplots(figsize=(10, 7))
imp = rf_res['importance'].sort_values()
clrs = ['#1f77b4' if v >= imp.quantile(0.5) else '#aec7e8' for v in imp.values]
ax.barh(imp.index, imp.values, color=clrs, edgecolor='white', alpha=0.85)
ax.set_title("Feature Importance -- Random Forest (Nifty50_USD)", fontweight='bold', fontsize=12)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig('plot_s6_rf_importance.png', dpi=150)
plt.close()

# Plot 3: Feature Coefficients LR
fig, ax = plt.subplots(figsize=(10, 7))
coef = lr_res['coef'].sort_values()
clrs = ['red' if v < 0 else 'steelblue' for v in coef.values]
ax.barh(coef.index, coef.values, color=clrs, edgecolor='white', alpha=0.85)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Feature Coefficients -- Linear Regression (Nifty50_USD)\nBlue = positive  |  Red = negative",
             fontweight='bold', fontsize=12)
ax.set_xlabel("Coefficient (standardised features)")
plt.tight_layout()
plt.savefig('plot_s6_lr_coef.png', dpi=150)
plt.close()

# Plot 4: RMSE & Dir.Acc comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Linear Reg', 'Random Forest'],
            [lr_res['RMSE'], rf_res['RMSE']],
            color=['#1f77b4', '#d62728'], width=0.4, edgecolor='white', alpha=0.85)
for i, v in enumerate([lr_res['RMSE'], rf_res['RMSE']]):
    axes[0].text(i, v + 0.0001, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
axes[0].set_title("RMSE (lower = better)", fontweight='bold')
axes[0].set_ylabel("RMSE")

axes[1].bar(['Linear Reg', 'Random Forest'],
            [lr_res['Dir_Acc']*100, rf_res['Dir_Acc']*100],
            color=['#1f77b4', '#d62728'], width=0.4, edgecolor='white', alpha=0.85)
for i, v in enumerate([lr_res['Dir_Acc']*100, rf_res['Dir_Acc']*100]):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[1].axhline(50, color='black', linestyle='--', linewidth=1, label='Baseline 50%')
axes[1].set_title("Directional Accuracy (higher = better)", fontweight='bold')
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
plt.suptitle("Model Comparison -- Nifty50_USD (Full: 2005-2024)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s6_comparison.png', dpi=150)
plt.close()

print("\nAll 4 plots saved.")
