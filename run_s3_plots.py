import pandas as pd, numpy as np, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

asset_df = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date'])
asset_df = asset_df.set_index('Date')
asset_df.index = pd.to_datetime(asset_df.index).tz_localize(None)
asset_df = asset_df.sort_index()

macro_df = pd.read_csv('macro_finaldata_2005_2024.csv', parse_dates=['Date'])
macro_df = macro_df.set_index('Date')
macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
macro_df = macro_df.sort_index()

ASSETS = ['SP500', 'Gold', 'USBond']
COLORS = {'SP500': '#ff7f0e', 'Gold': '#FFD700', 'USBond': '#2ca02c'}

SP500_FEATURES = ['US_CPI_YoY','US_Fed_Funds_Rate','US_10Y_Yield','US_VIX','US_Unemployment_Rate',
                  'US_Retail_Sales_YoY','US_CFNAI','US_M2_YoY','US_Credit_Spread','US_Yield_Curve_Spread']
GOLD_FEATURES = ['US_CPI_YoY','US_Fed_Funds_Rate','US_10Y_Yield','US_VIX','DXY_Dollar_Index',
                 'US_M2_YoY','US_Credit_Spread','Oil_Brent_USD','US_Unemployment_Rate']
USBOND_FEATURES = ['US_CPI_YoY','US_Fed_Funds_Rate','US_10Y_Yield','US_2Y_Yield','US_Yield_Curve_Spread',
                   'US_Credit_Spread','US_VIX','US_M2_YoY','US_Unemployment_Rate']
FEATURE_MAP = {'SP500': SP500_FEATURES, 'Gold': GOLD_FEATURES, 'USBond': USBOND_FEATURES}

TRAIN_END = '2019-12-31'
TEST_START = '2020-01-01'

monthly_prices = asset_df[ASSETS].resample('ME').last().ffill()
monthly_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
macro_lagged = macro_df.shift(1)

def build_dataset(asset, features):
    data = pd.concat([monthly_returns[[asset]], macro_lagged[features]], axis=1).dropna()
    train = data.loc[:TRAIN_END]
    test = data.loc[TEST_START:]
    return {
        'train': train, 'test': test,
        'X_train': train[features].values, 'X_test': test[features].values,
        'y_train': train[asset].values, 'y_test': test[asset].values,
        'features': features,
    }

datasets = {a: build_dataset(a, FEATURE_MAP[a]) for a in ASSETS}

# Train models
lr_preds = {}
rf_preds = {}
lr_results = {}
rf_results = {}

for asset, d in datasets.items():
    sc = StandardScaler()
    Xtr = sc.fit_transform(d['X_train'])
    Xte = sc.transform(d['X_test'])
    lr = LinearRegression().fit(Xtr, d['y_train'])
    lr_preds[asset] = lr.predict(Xte)
    lr_results[asset] = {
        'Test_RMSE': np.sqrt(mean_squared_error(d['y_test'], lr_preds[asset])),
        'Test_R2': r2_score(d['y_test'], lr_preds[asset]),
        'Dir_Acc': np.mean(np.sign(lr_preds[asset]) == np.sign(d['y_test'])),
    }
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3,
                               random_state=42, n_jobs=-1)
    rf.fit(d['X_train'], d['y_train'])
    rf_preds[asset] = rf.predict(d['X_test'])
    rf_results[asset] = {
        'Test_RMSE': np.sqrt(mean_squared_error(d['y_test'], rf_preds[asset])),
        'Test_R2': r2_score(d['y_test'], rf_preds[asset]),
        'Dir_Acc': np.mean(np.sign(rf_preds[asset]) == np.sign(d['y_test'])),
        'importance': pd.Series(rf.feature_importances_, index=d['features']),
    }

test_index = datasets['SP500']['test'].index

# Plot 1: LR pred vs actual
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, asset in zip(axes, ASSETS):
    ax.bar(test_index, datasets[asset]['y_test'], width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(test_index, lr_preds[asset], color='red', linewidth=1.5, marker='o', markersize=3, label='Predicted (LR)')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(asset, fontweight='bold')
    ax.set_ylabel('Monthly Log Return')
    ax.legend(fontsize=8)
plt.suptitle('Linear Regression - Predicted vs Actual (2020-2024)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s3_lr_pred.png', dpi=150)
plt.close()

# Plot 2: RF pred vs actual
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, asset in zip(axes, ASSETS):
    ax.bar(test_index, datasets[asset]['y_test'], width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(test_index, rf_preds[asset], color='blue', linewidth=1.5, marker='o', markersize=3, label='Predicted (RF)')
    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(asset, fontweight='bold')
    ax.set_ylabel('Monthly Log Return')
    ax.legend(fontsize=8)
plt.suptitle('Random Forest - Predicted vs Actual (2020-2024)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s3_rf_pred.png', dpi=150)
plt.close()

# Plot 3: Feature importance RF
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, asset in zip(axes, ASSETS):
    imp = rf_results[asset]['importance'].sort_values()
    ax.barh(imp.index, imp.values, color=COLORS[asset], edgecolor='white', alpha=0.85)
    ax.set_title(asset, fontweight='bold')
    ax.set_xlabel('Importance')
plt.suptitle('Feature Importance - Random Forest (2005-2019 Training)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s3_rf_importance.png', dpi=150)
plt.close()

# Plot 4: RMSE & Dir Acc comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(ASSETS))
width = 0.35
axes[0].bar(x - width/2, [lr_results[a]['Test_RMSE'] for a in ASSETS], width,
            label='Linear Regression', color='#1f77b4', alpha=0.85)
axes[0].bar(x + width/2, [rf_results[a]['Test_RMSE'] for a in ASSETS], width,
            label='Random Forest', color='#d62728', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(ASSETS)
axes[0].set_title('Test RMSE Comparison', fontweight='bold')
axes[0].set_ylabel('RMSE')
axes[0].legend()
axes[1].bar(x - width/2, [lr_results[a]['Dir_Acc'] * 100 for a in ASSETS], width,
            label='Linear Regression', color='#1f77b4', alpha=0.85)
axes[1].bar(x + width/2, [rf_results[a]['Dir_Acc'] * 100 for a in ASSETS], width,
            label='Random Forest', color='#d62728', alpha=0.85)
axes[1].axhline(50, color='black', linestyle='--', linewidth=1, label='Random baseline (50%)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(ASSETS)
axes[1].set_title('Directional Accuracy (%)', fontweight='bold')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
plt.suptitle('Model Comparison - Test Set (2020-2024)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s3_comparison.png', dpi=150)
plt.close()

# Plot 5: Correlation heatmap monthly
corr = monthly_returns.corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, fmt='.4f', cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, linecolor='white', cbar_kws={'label': 'Pearson r'})
ax.set_title('Correlation - Monthly Log Returns (2005-2024)', fontweight='bold')
plt.tight_layout()
plt.savefig('plot_s3_corr_monthly.png', dpi=150)
plt.close()

print('All 5 plots saved.')
