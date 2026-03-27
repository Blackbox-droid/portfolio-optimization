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
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════
# Asset prices (daily)
asset_tr = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date']).set_index('Date').sort_index()
asset_tr.index = pd.to_datetime(asset_tr.index).tz_localize(None)
asset_te = pd.read_csv('data_daily_2025_2026.csv', parse_dates=['Date']).set_index('Date').sort_index()
asset_te.index = pd.to_datetime(asset_te.index).tz_localize(None)

# Macro features (monthly)
macro_tr = pd.read_csv('macro_finaldata_2005_2024.csv', parse_dates=['Date']).set_index('Date').sort_index()
macro_tr.index = pd.to_datetime(macro_tr.index).tz_localize(None)
macro_te = pd.read_csv('macro_finaldata_2025_2026.csv', parse_dates=['Date']).set_index('Date').sort_index()
macro_te.index = pd.to_datetime(macro_te.index).tz_localize(None)

MACRO_COLS = list(macro_tr.columns)
TARGETS = ['SP500', 'Gold', 'USBond']

print("=" * 85)
print("  ML RETURN PREDICTION -- SP500, Gold, USBond")
print("  Train: 2005-2024 | Test: 2025 onwards")
print("  Models: Linear Regression, SVR, Gradient Boosting, Random Forest")
print("=" * 85)

# ════════════════════════════════════════════════════════════════════
# 2. PREPARE DATA
# ════════════════════════════════════════════════════════════════════
def prepare_data(asset_df, macro_df, targets):
    """Convert daily prices to monthly log returns, merge with lagged macro features."""
    monthly_price = asset_df[targets].resample('ME').last().ffill()
    monthly_ret   = np.log(monthly_price / monthly_price.shift(1)).dropna()

    macro_m = macro_df[MACRO_COLS].resample('ME').last().ffill().bfill().fillna(0)

    # Engineered change features
    change_pairs = {
        'Oil_Change':    'Oil_Brent_USD',
        'Gold_Change':   'Gold_USD',
        'DXY_Change':    'DXY_Dollar_Index',
        'Forex_Change':  'IN_Forex_Reserves_USD',
        'US_IP_Change':  'US_Industrial_Production',
        'VIX_Change':    'US_VIX',
    }
    for new_col, src_col in change_pairs.items():
        if src_col in macro_m.columns:
            macro_m[new_col] = macro_m[src_col].pct_change()

    feat_cols = list(macro_m.columns)
    features_lagged = macro_m[feat_cols].shift(1)
    merged = pd.concat([monthly_ret, features_lagged], axis=1).dropna()
    merged = merged.replace([np.inf, -np.inf], 0)
    return merged, feat_cols

train_data, FEATURE_COLS = prepare_data(asset_tr, macro_tr, TARGETS)
test_data, _             = prepare_data(asset_te, macro_te, TARGETS)

print(f"\n  Train: {len(train_data)} months ({train_data.index[0].date()} -> {train_data.index[-1].date()})")
print(f"  Test : {len(test_data)} months ({test_data.index[0].date()} -> {test_data.index[-1].date()})")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Feature list: {FEATURE_COLS}")

# ════════════════════════════════════════════════════════════════════
# 3. DEFINE MODELS
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

# ════════════════════════════════════════════════════════════════════
# 4. TRAIN & EVALUATE PER TARGET ASSET
# ════════════════════════════════════════════════════════════════════
all_results = {}  # {target: {model_name: {metrics + predictions}}}

for target in TARGETS:
    print("\n" + "=" * 85)
    print(f"  TARGET: {target}")
    print("=" * 85)

    X_train = train_data[FEATURE_COLS].values
    y_train = train_data[target].values
    X_test  = test_data[FEATURE_COLS].values
    y_test  = test_data[target].values

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    asset_results = {}

    for mname, (model_template, needs_scaling) in MODELS.items():
        mdl = clone(model_template)
        Xtr = Xs_train if needs_scaling else X_train
        Xte = Xs_test  if needs_scaling else X_test

        mdl.fit(Xtr, y_train)
        pred_train = mdl.predict(Xtr)
        pred_test  = mdl.predict(Xte)

        # Train metrics
        tr_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
        tr_mae  = mean_absolute_error(y_train, pred_train)
        tr_r2   = r2_score(y_train, pred_train)
        tr_dacc = np.mean(np.sign(pred_train) == np.sign(y_train))

        # Test metrics
        te_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        te_mae  = mean_absolute_error(y_test, pred_test)
        te_r2   = r2_score(y_test, pred_test)
        te_dacc = np.mean(np.sign(pred_test) == np.sign(y_test))

        asset_results[mname] = {
            'model'     : mdl,
            'pred_train': pred_train,
            'pred_test' : pred_test,
            'tr_rmse': tr_rmse, 'tr_mae': tr_mae, 'tr_r2': tr_r2, 'tr_dacc': tr_dacc,
            'te_rmse': te_rmse, 'te_mae': te_mae, 'te_r2': te_r2, 'te_dacc': te_dacc,
        }

    all_results[target] = asset_results

    # -- Print summary table --
    print(f"\n  {'Model':<22} {'Train RMSE':>10} {'Train R2':>10} {'Train Dir':>10} | {'Test RMSE':>10} {'Test MAE':>10} {'Test R2':>10} {'Test Dir':>10}")
    print("  " + "-" * 105)
    for mname, res in asset_results.items():
        print(f"  {mname:<22} {res['tr_rmse']:>10.6f} {res['tr_r2']:>10.4f} {res['tr_dacc']:>9.1%} | {res['te_rmse']:>10.6f} {res['te_mae']:>10.6f} {res['te_r2']:>10.4f} {res['te_dacc']:>9.1%}")

    # -- Month-by-month predictions --
    print(f"\n  MONTH-BY-MONTH PREDICTIONS ({target}):")
    header = f"  {'Month':<10} {'Actual':>8}"
    for mname in MODELS:
        header += f" {mname[:10]:>12}"
    print(header)
    print("  " + "-" * (10 + 8 + 12 * len(MODELS) + 2))
    for i, date in enumerate(test_data.index):
        row = f"  {date.strftime('%Y-%m'):<10} {y_test[i]:>8.4f}"
        for mname in MODELS:
            row += f" {asset_results[mname]['pred_test'][i]:>12.4f}"
        print(row)

    # -- Feature importance (RF and GB) --
    print(f"\n  TOP 10 FEATURE IMPORTANCES ({target}):")
    for mname in ['Random Forest', 'Gradient Boosting']:
        mdl = asset_results[mname]['model']
        imp = pd.Series(mdl.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        top10 = imp.head(10)
        print(f"\n  {mname}:")
        for feat, val in top10.items():
            bar = "#" * int(val / imp.max() * 30)
            print(f"    {feat:<30} {val:.4f}  {bar}")

# ════════════════════════════════════════════════════════════════════
# 5. CROSS-ASSET COMPARISON
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 85)
print("  CROSS-ASSET MODEL COMPARISON (TEST SET)")
print("=" * 85)

print(f"\n  {'Asset':<10} {'Model':<22} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'Dir Acc':>10}")
print("  " + "-" * 72)
for target in TARGETS:
    for mname in MODELS:
        res = all_results[target][mname]
        print(f"  {target:<10} {mname:<22} {res['te_rmse']:>10.6f} {res['te_mae']:>10.6f} {res['te_r2']:>10.4f} {res['te_dacc']:>9.1%}")
    print()

# Best model per asset
print("  BEST MODEL PER ASSET (by Test RMSE):")
print("  " + "-" * 55)
for target in TARGETS:
    best = min(all_results[target].items(), key=lambda x: x[1]['te_rmse'])
    print(f"  {target:<10} -> {best[0]:<22} (RMSE={best[1]['te_rmse']:.6f}, R2={best[1]['te_r2']:.4f}, Dir={best[1]['te_dacc']:.1%})")

# ════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ════════════════════════════════════════════════════════════════════

model_colors = {
    'Linear Regression': '#1f77b4',
    'SVR (RBF)':         '#ff7f0e',
    'Random Forest':     '#2ca02c',
    'Gradient Boosting': '#d62728',
}

# -- Plot 1: Actual vs Predicted (Test) -- one subplot per model, one figure per asset --
for target in TARGETS:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    y_test_t = test_data[target].values
    for ax, mname in zip(axes.flatten(), MODELS):
        pred = all_results[target][mname]['pred_test']
        rmse = all_results[target][mname]['te_rmse']
        r2   = all_results[target][mname]['te_r2']
        dacc = all_results[target][mname]['te_dacc']

        ax.bar(test_data.index, y_test_t, width=20, color='#555555', alpha=0.4,
               label='Actual', zorder=2)
        ax.plot(test_data.index, pred, '-o', color=model_colors[mname],
                lw=2, ms=6, label='Predicted', zorder=3)
        ax.axhline(0, color='black', lw=0.7, ls=':')
        ax.set_title(f"{mname} (RMSE={rmse:.4f}, R²={r2:.4f}, Dir={dacc:.0%})",
                     fontweight='bold', fontsize=10)
        ax.set_ylabel('Monthly Log Return')
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f'{target} -- Test Predictions (2025-2026)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'ml_{target.lower()}_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

# -- Plot 2: Train fit -- scatter plot actual vs predicted --
for target in TARGETS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    y_train_t = train_data[target].values
    for ax, mname in zip(axes.flatten(), MODELS):
        pred = all_results[target][mname]['pred_train']
        r2   = all_results[target][mname]['tr_r2']
        ax.scatter(y_train_t, pred, alpha=0.5, s=25, color=model_colors[mname], edgecolors='white')
        lims = [min(y_train_t.min(), pred.min()), max(y_train_t.max(), pred.max())]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.7, label='Perfect fit')
        ax.set_xlabel('Actual Return')
        ax.set_ylabel('Predicted Return')
        ax.set_title(f"{mname} (Train R²={r2:.4f})", fontweight='bold', fontsize=10)
        ax.legend(fontsize=9)

    plt.suptitle(f'{target} -- Train: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'ml_{target.lower()}_train_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

# -- Plot 3: Feature importance comparison (RF vs GB) --
for target in TARGETS:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, mname in zip(axes, ['Random Forest', 'Gradient Boosting']):
        mdl = all_results[target][mname]['model']
        imp = pd.Series(mdl.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
        top15 = imp.tail(15)
        ax.barh(top15.index, top15.values, color=model_colors[mname], alpha=0.8, edgecolor='white')
        ax.set_title(f'{mname}', fontweight='bold')
        ax.set_xlabel('Feature Importance')

    plt.suptitle(f'{target} -- Top 15 Feature Importances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'ml_{target.lower()}_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

# -- Plot 4: Cross-asset RMSE comparison bar chart --
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, metric, label in zip(axes, ['te_rmse', 'te_r2', 'te_dacc'],
                              ['Test RMSE (lower=better)', 'Test R² (higher=better)', 'Test Dir Accuracy (higher=better)']):
    x = np.arange(len(TARGETS))
    width = 0.18
    for i, mname in enumerate(MODELS):
        vals = [all_results[t][mname][metric] for t in TARGETS]
        ax.bar(x + i * width, vals, width, label=mname, color=model_colors[mname], alpha=0.85, edgecolor='white')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(TARGETS, fontweight='bold')
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=8)

plt.suptitle('Cross-Asset Model Comparison (Test: 2025-2026)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ml_cross_asset_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Plot 5: Residual distributions (test) --
for target in TARGETS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    y_test_t = test_data[target].values
    for ax, mname in zip(axes.flatten(), MODELS):
        pred = all_results[target][mname]['pred_test']
        residuals = y_test_t - pred
        ax.bar(test_data.index, residuals, width=20, color=model_colors[mname], alpha=0.7, edgecolor='white')
        ax.axhline(0, color='black', lw=1, ls='-')
        ax.set_title(f'{mname} (Mean Resid={residuals.mean():.4f})', fontweight='bold', fontsize=10)
        ax.set_ylabel('Residual')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f'{target} -- Test Residuals (2025-2026)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'ml_{target.lower()}_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()

# -- Plot 6: Cumulative returns -- actual vs model-predicted portfolio --
for target in TARGETS:
    fig, ax = plt.subplots(figsize=(14, 6))
    y_test_t = test_data[target].values
    cum_actual = np.cumsum(y_test_t)
    ax.plot(test_data.index, cum_actual, 'k-o', lw=2.5, ms=6, label='Actual', zorder=5)
    for mname in MODELS:
        pred = all_results[target][mname]['pred_test']
        # Strategy: go long if pred > 0, else stay flat (0 return)
        strategy_ret = np.where(pred > 0, y_test_t, 0)
        cum_strategy = np.cumsum(strategy_ret)
        ax.plot(test_data.index, cum_strategy, '-o', color=model_colors[mname],
                lw=1.5, ms=4, label=f'{mname}', alpha=0.85)

    ax.axhline(0, color='grey', lw=0.7, ls=':')
    ax.set_title(f'{target} -- Cumulative Returns: Actual vs Long-Only Signal (Test 2025-2026)',
                 fontweight='bold')
    ax.set_ylabel('Cumulative Log Return')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'ml_{target.lower()}_cumulative.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\n" + "=" * 85)
print("  PLOTS SAVED")
print("=" * 85)
for target in TARGETS:
    tl = target.lower()
    print(f"  ml_{tl}_predictions.png        - Test predictions (all 4 models)")
    print(f"  ml_{tl}_train_scatter.png      - Train actual vs predicted scatter")
    print(f"  ml_{tl}_feature_importance.png  - RF & GB feature importances")
    print(f"  ml_{tl}_residuals.png          - Test residuals")
    print(f"  ml_{tl}_cumulative.png         - Cumulative return strategies")
print(f"  ml_cross_asset_comparison.png   - Cross-asset RMSE/R2/DirAcc comparison")
print("\nDONE.")
