import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

# ════════════════════════════════════════════════════════════════════
# 1. LOAD & MERGE ALL DATA
# ════════════════════════════════════════════════════════════════════
# --- Asset data ---
asset_tr = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date']).set_index('Date').sort_index()
asset_tr.index = pd.to_datetime(asset_tr.index).tz_localize(None)
asset_te = pd.read_csv('data_daily_2025_2026.csv', parse_dates=['Date']).set_index('Date').sort_index()
asset_te.index = pd.to_datetime(asset_te.index).tz_localize(None)

# --- Macro data: merge India + US for richer feature set ---
def load_and_merge_macro(india_path, us_path):
    india = pd.read_csv(india_path, parse_dates=['Date']).set_index('Date').sort_index()
    india.index = pd.to_datetime(india.index).tz_localize(None)
    us = pd.read_csv(us_path, parse_dates=['Date']).set_index('Date').sort_index()
    us.index = pd.to_datetime(us.index).tz_localize(None)
    # India-only columns (not in US dataset)
    india_only = [c for c in india.columns if c not in us.columns]
    merged = us.join(india[india_only], how='outer').sort_index()
    return merged

macro_tr = load_and_merge_macro('macro_finaldata_india.csv', 'macro_finaldata_2005_2024.csv')
macro_te = load_and_merge_macro('macro_india_2025_2026.csv', 'macro_finaldata_2025_2026.csv')

TARGET = 'Nifty50_USD'

print("=" * 80)
print("  PCA + GMM CLUSTERING — Nifty50_USD (All Macro Features)")
print("  Train: 2005-2024 | Test: Jan 2025 - Feb 2026")
print("  Macro sources: India (CPI,PMI,Repo,M2,VIX) + US (24 variables)")
print("=" * 80)

# ════════════════════════════════════════════════════════════════════
# 2. PREPARE FEATURES & RETURNS
# ════════════════════════════════════════════════════════════════════
MACRO_COLS = list(macro_tr.columns)

def prepare_data(asset_df, macro_df):
    monthly_price = asset_df[[TARGET]].resample('ME').last().ffill()
    monthly_ret   = np.log(monthly_price / monthly_price.shift(1)).dropna()

    macro_m = macro_df[MACRO_COLS].resample('ME').last().ffill().bfill().fillna(0)

    # Engineered change features
    change_cols = {
        'CPI_Change':   'CPI',
        'M2_Change':    'M2',
        'Oil_Change':   'Oil_Brent_USD',
        'DXY_Change':   'DXY_Dollar_Index',
        'Forex_Change': 'IN_Forex_Reserves_USD',
        'Gold_Change':  'Gold_USD',
        'US_IP_Change': 'US_Industrial_Production',
    }
    for new_col, src_col in change_cols.items():
        if src_col in macro_m.columns:
            macro_m[new_col] = macro_m[src_col].pct_change()

    if 'India_VIX' in macro_m.columns:
        macro_m['VIX_Change'] = macro_m['India_VIX'].pct_change()

    feat_cols = list(macro_m.columns)
    features_lagged = macro_m[feat_cols].shift(1)
    merged = pd.concat([monthly_ret, features_lagged], axis=1).dropna()
    merged = merged.replace([np.inf, -np.inf], 0)
    return merged, feat_cols

train_data, FEATURE_COLS = prepare_data(asset_tr, macro_tr)
test_data, _             = prepare_data(asset_te, macro_te)

X_train = train_data[FEATURE_COLS].values
y_train = train_data[TARGET].values
X_test  = test_data[FEATURE_COLS].values
y_test  = test_data[TARGET].values

print(f"\n  Train: {len(train_data)} months ({train_data.index[0].date()} -> {train_data.index[-1].date()})")
print(f"  Test : {len(test_data)} months ({test_data.index[0].date()} -> {test_data.index[-1].date()})")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Feature list: {FEATURE_COLS}")

# Standardize
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

# ════════════════════════════════════════════════════════════════════
# 3. PCA — DIMENSIONALITY REDUCTION
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 80)

pca_full = PCA().fit(Xs_train)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)

print("\n  Component   Explained Var   Cumulative Var")
print("  " + "-" * 45)
for i in range(min(15, len(cum_var))):
    print("  PC{:<10d} {:>12.4f}     {:>12.4f}".format(
        i+1, pca_full.explained_variance_ratio_[i], cum_var[i]))

n_comp_90 = np.argmax(cum_var >= 0.90) + 1
n_comp_95 = np.argmax(cum_var >= 0.95) + 1
print(f"\n  Components for 90% variance: {n_comp_90}")
print(f"  Components for 95% variance: {n_comp_95}")

N_PCA = n_comp_90
pca = PCA(n_components=N_PCA, random_state=42)
Z_train = pca.fit_transform(Xs_train)
Z_test  = pca.transform(Xs_test)

print(f"\n  Using {N_PCA} PCs (explains {cum_var[N_PCA-1]*100:.1f}% variance)")
print(f"  Reduced: {Xs_train.shape[1]} features -> {N_PCA} PCs")

# PCA loadings
print("\n  TOP LOADINGS PER PRINCIPAL COMPONENT:")
loadings = pd.DataFrame(pca.components_.T, index=FEATURE_COLS,
                        columns=[f'PC{i+1}' for i in range(N_PCA)])
for pc in loadings.columns[:5]:
    top = loadings[pc].abs().nlargest(5)
    signs = ['+' if loadings[pc][f] > 0 else '-' for f in top.index]
    feats = [f"{s}{f}({v:.3f})" for f, v, s in zip(top.index, top.values, signs)]
    print(f"  {pc}: {', '.join(feats)}")

# ════════════════════════════════════════════════════════════════════
# 4. GMM CLUSTERING ON PCA-REDUCED DATA
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  GAUSSIAN MIXTURE MODEL (GMM) CLUSTERING")
print("=" * 80)

K_range = range(2, 8)
bic_scores = []
aic_scores = []
sil_scores = []

for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                           n_init=10, random_state=42)
    gmm.fit(Z_train)
    labels = gmm.predict(Z_train)
    bic_scores.append(gmm.bic(Z_train))
    aic_scores.append(gmm.aic(Z_train))
    if len(set(labels)) > 1:
        sil_scores.append(silhouette_score(Z_train, labels))
    else:
        sil_scores.append(0)

print("\n  K    BIC          AIC          Silhouette")
print("  " + "-" * 45)
for k, b, a, s in zip(K_range, bic_scores, aic_scores, sil_scores):
    print(f"  {k}    {b:>10.1f}   {a:>10.1f}   {s:>10.4f}")

optimal_k_bic = list(K_range)[np.argmin(bic_scores)]
optimal_k_sil = list(K_range)[np.argmax(sil_scores)]
print(f"\n  Optimal K (BIC):        {optimal_k_bic}")
print(f"  Optimal K (Silhouette): {optimal_k_sil}")

K = optimal_k_bic
gmm_final = GaussianMixture(n_components=K, covariance_type='full',
                              n_init=10, random_state=42)
gmm_final.fit(Z_train)

train_clusters = gmm_final.predict(Z_train)
test_clusters  = gmm_final.predict(Z_test)
train_probs    = gmm_final.predict_proba(Z_train)
test_probs     = gmm_final.predict_proba(Z_test)

print(f"\n  Using K={K} clusters")
print(f"  Train cluster distribution: {dict(zip(*np.unique(train_clusters, return_counts=True)))}")
print(f"  Test  cluster distribution: {dict(zip(*np.unique(test_clusters, return_counts=True)))}")

# ════════════════════════════════════════════════════════════════════
# 5. REGIME ANALYSIS — RETURN STATS PER CLUSTER
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  REGIME ANALYSIS — Nifty50_USD Return Statistics per Cluster")
print("=" * 80)

train_data_copy = train_data.copy()
train_data_copy['Cluster'] = train_clusters

print("\n  {:<10} {:>8} {:>12} {:>12} {:>12} {:>12} {:>10}".format(
    "Cluster", "Count", "Mean Ret", "Std Dev", "Min", "Max", "Sharpe"))
print("  " + "-" * 78)

cluster_stats = {}
for c in range(K):
    mask = train_data_copy['Cluster'] == c
    rets = train_data_copy.loc[mask, TARGET]
    stats = {
        'count'  : len(rets),
        'mean'   : rets.mean(),
        'std'    : rets.std(),
        'min'    : rets.min(),
        'max'    : rets.max(),
        'sharpe' : rets.mean() / rets.std() if rets.std() > 0 else 0,
    }
    cluster_stats[c] = stats
    print("  {:<10} {:>8d} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        f"C{c}", stats['count'], stats['mean'], stats['std'],
        stats['min'], stats['max'], stats['sharpe']))

# Label regimes
regime_labels = {}
for c in range(K):
    m = cluster_stats[c]['mean']
    s = cluster_stats[c]['std']
    if m > 0.02 and s < 0.06:
        regime_labels[c] = 'Bull-Low Vol'
    elif m > 0.02:
        regime_labels[c] = 'Bull-High Vol'
    elif m < -0.01 and s > 0.06:
        regime_labels[c] = 'Bear-High Vol'
    elif m < -0.01:
        regime_labels[c] = 'Bear-Low Vol'
    elif s > 0.08:
        regime_labels[c] = 'Crisis'
    else:
        regime_labels[c] = 'Sideways'

print("\n  Regime Labels:")
for c, label in regime_labels.items():
    print(f"  C{c}: {label} (mean={cluster_stats[c]['mean']:.4f}, std={cluster_stats[c]['std']:.4f})")

# ════════════════════════════════════════════════════════════════════
# 6. CLUSTER-CONDITIONAL PREDICTION
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  CLUSTER-CONDITIONAL RETURN PREDICTION")
print("=" * 80)

# Method 1: Predict return as cluster mean
pred_cluster_mean = np.array([cluster_stats[c]['mean'] for c in test_clusters])

# Method 2: Probability-weighted return
cluster_means = np.array([cluster_stats[c]['mean'] for c in range(K)])
pred_prob_weighted = test_probs @ cluster_means

# Method 3: Per-cluster linear regression on PCs
pred_cluster_lr = np.zeros(len(test_data))
for i in range(len(test_data)):
    c = test_clusters[i]
    mask = train_clusters == c
    if mask.sum() >= 5:
        lr = LinearRegression()
        lr.fit(Z_train[mask], y_train[mask])
        pred_cluster_lr[i] = lr.predict(Z_test[i:i+1])[0]
    else:
        pred_cluster_lr[i] = cluster_stats[c]['mean']

# Method 4: Global LR on PCs (baseline)
lr_global = LinearRegression().fit(Z_train, y_train)
pred_pca_lr = lr_global.predict(Z_test)

methods = {
    'Cluster Mean'       : pred_cluster_mean,
    'Prob-Weighted Mean' : pred_prob_weighted,
    'Cluster-LR on PCs'  : pred_cluster_lr,
    'Global-LR on PCs'   : pred_pca_lr,
}

print("\n  {:<22} {:>10} {:>10} {:>10} {:>10}".format(
    "Method", "RMSE", "MAE", "R2", "Dir Acc"))
print("  " + "-" * 62)
for mname, pred in methods.items():
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    dacc = np.mean(np.sign(pred) == np.sign(y_test))
    print("  {:<22} {:>10.6f} {:>10.6f} {:>10.4f} {:>9.1%}".format(
        mname, rmse, mae, r2, dacc))

# Month-by-month
print("\n  MONTH-BY-MONTH PREDICTIONS:")
print("  {:<10} {:>8} {:>8} {:>10} {:>10} {:>12} {:>10}".format(
    "Month", "Actual", "Cluster", "ClustMean", "ProbWt", "ClustLR", "GlobalLR"))
print("  " + "-" * 70)
for i, date in enumerate(test_data.index):
    print("  {:<10} {:>8.4f} {:>8} {:>10.4f} {:>10.4f} {:>12.4f} {:>10.4f}".format(
        date.strftime('%Y-%m'), y_test[i], f"C{test_clusters[i]}",
        pred_cluster_mean[i], pred_prob_weighted[i],
        pred_cluster_lr[i], pred_pca_lr[i]))

# ════════════════════════════════════════════════════════════════════
# 7. PORTFOLIO ALLOCATION PER REGIME
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  REGIME-BASED PORTFOLIO ALLOCATION SUGGESTION")
print("=" * 80)

ASSETS = ['Nifty50_USD', 'SP500', 'Gold', 'USBond']

# Compute asset return stats per cluster (train)
asset_monthly_tr = asset_tr[ASSETS].resample('ME').last().ffill()
asset_monthly_ret_tr = np.log(asset_monthly_tr / asset_monthly_tr.shift(1)).dropna()
asset_monthly_ret_tr = asset_monthly_ret_tr.loc[train_data.index]
asset_monthly_ret_tr['Cluster'] = train_clusters

print("\n  ASSET RETURNS BY REGIME (Train):")
for c in range(K):
    mask = asset_monthly_ret_tr['Cluster'] == c
    sub = asset_monthly_ret_tr.loc[mask, ASSETS]
    print(f"\n  C{c}: {regime_labels[c]} ({mask.sum()} months)")
    print("  {:<15} {:>10} {:>10} {:>10}".format("Asset", "Mean", "Std", "Sharpe"))
    print("  " + "-" * 45)
    for a in ASSETS:
        mn = sub[a].mean()
        sd = sub[a].std()
        sh = mn / sd if sd > 0 else 0
        print("  {:<15} {:>10.4f} {:>10.4f} {:>10.4f}".format(a, mn, sd, sh))

# ════════════════════════════════════════════════════════════════════
# 8. PLOTS
# ════════════════════════════════════════════════════════════════════

# Plot 1: Explained Variance (Scree Plot)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
            pca_full.explained_variance_ratio_, color='steelblue', alpha=0.8, edgecolor='white')
axes[0].set_title('Individual Explained Variance', fontweight='bold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')

axes[1].plot(range(1, len(cum_var)+1), cum_var, 'bo-', lw=2, ms=6)
axes[1].axhline(0.90, color='red', ls='--', lw=1.5, label='90% threshold')
axes[1].axhline(0.95, color='orange', ls='--', lw=1.5, label='95% threshold')
axes[1].axvline(N_PCA, color='green', ls=':', lw=2, label=f'Selected: {N_PCA} PCs')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance')
axes[1].legend()

plt.suptitle('PCA Scree Plot -- Combined Macro Features (2005-2024)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_gmm_scree_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: BIC/AIC/Silhouette for GMM K selection
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(list(K_range), bic_scores, 'bo-', lw=2, ms=8)
axes[0].axvline(optimal_k_bic, color='red', ls='--', lw=2, label=f'Optimal K={optimal_k_bic}')
axes[0].set_title('BIC Score (lower = better)', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('BIC')
axes[0].legend()

axes[1].plot(list(K_range), aic_scores, 'go-', lw=2, ms=8)
axes[1].set_title('AIC Score (lower = better)', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('AIC')

axes[2].plot(list(K_range), sil_scores, 'ro-', lw=2, ms=8)
axes[2].axvline(optimal_k_sil, color='blue', ls='--', lw=2, label=f'Optimal K={optimal_k_sil}')
axes[2].set_title('Silhouette Score (higher = better)', fontweight='bold')
axes[2].set_xlabel('Number of Clusters (K)')
axes[2].set_ylabel('Silhouette')
axes[2].legend()

plt.suptitle('GMM Cluster Selection Criteria', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_gmm_cluster_selection.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: PCA scatter (PC1 vs PC2) colored by cluster
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter_tr = axes[0].scatter(Z_train[:, 0], Z_train[:, 1], c=train_clusters,
                              cmap='Set1', s=40, alpha=0.7, edgecolors='white')
axes[0].set_title('Train (2005-2024)', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter_tr, ax=axes[0], label='Cluster')

scatter_te = axes[1].scatter(Z_test[:, 0], Z_test[:, 1], c=test_clusters,
                              cmap='Set1', s=80, alpha=0.9, edgecolors='black', linewidths=1.5)
for i, date in enumerate(test_data.index):
    axes[1].annotate(date.strftime('%b%y'), (Z_test[i, 0], Z_test[i, 1]),
                     fontsize=7, ha='center', va='bottom')
axes[1].set_title('Test (2025-2026)', fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter_te, ax=axes[1], label='Cluster')

plt.suptitle('PCA + GMM Clusters -- Nifty50_USD Macro Regimes', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_gmm_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Regime timeline (train)
fig, ax = plt.subplots(figsize=(16, 5))
colors_map = plt.cm.Set1(np.linspace(0, 1, K))
for i, (date, ret) in enumerate(zip(train_data.index, y_train)):
    ax.bar(date, ret, width=25, color=colors_map[train_clusters[i]], alpha=0.7)
ax.axhline(0, color='black', lw=0.7, ls=':')

from matplotlib.patches import Patch
handles = [Patch(facecolor=colors_map[c], label=f'C{c}: {regime_labels[c]}')
           for c in range(K)]
ax.legend(handles=handles, fontsize=9, loc='upper left')
ax.set_title('Nifty50_USD Monthly Returns Colored by GMM Regime (2005-2024)',
             fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Log Return')
plt.tight_layout()
plt.savefig('pca_gmm_regime_timeline.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Test predictions — all methods
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (mname, pred) in zip(axes.flatten(), methods.items()):
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    dacc = np.mean(np.sign(pred) == np.sign(y_test))
    ax.bar(test_data.index, y_test, width=20, color='#555555', alpha=0.4,
           label='Actual', zorder=2)
    ax.plot(test_data.index, pred, 'r-o', lw=2, ms=5, label='Predicted', zorder=3)
    ax.axhline(0, color='black', lw=0.7, ls=':')
    ax.set_title("{} (RMSE={:.4f}, R2={:.4f}, Dir={:.0%})".format(
        mname, rmse, r2, dacc), fontweight='bold', fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Log Return')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('PCA+GMM Predictions vs Actual -- Nifty50_USD (Test: 2025-2026)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_gmm_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: Cluster return distributions (box plot)
fig, ax = plt.subplots(figsize=(10, 6))
data_for_box = [train_data_copy.loc[train_data_copy['Cluster'] == c, TARGET].values
                for c in range(K)]
bp = ax.boxplot(data_for_box, patch_artist=True, labels=[f'C{c}\n{regime_labels[c]}'
                for c in range(K)])
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors_map[i])
    patch.set_alpha(0.7)
ax.axhline(0, color='red', lw=1, ls='--')
ax.set_title('Return Distribution per GMM Cluster (Train: 2005-2024)', fontweight='bold')
ax.set_ylabel('Monthly Log Return')
plt.tight_layout()
plt.savefig('pca_gmm_cluster_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 7: Cluster probabilities for test months (heatmap)
fig, ax = plt.subplots(figsize=(12, 5))
prob_df = pd.DataFrame(test_probs,
                       index=[d.strftime('%Y-%m') for d in test_data.index],
                       columns=[f'C{c}: {regime_labels[c]}' for c in range(K)])
sns.heatmap(prob_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='white', vmin=0, vmax=1)
ax.set_title('GMM Cluster Probabilities -- Test Months (2025-2026)', fontweight='bold')
ax.set_xlabel('Cluster (Regime)')
ax.set_ylabel('Month')
plt.tight_layout()
plt.savefig('pca_gmm_test_probabilities.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 8: PCA Loadings heatmap (top 5 PCs)
fig, ax = plt.subplots(figsize=(8, 12))
n_show = min(5, N_PCA)
sns.heatmap(loadings.iloc[:, :n_show], annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, linecolor='white')
ax.set_title(f'PCA Loadings (Top {n_show} PCs)', fontweight='bold')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig('pca_gmm_loadings.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 9: Multi-asset regime returns heatmap
fig, ax = plt.subplots(figsize=(10, 5))
regime_asset_means = pd.DataFrame(index=[f'C{c}: {regime_labels[c]}' for c in range(K)],
                                   columns=ASSETS)
for c in range(K):
    mask = asset_monthly_ret_tr['Cluster'] == c
    for a in ASSETS:
        regime_asset_means.loc[f'C{c}: {regime_labels[c]}', a] = \
            asset_monthly_ret_tr.loc[mask, a].mean()

regime_asset_means = regime_asset_means.astype(float)
sns.heatmap(regime_asset_means, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
            ax=ax, linewidths=1, linecolor='white')
ax.set_title('Mean Monthly Return per Regime & Asset (Train: 2005-2024)', fontweight='bold')
plt.tight_layout()
plt.savefig('pca_gmm_regime_asset_returns.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[9 Plots Saved]")
print("  pca_gmm_scree_plot.png          - PCA explained variance")
print("  pca_gmm_cluster_selection.png   - BIC/AIC/Silhouette for K")
print("  pca_gmm_scatter.png             - PC1 vs PC2 colored by cluster")
print("  pca_gmm_regime_timeline.png     - Regime timeline (train)")
print("  pca_gmm_predictions.png         - Test predictions (all methods)")
print("  pca_gmm_cluster_boxplot.png     - Return distribution per cluster")
print("  pca_gmm_test_probabilities.png  - Cluster probabilities (test)")
print("  pca_gmm_loadings.png            - PCA loadings heatmap")
print("  pca_gmm_regime_asset_returns.png - Multi-asset returns by regime")
print("\nDONE.")
