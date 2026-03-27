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
from matplotlib.patches import Patch

# ====================================================================
# 1. LOAD DATA
# ====================================================================
asset_tr = pd.read_csv('data_daily_2005_2024.csv', parse_dates=['Date']).set_index('Date').sort_index()
asset_tr.index = pd.to_datetime(asset_tr.index).tz_localize(None)
asset_te = pd.read_csv('data_daily_2025_2026.csv', parse_dates=['Date']).set_index('Date').sort_index()
asset_te.index = pd.to_datetime(asset_te.index).tz_localize(None)

macro_tr = pd.read_csv('macro_finaldata_2005_2024.csv', parse_dates=['Date']).set_index('Date').sort_index()
macro_tr.index = pd.to_datetime(macro_tr.index).tz_localize(None)
macro_te = pd.read_csv('macro_finaldata_2025_2026.csv', parse_dates=['Date']).set_index('Date').sort_index()
macro_te.index = pd.to_datetime(macro_te.index).tz_localize(None)

MACRO_COLS = list(macro_tr.columns)
TARGET = 'Gold'
EXCLUDED_FEATURES = ['Gold_USD', 'Gold_Change']

print("=" * 85)
print("  PCA + GMM CLUSTERING -- GOLD (excluding Gold_USD & Gold_Change)")
print("  Train: 2005-2024 | Test: 2025 onwards")
print("  Features: 28 (30 - 2 gold-related)")
print("=" * 85)

# ====================================================================
# 2. PREPARE FEATURES & RETURNS
# ====================================================================
def prepare_data(asset_df, macro_df):
    monthly_price = asset_df[[TARGET]].resample('ME').last().ffill()
    monthly_ret   = np.log(monthly_price / monthly_price.shift(1)).dropna()

    macro_m = macro_df[MACRO_COLS].resample('ME').last().ffill().bfill().fillna(0)

    change_pairs = {
        'Oil_Change':   'Oil_Brent_USD',
        'Gold_Change':  'Gold_USD',
        'DXY_Change':   'DXY_Dollar_Index',
        'Forex_Change': 'IN_Forex_Reserves_USD',
        'US_IP_Change': 'US_Industrial_Production',
        'VIX_Change':   'US_VIX',
    }
    for new_col, src_col in change_pairs.items():
        if src_col in macro_m.columns:
            macro_m[new_col] = macro_m[src_col].pct_change()

    # EXCLUDE Gold_USD and Gold_Change
    feat_cols = [c for c in macro_m.columns if c not in EXCLUDED_FEATURES]

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
print(f"  Excluded: {EXCLUDED_FEATURES}")
print(f"  Feature list: {FEATURE_COLS}")

# Standardize
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

# ====================================================================
# 3. PCA -- DIMENSIONALITY REDUCTION
# ====================================================================
print("\n" + "=" * 85)
print("  PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 85)

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

# ====================================================================
# 4. GMM CLUSTERING
# ====================================================================
print("\n" + "=" * 85)
print("  GAUSSIAN MIXTURE MODEL (GMM) CLUSTERING")
print("=" * 85)

K_range = range(2, 8)
bic_scores, aic_scores, sil_scores = [], [], []

for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                           n_init=10, random_state=42)
    gmm.fit(Z_train)
    labels = gmm.predict(Z_train)
    bic_scores.append(gmm.bic(Z_train))
    aic_scores.append(gmm.aic(Z_train))
    sil_scores.append(silhouette_score(Z_train, labels) if len(set(labels)) > 1 else 0)

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

# ====================================================================
# 5. REGIME ANALYSIS -- GOLD RETURN STATS PER CLUSTER
# ====================================================================
print("\n" + "=" * 85)
print("  REGIME ANALYSIS -- Gold Return Statistics per Cluster")
print("=" * 85)

train_data_c = train_data.copy()
train_data_c['Cluster'] = train_clusters

cluster_stats = {}
regime_labels = {}

print("\n  {:<12} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
    "Cluster", "Count", "Mean", "Std", "Min", "Max", "Sharpe"))
print("  " + "-" * 70)

for c in range(K):
    mask = train_data_c['Cluster'] == c
    rets = train_data_c.loc[mask, TARGET]
    stats = {
        'count': len(rets),
        'mean':  rets.mean(),
        'std':   rets.std(),
        'min':   rets.min(),
        'max':   rets.max(),
        'sharpe': rets.mean() / rets.std() if rets.std() > 0 else 0,
    }
    cluster_stats[c] = stats
    print("  {:<12} {:>6d} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.4f}".format(
        f"C{c}", stats['count'], stats['mean'], stats['std'],
        stats['min'], stats['max'], stats['sharpe']))

# Label regimes
for c in range(K):
    m = cluster_stats[c]['mean']
    s = cluster_stats[c]['std']
    if m > 0.02 and s < 0.04:
        regime_labels[c] = 'Bull-Low Vol'
    elif m > 0.02:
        regime_labels[c] = 'Bull-High Vol'
    elif m < -0.01 and s > 0.06:
        regime_labels[c] = 'Bear-High Vol'
    elif m < -0.01:
        regime_labels[c] = 'Bear-Low Vol'
    elif s > 0.06:
        regime_labels[c] = 'Crisis'
    else:
        regime_labels[c] = 'Sideways'

print("\n  Regime Labels:")
for c, label in regime_labels.items():
    print(f"  C{c}: {label} (mean={cluster_stats[c]['mean']:.4f}, std={cluster_stats[c]['std']:.4f})")

# ====================================================================
# 6. CLUSTER-CONDITIONAL PREDICTION
# ====================================================================
print("\n" + "=" * 85)
print("  CLUSTER-CONDITIONAL RETURN PREDICTION -- Gold")
print("=" * 85)

# Method 1: Cluster mean
pred_cluster_mean = np.array([cluster_stats[c]['mean'] for c in test_clusters])

# Method 2: Probability-weighted mean
cl_means = np.array([cluster_stats[c]['mean'] for c in range(K)])
pred_prob_weighted = test_probs @ cl_means

# Method 3: Per-cluster LR on PCs
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

# Method 4: Global LR on PCs
lr_global = LinearRegression().fit(Z_train, y_train)
pred_pca_lr = lr_global.predict(Z_test)

methods = {
    'Cluster Mean'      : pred_cluster_mean,
    'Prob-Weighted Mean': pred_prob_weighted,
    'Cluster-LR on PCs' : pred_cluster_lr,
    'Global-LR on PCs'  : pred_pca_lr,
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
print("\n  MONTH-BY-MONTH PREDICTIONS (Gold):")
print("  {:<10} {:>8} {:>6} {:>10} {:>10} {:>10} {:>10}".format(
    "Month", "Actual", "Clust", "ClMean", "ProbWt", "ClLR", "GlobLR"))
print("  " + "-" * 66)
for i, date in enumerate(test_data.index):
    print("  {:<10} {:>8.4f} {:>6} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        date.strftime('%Y-%m'), y_test[i], f"C{test_clusters[i]}",
        pred_cluster_mean[i], pred_prob_weighted[i],
        pred_cluster_lr[i], pred_pca_lr[i]))

# ====================================================================
# 7. PLOTS
# ====================================================================
colors_map = plt.cm.Set1(np.linspace(0, 1, K))
PREFIX = 'pca_gmm_gold_nogoldfeat'

# -- Plot 1: Scree plot --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
            pca_full.explained_variance_ratio_, color='goldenrod', alpha=0.8, edgecolor='white')
axes[0].set_title('Individual Explained Variance', fontweight='bold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')

axes[1].plot(range(1, len(cum_var)+1), cum_var, 'o-', color='goldenrod', lw=2, ms=6)
axes[1].axhline(0.90, color='red', ls='--', lw=1.5, label='90% threshold')
axes[1].axhline(0.95, color='orange', ls='--', lw=1.5, label='95% threshold')
axes[1].axvline(N_PCA, color='green', ls=':', lw=2, label=f'Selected: {N_PCA} PCs')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance')
axes[1].legend()

plt.suptitle('PCA Scree -- Gold (excl. Gold_USD & Gold_Change)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PREFIX}_scree.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Plot 2: BIC/AIC/Silhouette --
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(list(K_range), bic_scores, 'bo-', lw=2, ms=8)
axes[0].axvline(optimal_k_bic, color='red', ls='--', lw=2, label=f'Optimal K={optimal_k_bic}')
axes[0].set_title('BIC (lower=better)', fontweight='bold')
axes[0].set_xlabel('K'); axes[0].set_ylabel('BIC'); axes[0].legend()

axes[1].plot(list(K_range), aic_scores, 'go-', lw=2, ms=8)
axes[1].set_title('AIC (lower=better)', fontweight='bold')
axes[1].set_xlabel('K'); axes[1].set_ylabel('AIC')

axes[2].plot(list(K_range), sil_scores, 'ro-', lw=2, ms=8)
axes[2].axvline(optimal_k_sil, color='blue', ls='--', lw=2, label=f'Optimal K={optimal_k_sil}')
axes[2].set_title('Silhouette (higher=better)', fontweight='bold')
axes[2].set_xlabel('K'); axes[2].set_ylabel('Silhouette'); axes[2].legend()

plt.suptitle('GMM Cluster Selection -- Gold (excl. Gold features)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PREFIX}_cluster_selection.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Plot 3: PCA scatter (train + test) --
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc_tr = axes[0].scatter(Z_train[:, 0], Z_train[:, 1], c=train_clusters,
                         cmap='Set1', s=40, alpha=0.7, edgecolors='white')
axes[0].set_title('Train (2005-2024)', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(sc_tr, ax=axes[0], label='Cluster')

sc_te = axes[1].scatter(Z_test[:, 0], Z_test[:, 1], c=test_clusters,
                         cmap='Set1', s=80, alpha=0.9, edgecolors='black', linewidths=1.5)
for i, date in enumerate(test_data.index):
    axes[1].annotate(date.strftime('%b%y'), (Z_test[i, 0], Z_test[i, 1]),
                     fontsize=7, ha='center', va='bottom')
axes[1].set_title('Test (2025-2026)', fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(sc_te, ax=axes[1], label='Cluster')

plt.suptitle('PCA + GMM Clusters -- Gold Macro Regimes (no Gold features)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PREFIX}_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Plot 4: Regime timeline --
fig, ax = plt.subplots(figsize=(16, 5))
for i, (date, ret) in enumerate(zip(train_data.index, y_train)):
    ax.bar(date, ret, width=25, color=colors_map[train_clusters[i]], alpha=0.7)
ax.axhline(0, color='black', lw=0.7, ls=':')

handles = [Patch(facecolor=colors_map[c], label=f'C{c}: {regime_labels[c]}') for c in range(K)]
ax.legend(handles=handles, fontsize=9, loc='upper left')
ax.set_title('Gold Monthly Returns by GMM Regime (Train 2005-2024, no Gold features)',
             fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Log Return')
plt.tight_layout()
plt.savefig(f'{PREFIX}_regime_timeline.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Plot 5: Predictions (test) --
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (mname, pred) in zip(axes.flatten(), methods.items()):
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    dacc = np.mean(np.sign(pred) == np.sign(y_test))
    ax.bar(test_data.index, y_test, width=20, color='#555', alpha=0.4, label='Actual')
    ax.plot(test_data.index, pred, '-o', color='goldenrod', lw=2, ms=6, label='Predicted')
    ax.axhline(0, color='black', lw=0.7, ls=':')
    ax.set_title(f"{mname} (RMSE={rmse:.4f}, R2={r2:.4f}, Dir={dacc:.0%})",
                 fontweight='bold', fontsize=10)
    ax.set_ylabel('Log Return')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Gold -- PCA+GMM Predictions (Test 2025-2026, no Gold features)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PREFIX}_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Plot 6: Cluster boxplot --
fig, ax = plt.subplots(figsize=(12, 6))
data_box = [train_data_c.loc[train_data_c['Cluster'] == c, TARGET].values for c in range(K)]
bp = ax.boxplot(data_box, patch_artist=True,
                labels=[f'C{c}\n{regime_labels[c]}' for c in range(K)])
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors_map[i])
    patch.set_alpha(0.7)
ax.axhline(0, color='red', lw=1, ls='--')
ax.set_title('Gold Return Distribution per Cluster (Train 2005-2024, no Gold features)',
             fontweight='bold')
ax.set_ylabel('Monthly Log Return')
plt.tight_layout()
plt.savefig(f'{PREFIX}_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Plot 7: Cluster probabilities heatmap (test) --
fig, ax = plt.subplots(figsize=(12, 5))
prob_df = pd.DataFrame(test_probs,
                       index=[d.strftime('%Y-%m') for d in test_data.index],
                       columns=[f'C{c}: {regime_labels[c]}' for c in range(K)])
sns.heatmap(prob_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='white', vmin=0, vmax=1)
ax.set_title('GMM Cluster Probabilities -- Gold Test (2025-2026, no Gold features)',
             fontweight='bold')
ax.set_xlabel('Cluster (Regime)')
ax.set_ylabel('Month')
plt.tight_layout()
plt.savefig(f'{PREFIX}_test_probabilities.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Plot 8: PCA loadings heatmap --
fig, ax = plt.subplots(figsize=(8, 11))
n_show = min(5, N_PCA)
sns.heatmap(loadings.iloc[:, :n_show], annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, linecolor='white')
ax.set_title(f'PCA Loadings (Top {n_show} PCs, no Gold features)', fontweight='bold')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig(f'{PREFIX}_loadings.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Plot 9: Test timeline --
fig, ax = plt.subplots(figsize=(14, 5))
for i, (date, ret) in enumerate(zip(test_data.index, y_test)):
    ax.bar(date, ret, width=20, color=colors_map[test_clusters[i]], alpha=0.8,
           edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', lw=0.7, ls=':')
handles = [Patch(facecolor=colors_map[c], label=f'C{c}: {regime_labels[c]}') for c in range(K)]
ax.legend(handles=handles, fontsize=9, loc='upper left')
ax.set_title('Gold Test Returns by Regime (2025-2026, no Gold features)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Log Return')
plt.tight_layout()
plt.savefig(f'{PREFIX}_test_timeline.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 85)
print("  9 PLOTS SAVED (prefix: pca_gmm_gold_nogoldfeat_)")
print("=" * 85)
print(f"  {PREFIX}_scree.png               - PCA explained variance")
print(f"  {PREFIX}_cluster_selection.png    - BIC/AIC/Silhouette")
print(f"  {PREFIX}_scatter.png              - PC1 vs PC2 clusters")
print(f"  {PREFIX}_regime_timeline.png      - Train regime timeline")
print(f"  {PREFIX}_predictions.png          - Test predictions (4 methods)")
print(f"  {PREFIX}_boxplot.png              - Return distribution per cluster")
print(f"  {PREFIX}_test_probabilities.png   - Test cluster probabilities")
print(f"  {PREFIX}_loadings.png             - PCA loadings heatmap")
print(f"  {PREFIX}_test_timeline.png        - Test regime timeline")
print("\nDONE.")
