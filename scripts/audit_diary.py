import pandas as pd, numpy as np, re
from scipy import stats

diary = pd.read_csv('diary/results_diary.csv')
rmse_col = 'test_rmse' if 'test_rmse' in diary.columns else 'rmse'
pearson_col = 'test_pearson_r' if 'test_pearson_r' in diary.columns else 'pearson_r'

# 1. RF cross-seed
rf_mask = diary['experiment_id'].str.startswith('rf_ecfp4_aac')
rf_mask = rf_mask & ~diary['experiment_id'].str.contains('matched|zeroshot|leaky|__seed|__leaky', na=False)
rf = diary[rf_mask].sort_values(rmse_col)
print('=== RF (ECFP4+AAC) across seeds ===')
for _, r in rf.iterrows():
    print(f"  {r['experiment_id']:<40s} RMSE={r[rmse_col]:.4f}  r={r.get(pearson_col,'N/A')}")

# 2. Pathological outliers
bad = diary[diary[rmse_col] > 2.0][['experiment_id','model_family',rmse_col,pearson_col]].sort_values(rmse_col, ascending=False)
print(f'\n=== Pathological (RMSE > 2.0): {len(bad)} ===')
for _, r in bad.iterrows():
    print(f"  {r['experiment_id']:<50s} family={str(r.get('model_family','?')):<15s} RMSE={r[rmse_col]:.4f}")

# 3. Multi-seed deep models
def base_id(eid):
    return re.sub(r'(__seed\d+|_seed\d+|__leaky.*)$', '', str(eid))

deep = diary[diary['model_family'].isin(['mlp','transformer','gcn','gat','bica_v2','bica'])].copy()
deep['base'] = deep['experiment_id'].apply(base_id)
multi = deep.groupby('base').filter(lambda g: g['experiment_id'].nunique() >= 3)
print(f'\n=== Deep models with 3+ seeds ===')
for base, grp in multi.groupby('base'):
    vals = grp[rmse_col].values
    print(f"  {base:<45s} n={len(vals)}  mean={np.mean(vals):.4f}  std={np.std(vals,ddof=1):.4f}  range=[{min(vals):.4f},{max(vals):.4f}]")

# 4. RF seed-by-seed
print('\n=== RF by seed ===')
rf_all = diary[diary['experiment_id'].str.contains('rf_ecfp4_aac')]
rf_all = rf_all[~rf_all['experiment_id'].str.contains('matched|zeroshot|leaky', na=False)]
for seed in ['42','123','456','99']:
    subset = rf_all[rf_all['experiment_id'].str.contains(f'seed{seed}') | ((rf_all['experiment_id'] == 'rf_ecfp4_aac') & (seed == '42'))]
    if len(subset) > 0:
        vals = subset[rmse_col].values
        print(f"  Seed {seed}: n={len(vals)}  min={min(vals):.4f}  max={max(vals):.4f}  vals={[f'{v:.4f}' for v in vals]}")

# 5. Winner's curse check - how many runs per config?
print('\n=== Runs per config (top RMSE models) ===')
config_counts = diary.groupby('experiment_id').size().sort_values(ascending=False)
for eid, count in config_counts.head(10).items():
    subset = diary[diary['experiment_id'] == eid]
    vals = subset[rmse_col].round(4).tolist()
    family = subset['model_family'].iloc[0] if 'model_family' in subset.columns else '?'
    print(f"  {eid:<50s} {count}x  family={family:<15s} RMSEs={vals}")
