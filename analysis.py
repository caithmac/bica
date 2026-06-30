import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('diary/results_diary.csv')
main = df[df['split_type'] == 'scaffold_bemis_murcko_seed42'].copy()
main = main[main['test_rmse'] < 5].copy()

print('=== DATASET ===')
print(f'Total: {len(df)}, Clean seed42: {len(main)}')

# RF feature importance
feature_cols = ['model_family', 'ligand_repr', 'protein_repr', 'fusion_strategy']
X_encoded = pd.DataFrame()
for col in feature_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(main[col].astype(str))
y = main['test_rmse'].values
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_encoded, y)
importances = rf.feature_importances_
for name, imp in zip(feature_cols, importances):
    print(f'{name}: {imp:.4f} ({imp/sum(importances)*100:.1f}%)')

# ANOVAs
for col in feature_cols:
    groups = [g['test_rmse'].values for _, g in main.groupby(col)]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f'ANOVA {col}: F={f_stat:.2f}, p={p_val:.2e}')

print()
print('Model family performance:')
for mf in sorted(main['model_family'].unique()):
    sub = main[main['model_family']==mf]
    b = sub.loc[sub['test_rmse'].idxmin()]
    print(f'{mf:15s}: best={b["test_rmse"]:.4f} avg={sub["test_rmse"].mean():.4f} n={len(sub)}')

best_tree = main[main['model_family']=='tree'].sort_values('test_rmse').iloc[0]
print(f'\nBest overall: {best_tree["experiment_id"]} RMSE={best_tree["test_rmse"]:.4f} R={best_tree["test_pearson_r"]:.4f}')

for mf in ['gcn','gat']:
    bm = main[main['model_family']==mf].sort_values('test_rmse').iloc[0]
    print(f'Best {mf.upper()}: RMSE={bm["test_rmse"]:.4f} ({(bm["test_rmse"]/best_tree["test_rmse"])-1:.1%} worse)')

print()
print('BiCA vs MLP (matched representations):')
for (lig, prot), grp in main[main['model_family'].isin(['bica','mlp'])].groupby(['ligand_repr','protein_repr']):
    bic = grp[grp['model_family']=='bica']['test_rmse'].mean()
    mlp_v = grp[grp['model_family']=='mlp']['test_rmse'].mean()
    if len(grp[grp['model_family']=='bica']) > 0 and len(grp[grp['model_family']=='mlp']) > 0:
        print(f'{lig:20s}+{prot:20s}: BiCA={bic:.4f} MLP={mlp_v:.4f} diff={(bic/mlp_v-1)*100:+.1f}%')

print()
print('Protein representation impact across model families:')
for mf in ['tree','mlp','gcn','gat','bica','transformer']:
    aac = main[(main['model_family']==mf)&(main['protein_repr']=='aac_20')]['test_rmse'].mean()
    esm = main[(main['model_family']==mf)&(main['protein_repr'].str.contains('esm2|esmc'))]['test_rmse'].mean()
    if not np.isnan(aac) and not np.isnan(esm):
        print(f'{mf:12s}: AAC={aac:.4f} -> ESM={esm:.4f} ({(aac/esm-1)*100:+.1f}%)')

print()
print('ESM-2 size scaling (all models):')
for sz in ['esm2_8M_320','esm2_35M_480','esm2_150M','esm2_650M']:
    sub = main[main['protein_repr']==sz]
    if len(sub) > 0:
        print(f'{sz:15s}: avg={sub["test_rmse"].mean():.4f} best={sub["test_rmse"].min():.4f}')

print()
print('=== TOP 10 EXPERIMENTS BY RMSE ===')
top10 = main.sort_values('test_rmse').head(10)
for _, row in top10.iterrows():
    print(f'{row["experiment_id"]:45s} | {row["model_family"]:15s} | {row["ligand_repr"]:20s} | {row["protein_repr"]:20s} | RMSE={row["test_rmse"]:.4f} | R={row["test_pearson_r"]:.4f}')

print()
print('=== BOTTOM 5 EXPERIMENTS (worst non-failed) ===')
bot5 = main.sort_values('test_rmse', ascending=False).head(5)
for _, row in bot5.iterrows():
    print(f'{row["experiment_id"]:45s} | {row["model_family"]:15s} | RMSE={row["test_rmse"]:.4f}')
