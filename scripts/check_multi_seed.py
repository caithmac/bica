import pandas as pd, numpy as np, re
d = pd.read_csv('diary/results_diary.csv')
r = 'test_rmse'

def base(eid):
    return re.sub(r'(__seed\d+|_seed\d+|__leaky.*)$', '', str(eid))

deep = d[d['model_family'].isin(['mlp','transformer','bica_v2','bica','deepdta'])].copy()
deep['base'] = deep['experiment_id'].apply(base)
counts = deep.groupby('base').agg(
    n=('experiment_id', 'nunique'),
    best=(r, 'min'),
    mean=(r, 'mean'),
    std=(r, 'std')
)
multi = counts[counts['n'] >= 3].sort_values('best')
print('Deep models with 3+ seeds:')
for b, row in multi.iterrows():
    n = int(row['n'])
    best = float(row['best'])
    mean = float(row['mean'])
    std = float(row['std'])
    print(f'  {b:<45s} n={n} best={best:.4f} mean={mean:.4f} std={std:.4f}')

# Fisher ablation
print('\nFisher ablation target counts per seed:')
from harness.data import get_splits_for_seed
for s in [42, 123, 456]:
    tr, _, _ = get_splits_for_seed(s)
    n = tr['Target'].nunique()
    ok = 'OK' if n <= 1024 else 'COLLISION!'
    print(f'  seed={s}: {n} unique targets ({ok})')
