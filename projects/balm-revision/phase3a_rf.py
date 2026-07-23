#!/usr/bin/env python3
"""Phase 3a: RF models only — fast CPU benchmark."""
import csv, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, "E:/Drug Discovery")
from harness.featurizers import ecfp, amino_acid_composition, esm2_embeddings, concat

SEEDS = [42, 123, 456]
SPLIT_TYPES = ['random', 'scaffold', 'cold_target', 'seq_clustered/30pct', 'seq_clustered/40pct']
OUT_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/splits")
RESULTS_CSV = "E:/Drug Discovery/projects/balm-revision/phase3_results.csv"

print("Phase 3a: RF models", flush=True)

MODELS = [
    ('rf_ecfp4_aac', 'RF_ECFP4+AAC', 'tree', lambda df: concat(ecfp(df['Drug_canonical'].tolist()), amino_acid_composition(df['Target'].tolist()))),
    ('rf_ecfp4_esm2_8m', 'RF_ECFP4+ESM2-8M', 'tree', lambda df: concat(ecfp(df['Drug_canonical'].tolist()), esm2_embeddings(df['Target'].tolist(), model_size="8M"))),
    ('rf_ecfp4_only', 'RF_ECFP4-only', 'tree', lambda df: ecfp(df['Drug_canonical'].tolist())),
]

FIELD_NAMES = ['timestamp','model','model_family','split_type','seed',
               'n_train','n_val','n_test',
               'val_rmse','val_mae','val_pearson',
               'test_rmse','test_mae','test_pearson','test_spearman',
               'n_params','train_time_s','epochs_trained','device']

# Resume
existing = set()
if os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0:
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            existing.add((row['model'], row['split_type'], row['seed']))
    print(f"Resuming: {len(existing)} done", flush=True)

total = len(MODELS) * len(SPLIT_TYPES) * len(SEEDS)
done_count = 0

with open(RESULTS_CSV, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
    if not existing:
        writer.writeheader()
        f.flush()
    
    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            split_dir = OUT_DIR / split_type / f"seed_{seed}"
            train_df = pd.read_csv(split_dir / "train.csv")
            val_df = pd.read_csv(split_dir / "val.csv")
            test_df = pd.read_csv(split_dir / "test.csv")
            n_tr, n_v, n_te = len(train_df), len(val_df), len(test_df)
            
            for key, name, family, featurize_fn in MODELS:
                done_count += 1
                
                if (name, split_type, str(seed)) in existing:
                    continue
                
                print(f"[{done_count}/{total}] {name} | {split_type} | seed={seed}", flush=True)
                t0 = time.time()
                
                try:
                    X_tr = featurize_fn(train_df)
                    X_v = featurize_fn(val_df)
                    X_te = featurize_fn(test_df)
                    y_tr = train_df['Y'].values
                    y_v = val_df['Y'].values
                    y_te = test_df['Y'].values
                    
                    rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=seed, n_jobs=-1)
                    rf.fit(X_tr, y_tr)
                    
                    yp_v = rf.predict(X_v)
                    yp_te = rf.predict(X_te)
                    
                    n_params = sum(t.tree_.node_count for t in rf.estimators_)
                    
                    row = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'model': name, 'model_family': family,
                        'split_type': split_type, 'seed': seed,
                        'n_train': n_tr, 'n_val': n_v, 'n_test': n_te,
                        'val_rmse': round(float(np.sqrt(mean_squared_error(y_v, yp_v))), 4),
                        'val_mae': round(float(mean_absolute_error(y_v, yp_v)), 4),
                        'val_pearson': round(float(pearsonr(y_v, yp_v)[0]), 4),
                        'test_rmse': round(float(np.sqrt(mean_squared_error(y_te, yp_te))), 4),
                        'test_mae': round(float(mean_absolute_error(y_te, yp_te)), 4),
                        'test_pearson': round(float(pearsonr(y_te, yp_te)[0]), 4),
                        'test_spearman': round(float(spearmanr(y_te, yp_te)[0]), 4),
                        'n_params': n_params,
                        'train_time_s': round(time.time() - t0, 1),
                        'epochs_trained': '', 'device': 'CPU',
                    }
                    writer.writerow(row)
                    f.flush()
                    print(f"  RMSE={row['test_rmse']:.4f} R={row['test_pearson']:.4f} ({row['train_time_s']:.0f}s)", flush=True)
                    
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)
                    import traceback; traceback.print_exc()

print(f"\nPhase 3a DONE: {done_count} fits attempted", flush=True)
