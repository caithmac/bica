#!/usr/bin/env python3
"""Phase 3: Fair model benchmark across all splits.
10 models × 5 split types × 3 seeds = 150 fits (RF fast, DL slow).
Uses harness from E:/Drug Discovery/harness/ for featurizers and trainers.
Results saved incrementally to phase3_results.csv.
"""
import csv, json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# Add harness to path
sys.path.insert(0, "E:/Drug Discovery")
from harness.data import load_raw, get_splits_for_seed as harness_scaffold_split
from harness.featurizers import esm2_embeddings, chemberta_embeddings, ecfp, amino_acid_composition, concat
from harness.trainer import train_torch, train_sklearn

SEEDS = [42, 123, 456]
SPLIT_TYPES = ['random', 'scaffold', 'cold_target', 'seq_clustered/30pct', 'seq_clustered/40pct']
OUT_DIR = "E:/Drug Discovery/projects/balm-revision/data/splits"
RESULTS_CSV = "E:/Drug Discovery/projects/balm-revision/phase3_results.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

print(f"Device: {DEVICE}")

# --- Load frozen BALM data ---
df = pd.read_parquet("E:/Drug Discovery/projects/balm-revision/data/frozen/balm_filtered.parquet")

# ============================================================================
# Model definitions
# ============================================================================
class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================================
# Feature extraction (cached)
# ============================================================================
FEAT_CACHE = {}
def get_features(df_split, model_key):
    """Extract and cache features per model type."""
    cache_key = (model_key, id(df_split))
    if cache_key in FEAT_CACHE:
        return FEAT_CACHE[cache_key]
    
    y = df_split['Y'].values.astype(np.float32)
    
    if model_key == 'rf_ecfp4_aac':
        X = concat(ecfp(df_split['Drug_canonical'].tolist()), 
                   amino_acid_composition(df_split['Target'].tolist()))
    elif model_key == 'rf_ecfp4_esm2_8m':
        X = concat(ecfp(df_split['Drug_canonical'].tolist()),
                   esm2_embeddings(df_split['Target'].tolist(), model_size="8M"))
    elif model_key == 'rf_ecfp4_only':
        X = ecfp(df_split['Drug_canonical'].tolist())
    elif model_key in ('mlp_shallow_ecfp4_aac', 'mlp_deep_ecfp4_aac'):
        X = concat(ecfp(df_split['Drug_canonical'].tolist()),
                   amino_acid_composition(df_split['Target'].tolist()))
    elif model_key == 'mlp_deep_esm2_8m':
        X = concat(ecfp(df_split['Drug_canonical'].tolist()),
                   esm2_embeddings(df_split['Target'].tolist(), model_size="8M"))
    elif model_key == 'mlp_deep_cb_esm2_8m':
        X = concat(chemberta_embeddings(df_split['Drug_canonical'].tolist()),
                   esm2_embeddings(df_split['Target'].tolist(), model_size="8M"))
    else:
        raise ValueError(f"Unknown model_key: {model_key}")
    
    FEAT_CACHE[cache_key] = (X, y)
    return X, y

# ============================================================================
# Model runners
# ============================================================================
def run_rf(X_train, y_train, X_val, y_val, X_test, y_test, seed):
    """RF 500 trees, max_depth=20."""
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    yp_val = rf.predict(X_val)
    yp_test = rf.predict(X_test)
    
    return {
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, yp_val))),
        'val_mae': float(mean_absolute_error(y_val, yp_val)),
        'val_pearson': float(pearsonr(y_val, yp_val)[0]),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, yp_test))),
        'test_mae': float(mean_absolute_error(y_test, yp_test)),
        'test_pearson': float(pearsonr(y_test, yp_test)[0]),
        'test_spearman': float(spearmanr(y_test, yp_test)[0]),
        'n_params': sum(t.tree_.node_count for t in rf.estimators_),
        'train_time_s': 0,  # too fast to measure
    }

def run_mlp(X_train, y_train, X_val, y_val, X_test, y_test, hidden_dims, seed, name, lr=5e-4):
    """Train MLP with harness."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = X_train.shape[1]
    model = MLPHead(input_dim, hidden_dims).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    
    t0 = time.time()
    val_metrics, test_metrics, train_time, epochs, test_pred = train_torch(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        lr=lr, batch_size=BATCH_SIZE, epochs=200, patience=15, device=DEVICE,
        weight_decay=0.01
    )
    train_time_s = time.time() - t0
    
    return {
        'val_rmse': float(val_metrics['rmse']),
        'val_mae': float(val_metrics.get('mae', 0)),
        'val_pearson': float(val_metrics.get('pearson_r', 0)),
        'test_rmse': float(test_metrics['rmse']),
        'test_mae': float(test_metrics.get('mae', 0)),
        'test_pearson': float(test_metrics.get('pearson_r', 0)),
        'test_spearman': float(test_metrics.get('spearman_r', 0)),
        'n_params': n_params,
        'train_time_s': round(train_time_s, 1),
        'epochs_trained': epochs,
    }

# ============================================================================
# Model registry
# ============================================================================
MODELS = [
    {'key': 'rf_ecfp4_aac', 'name': 'RF_ECFP4+AAC', 'family': 'tree', 'fn': run_rf,
     'features': ['ecfp4', 'aac'], 'n_params_approx': '~6M'},
    {'key': 'rf_ecfp4_esm2_8m', 'name': 'RF_ECFP4+ESM2-8M', 'family': 'tree', 'fn': run_rf,
     'features': ['ecfp4', 'esm2_8M'], 'n_params_approx': '~6M'},
    {'key': 'rf_ecfp4_only', 'name': 'RF_ECFP4-only', 'family': 'tree', 'fn': run_rf,
     'features': ['ecfp4'], 'n_params_approx': '~4M'},
    {'key': 'mlp_shallow_ecfp4_aac', 'name': 'MLP_shallow_ECFP4+AAC', 'family': 'mlp',
     'fn': lambda *a, **kw: run_mlp(*a, hidden_dims=[256], **kw),
     'features': ['ecfp4', 'aac'], 'n_params_approx': '~267K'},
    {'key': 'mlp_deep_ecfp4_aac', 'name': 'MLP_deep_ECFP4+AAC', 'family': 'mlp',
     'fn': lambda *a, **kw: run_mlp(*a, hidden_dims=[512, 256, 128], **kw),
     'features': ['ecfp4', 'aac'], 'n_params_approx': '~1.7M'},
    {'key': 'mlp_deep_esm2_8m', 'name': 'MLP_deep_ECFP4+ESM2-8M', 'family': 'mlp',
     'fn': lambda *a, **kw: run_mlp(*a, hidden_dims=[512, 256, 128], **kw),
     'features': ['ecfp4', 'esm2_8M'], 'n_params_approx': '~1.7M'},
    {'key': 'mlp_deep_cb_esm2_8m', 'name': 'MLP_deep_ChemBERTa+ESM2-8M', 'family': 'mlp',
     'fn': lambda *a, **kw: run_mlp(*a, hidden_dims=[512, 256, 128], **kw),
     'features': ['chemberta', 'esm2_8M'], 'n_params_approx': '~1.7M'},
]

# ============================================================================
# Main loop
# ============================================================================
FIELD_NAMES = ['timestamp', 'model', 'model_family', 'split_type', 'seed',
               'n_train', 'n_val', 'n_test',
               'val_rmse', 'val_mae', 'val_pearson',
               'test_rmse', 'test_mae', 'test_pearson', 'test_spearman',
               'n_params', 'train_time_s', 'epochs_trained', 'device']

existing = set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            existing.add((row['model'], row['split_type'], row['seed']))
    print(f"Resuming: {len(existing)} completed fits found")

with open(RESULTS_CSV, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
    if not existing:
        writer.writeheader()
    
    total = len(MODELS) * len(SPLIT_TYPES) * len(SEEDS)
    done = 0
    
    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            # Load split
            split_dir = Path(OUT_DIR) / split_type / f"seed_{seed}"
            train_df = pd.read_csv(split_dir / "train.csv")
            val_df = pd.read_csv(split_dir / "val.csv")
            test_df = pd.read_csv(split_dir / "test.csv")
            
            n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
            
            for model_info in MODELS:
                done += 1
                
                # Skip if already done
                if (model_info['name'], split_type, str(seed)) in existing:
                    continue
                
                print(f"[{done}/{total}] {model_info['name']} | {split_type} | seed={seed}")
                
                try:
                    # Extract features
                    X_train, y_train = get_features(train_df, model_info['key'])
                    X_val, y_val = get_features(val_df, model_info['key'])
                    X_test, y_test = get_features(test_df, model_info['key'])
                    
                    # Run model
                    result = model_info['fn'](
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        seed=seed, name=model_info['name']
                    )
                    
                    # Write row
                    row = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'model': model_info['name'],
                        'model_family': model_info['family'],
                        'split_type': split_type,
                        'seed': seed,
                        'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
                        **result,
                        'device': DEVICE,
                        'epochs_trained': result.get('epochs_trained', ''),
                    }
                    writer.writerow(row)
                    f.flush()
                    
                    print(f"  -> test_rmse={result['test_rmse']:.4f}, pearson={result['test_pearson']:.4f}, "
                          f"time={result.get('train_time_s', 0):.0f}s")
                    
                except Exception as e:
                    print(f"  FAILED: {e}")
                    continue

print(f"\n[Phase 3] COMPLETE — Results: {RESULTS_CSV}")
