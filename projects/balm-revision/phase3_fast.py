#!/usr/bin/env python3
"""Phase 3: Precompute ESM-2 embeddings once, then benchmark all models."""
import csv, os, sys, time, warnings, pickle
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
sys.path.insert(0, "E:/Drug Discovery")
from harness.featurizers import esm2_embeddings, chemberta_embeddings, ecfp, amino_acid_composition, concat
from harness.trainer import train_torch

SEEDS = [42, 123, 456]
SPLIT_TYPES = ['random', 'scaffold', 'cold_target', 'seq_clustered/30pct', 'seq_clustered/40pct']
SPLIT_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/splits")
CACHE_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/cache")
RESULTS_CSV = "E:/Drug Discovery/projects/balm-revision/phase3_results.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")

# ============================================================================
# Precompute ESM-2 embeddings once per split
# ============================================================================
print("\n=== Precomputing ESM-2 embeddings (once per split) ===\n")
for split_type in SPLIT_TYPES:
    for seed in SEEDS:
        cache_file = CACHE_DIR / f"esm2_8M_{split_type.replace('/', '_')}_seed{seed}.pkl"
        if cache_file.exists():
            print(f"  [SKIP] {cache_file.name}")
            continue
        print(f"  Computing: {split_type} seed={seed}")
        sdir = SPLIT_DIR / split_type / f"seed_{seed}"
        train_df = pd.read_csv(sdir / "train.csv")
        val_df = pd.read_csv(sdir / "val.csv")
        test_df = pd.read_csv(sdir / "test.csv")
        all_seqs = pd.concat([train_df['Target'], val_df['Target'], test_df['Target']]).tolist()
        n_seqs = len(all_seqs)
        print(f"    {n_seqs} sequences total")
        emb = esm2_embeddings(all_seqs, model_size="8M", batch_size=32)
        with open(cache_file, 'wb') as f:
            pickle.dump({'train_n': len(train_df), 'val_n': len(val_df), 'embeddings': emb}, f)
        print(f"    Saved {emb.shape}")

print("\n=== ESM-2 precomputation DONE ===\n")

# ============================================================================
# Helpers
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

def load_esm2_cache(split_type, seed):
    cache_file = CACHE_DIR / f"esm2_8M_{split_type.replace('/', '_')}_seed{seed}.pkl"
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    emb = data['embeddings']
    n_train, n_val = data['train_n'], data['val_n']
    return emb[:n_train], emb[n_train:n_train+n_val], emb[n_train+n_val:]

def run_rf(X_train, y_train, X_val, y_val, X_test, y_test, seed):
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
        'train_time_s': 0,
    }

def run_mlp(X_train, y_train, X_val, y_val, X_test, y_test, hidden_dims, seed, name, lr=5e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    input_dim = X_train.shape[1]
    model = MLPHead(input_dim, hidden_dims).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    val_metrics, test_metrics, train_time, epochs, test_pred = train_torch(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        lr=lr, batch_size=BATCH_SIZE, epochs=200, patience=15, device=DEVICE, weight_decay=0.01
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
        'n_params': n_params, 'train_time_s': round(train_time_s, 1),
        'epochs_trained': epochs,
    }

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

    total = 7 * len(SPLIT_TYPES) * len(SEEDS)
    done = 0

    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            sdir = SPLIT_DIR / split_type / f"seed_{seed}"
            train_df = pd.read_csv(sdir / "train.csv")
            val_df = pd.read_csv(sdir / "val.csv")
            test_df = pd.read_csv(sdir / "test.csv")
            n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
            y_train = train_df['Y'].values.astype(np.float32)
            y_val = val_df['Y'].values.astype(np.float32)
            y_test = test_df['Y'].values.astype(np.float32)

            # Precompute local features per split once
            print(f"\n  Featurizing: {split_type} seed={seed}")
            ecfp_train = ecfp(train_df['Drug_canonical'].tolist())
            ecfp_val = ecfp(val_df['Drug_canonical'].tolist())
            ecfp_test = ecfp(test_df['Drug_canonical'].tolist())
            aac_train = amino_acid_composition(train_df['Target'].tolist())
            aac_val = amino_acid_composition(val_df['Target'].tolist())
            aac_test = amino_acid_composition(test_df['Target'].tolist())
            esm2_train, esm2_val, esm2_test = load_esm2_cache(split_type, seed)
            cb_train = chemberta_embeddings(train_df['Drug_canonical'].tolist())
            cb_val = chemberta_embeddings(val_df['Drug_canonical'].tolist())
            cb_test = chemberta_embeddings(test_df['Drug_canonical'].tolist())
            print(f"  Featurization done.")

            # --- RF models ---
            for model_name, Xtr, Xv, Xte in [
                ('RF_ECFP4+AAC', concat(ecfp_train, aac_train), concat(ecfp_val, aac_val), concat(ecfp_test, aac_test)),
                ('RF_ECFP4+ESM2-8M', concat(ecfp_train, esm2_train), concat(ecfp_val, esm2_val), concat(ecfp_test, esm2_test)),
                ('RF_ECFP4-only', ecfp_train, ecfp_val, ecfp_test),
            ]:
                done += 1
                if (model_name, split_type, str(seed)) in existing:
                    continue
                print(f"  [{done}/{total}] {model_name} | {split_type} | seed={seed}")
                result = run_rf(Xtr, y_train, Xv, y_val, Xte, y_test, seed=seed)
                writer.writerow({'timestamp': datetime.utcnow().isoformat(),
                    'model': model_name, 'model_family': 'tree', 'split_type': split_type,
                    'seed': seed, 'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
                    **result, 'device': DEVICE, 'epochs_trained': ''})
                f.flush()
                print(f"    -> test_rmse={result['test_rmse']:.4f}, pearson={result['test_pearson']:.4f}")

            # --- MLP models ---
            for model_name, hidden_dims, Xtr, Xv, Xte in [
                ('MLP_shallow_ECFP4+AAC', [256], concat(ecfp_train, aac_train), concat(ecfp_val, aac_val), concat(ecfp_test, aac_test)),
                ('MLP_deep_ECFP4+AAC', [512,256,128], concat(ecfp_train, aac_train), concat(ecfp_val, aac_val), concat(ecfp_test, aac_test)),
                ('MLP_deep_ECFP4+ESM2-8M', [512,256,128], concat(ecfp_train, esm2_train), concat(ecfp_val, esm2_val), concat(ecfp_test, esm2_test)),
                ('MLP_deep_ChemBERTa+ESM2-8M', [512,256,128], concat(cb_train, esm2_train), concat(cb_val, esm2_val), concat(cb_test, esm2_test)),
            ]:
                done += 1
                if (model_name, split_type, str(seed)) in existing:
                    continue
                print(f"  [{done}/{total}] {model_name} | {split_type} | seed={seed}")
                try:
                    result = run_mlp(Xtr, y_train, Xv, y_val, Xte, y_test, hidden_dims, seed, model_name)
                    writer.writerow({'timestamp': datetime.utcnow().isoformat(),
                        'model': model_name, 'model_family': 'mlp', 'split_type': split_type,
                        'seed': seed, 'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
                        **result, 'device': DEVICE})
                    f.flush()
                    print(f"    -> test_rmse={result['test_rmse']:.4f}, pearson={result['test_pearson']:.4f}")
                except Exception as e:
                    print(f"    FAILED: {e}")

print(f"\n[Phase 3] COMPLETE — Results: {RESULTS_CSV}")
