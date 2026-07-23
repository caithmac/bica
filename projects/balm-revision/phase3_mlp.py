#!/usr/bin/env python3
"""Phase 3 MLP-only resumer — picks up from existing CSV."""
import csv, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
sys.path.insert(0, "E:/Drug Discovery")
from harness.featurizers import ecfp, amino_acid_composition, concat
from harness.trainer import train_torch

SEEDS = [42, 123, 456]
SPLIT_TYPES = ['random', 'scaffold', 'cold_target', 'seq_clustered/30pct', 'seq_clustered/40pct']
SPLIT_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/splits")
RESULTS_CSV = "E:/Drug Discovery/projects/balm-revision/phase3_results.csv"
BATCH_SIZE = 128

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

def run_mlp(X_train, y_train, X_val, y_val, X_test, y_test, hidden_dims, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    input_dim = X_train.shape[1]
    model = MLPHead(input_dim, hidden_dims)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    val_metrics, test_metrics, train_time, epochs, _ = train_torch(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        lr=5e-4, batch_size=BATCH_SIZE, max_epochs=200, patience=15, weight_decay=0.01
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

MLP_MODELS = [
    ('MLP_shallow_ECFP4+AAC',  [256]),
    ('MLP_shallow_ECFP4-only', [256]),
    ('MLP_deep_ECFP4+AAC',     [512, 256, 128]),
    ('MLP_deep_ECFP4-only',    [512, 256, 128]),
]

total = len(MLP_MODELS) * len(SPLIT_TYPES) * len(SEEDS)
cnt = 0

with open(RESULTS_CSV, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)

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

            ecfp_tr = ecfp(train_df['Drug_canonical'].tolist())
            ecfp_va = ecfp(val_df['Drug_canonical'].tolist())
            ecfp_te = ecfp(test_df['Drug_canonical'].tolist())
            aac_tr = amino_acid_composition(train_df['Target'].tolist())
            aac_va = amino_acid_composition(val_df['Target'].tolist())
            aac_te = amino_acid_composition(test_df['Target'].tolist())

            X_ecfp_aac_tr = concat(ecfp_tr, aac_tr)
            X_ecfp_aac_va = concat(ecfp_va, aac_va)
            X_ecfp_aac_te = concat(ecfp_te, aac_te)

            for model_name, hidden_dims in MLP_MODELS:
                cnt += 1
                if (model_name, split_type, str(seed)) in existing:
                    continue

                if 'AAC' in model_name:
                    Xtr, Xv, Xte = X_ecfp_aac_tr, X_ecfp_aac_va, X_ecfp_aac_te
                else:
                    Xtr, Xv, Xte = ecfp_tr, ecfp_va, ecfp_te

                print(f"[{cnt}/{total}] {model_name} | {split_type} | seed={seed} | dim={Xtr.shape[1]}", flush=True)
                try:
                    result = run_mlp(Xtr, y_train, Xv, y_val, Xte, y_test, hidden_dims, seed)
                    writer.writerow({
                        'timestamp': datetime.utcnow().isoformat(),
                        'model': model_name, 'model_family': 'mlp',
                        'split_type': split_type, 'seed': seed,
                        'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
                        **result, 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    })
                    f.flush()
                    print(f"  -> test_rmse={result['test_rmse']:.4f}, pearson={result['test_pearson']:.4f}, time={result['train_time_s']:.0f}s", flush=True)
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)

print(f"\n[Phase 3 MLP] DONE.")
