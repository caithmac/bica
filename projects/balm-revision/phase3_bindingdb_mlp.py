#!/usr/bin/env python3
"""BindingDB MLP-only — loads precomputed features, fixes dtype, runs MLPs."""
import csv, os, sys, time, warnings, pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn

warnings.filterwarnings('ignore')
sys.path.insert(0, "E:/Drug Discovery")
from harness.trainer import train_torch

SEEDS = [42, 123, 456]
SPLIT_TYPES = ['random', 'scaffold', 'cold_target', 'seq_clustered/30pct', 'seq_clustered/40pct']
SPLIT_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/splits_bindingdb")
RESULTS_CSV = "E:/Drug Discovery/projects/balm-revision/phase3_bindingdb_results.csv"
CACHE_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/cache_bindingdb")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
print(f"Device: {DEVICE}")

with open(CACHE_DIR/"ecfp_cache.pkl",'rb') as f: ecfp_dict = pickle.load(f)
with open(CACHE_DIR/"aac_cache.pkl",'rb') as f: aac_dict = pickle.load(f)
print(f"Loaded {len(ecfp_dict):,} ECFP + {len(aac_dict):,} AAC entries")

def build_features(df):
    ef = np.stack([ecfp_dict[s] for s in df['Drug_canonical']]).astype(np.float32)
    aa = np.stack([aac_dict[s] for s in df['Target']]).astype(np.float32)
    return ef, np.concatenate([ef, aa], axis=1)

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []; prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]); prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def run_mlp(Xtr, ytr, Xv, yv, Xte, yte, hidden, sd):
    torch.manual_seed(sd); np.random.seed(sd)
    Xtr = Xtr.astype(np.float32); Xv = Xv.astype(np.float32); Xte = Xte.astype(np.float32)
    ytr = ytr.astype(np.float32); yv = yv.astype(np.float32); yte = yte.astype(np.float32)
    m = MLPHead(Xtr.shape[1], hidden)
    np_ = sum(p.numel() for p in m.parameters())
    t0 = time.time()
    vm, tm, _, ep, _ = train_torch(m, Xtr, ytr, Xv, yv, Xte, yte,
                                    lr=5e-4, batch_size=BATCH_SIZE, max_epochs=200,
                                    patience=15, weight_decay=0.01)
    t = time.time()-t0
    return {
        'val_rmse': float(vm['rmse']), 'val_mae': float(vm.get('mae', 0)),
        'val_pearson': float(vm.get('pearson_r', 0)),
        'test_rmse': float(tm['rmse']), 'test_mae': float(tm.get('mae', 0)),
        'test_pearson': float(tm.get('pearson_r', 0)),
        'test_spearman': float(tm.get('spearman_r', 0)),
        'n_params': np_, 'train_time_s': round(t, 1), 'epochs_trained': ep,
    }

MODELS = [
    ('MLP_shallow_ECFP4+AAC', 'mlp', 'aac', lambda Xtr,ytr,Xv,yv,Xte,yte,sd: run_mlp(Xtr,ytr,Xv,yv,Xte,yte,[256],sd)),
    ('MLP_shallow_ECFP4-only', 'mlp', 'ecfp', lambda Xtr,ytr,Xv,yv,Xte,yte,sd: run_mlp(Xtr,ytr,Xv,yv,Xte,yte,[256],sd)),
    ('MLP_deep_ECFP4+AAC', 'mlp', 'aac', lambda Xtr,ytr,Xv,yv,Xte,yte,sd: run_mlp(Xtr,ytr,Xv,yv,Xte,yte,[512,256,128],sd)),
    ('MLP_deep_ECFP4-only', 'mlp', 'ecfp', lambda Xtr,ytr,Xv,yv,Xte,yte,sd: run_mlp(Xtr,ytr,Xv,yv,Xte,yte,[512,256,128],sd)),
]

FIELDS = ['timestamp','model','model_family','split_type','seed','n_train','n_val','n_test',
          'val_rmse','val_mae','val_pearson','test_rmse','test_mae','test_pearson','test_spearman',
          'n_params','train_time_s','epochs_trained','device']

existing = set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            existing.add((row['model'], row['split_type'], row['seed']))
print(f"Skipping {len(existing)} existing")

total = len(MODELS) * len(SPLIT_TYPES) * len(SEEDS)
cnt = 0

with open(RESULTS_CSV, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)

    for st in SPLIT_TYPES:
        for sd in SEEDS:
            sdir = SPLIT_DIR / st / f"seed_{sd}"
            tr = pd.read_csv(sdir/"train.csv")
            va = pd.read_csv(sdir/"val.csv")
            te = pd.read_csv(sdir/"test.csv")
            nt, nv, nte = len(tr), len(va), len(te)
            ytr = tr['Y'].values; yv = va['Y'].values; yte = te['Y'].values
            ef_tr, Xa_tr = build_features(tr)
            ef_va, Xa_va = build_features(va)
            ef_te, Xa_te = build_features(te)

            for name, fam, feat, fn in MODELS:
                cnt += 1
                if (name, st, str(sd)) in existing:
                    continue
                Xtr_ = Xa_tr if feat == 'aac' else ef_tr
                Xv_ = Xa_va if feat == 'aac' else ef_va
                Xte_ = Xa_te if feat == 'aac' else ef_te
                print(f"[{cnt}/{total}] {name} | {st} | seed={sd} | dim={Xtr_.shape[1]}", flush=True)
                try:
                    r = fn(Xtr_, ytr, Xv_, yv, Xte_, yte, sd)
                    w.writerow({'timestamp': datetime.utcnow().isoformat(), 'model': name,
                                'model_family': fam, 'split_type': st, 'seed': sd,
                                'n_train': nt, 'n_val': nv, 'n_test': nte, **r,
                                'device': DEVICE})
                    f.flush()
                    print(f"  -> test_rmse={r['test_rmse']:.4f} pearson={r['test_pearson']:.4f} time={r['train_time_s']:.0f}s", flush=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  FAILED: {e}", flush=True)

print(f"\n[DONE]")
