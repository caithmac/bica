#!/usr/bin/env python3
"""Debug MLP — single fit to find the bug."""
import pickle, numpy as np, pandas as pd, torch, torch.nn as nn, sys, time, traceback
sys.path.insert(0, "E:/Drug Discovery")
from harness.trainer import train_torch

CACHE_DIR = "E:/Drug Discovery/projects/balm-revision/data/cache_bindingdb"
with open(f"{CACHE_DIR}/ecfp_cache.pkl",'rb') as f: ecfp_dict = pickle.load(f)
with open(f"{CACHE_DIR}/aac_cache.pkl",'rb') as f: aac_dict = pickle.load(f)

SPLIT_DIR = "E:/Drug Discovery/projects/balm-revision/data/splits_bindingdb"

class MLPHead(nn.Module):
    def __init__(self, d, hidden, dropout=0.2):
        super().__init__()
        layers = []; prev = d
        for h in hidden: layers.extend([nn.Linear(prev,h),nn.ReLU(),nn.Dropout(dropout)]); prev=h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

# Test each split type
for st in ['random','scaffold','cold_target','seq_clustered/30pct','seq_clustered/40pct']:
    try:
        tr = pd.read_csv(f"{SPLIT_DIR}/{st}/seed_42/train.csv")
        va = pd.read_csv(f"{SPLIT_DIR}/{st}/seed_42/val.csv")
        te = pd.read_csv(f"{SPLIT_DIR}/{st}/seed_42/test.csv")
        
        ef_tr = np.stack([ecfp_dict[s] for s in tr['Drug_canonical']]).astype(np.float32)
        aa_tr = np.stack([aac_dict[s] for s in tr['Target']]).astype(np.float32)
        Xtr = np.concatenate([ef_tr, aa_tr], axis=1)
        ef_va = np.stack([ecfp_dict[s] for s in va['Drug_canonical']]).astype(np.float32)
        aa_va = np.stack([aac_dict[s] for s in va['Target']]).astype(np.float32)
        Xv = np.concatenate([ef_va, aa_va], axis=1)
        ef_te = np.stack([ecfp_dict[s] for s in te['Drug_canonical']]).astype(np.float32)
        aa_te = np.stack([aac_dict[s] for s in te['Target']]).astype(np.float32)
        Xte = np.concatenate([ef_te, aa_te], axis=1)
        
        ytr = tr['Y'].values.astype(np.float32)
        yv = va['Y'].values.astype(np.float32)
        yte = te['Y'].values.astype(np.float32)
        
        print(f"{st}: Xtr={Xtr.shape} {Xtr.dtype}, ytr={ytr.shape} {ytr.dtype}")
        
        torch.manual_seed(42)
        m = MLPHead(Xtr.shape[1], [256])
        t0 = time.time()
        vm,tm,_,ep,_ = train_torch(m, Xtr, ytr, Xv, yv, Xte, yte,
                                    lr=5e-4, batch_size=128, max_epochs=30,
                                    patience=5, weight_decay=0.01)
        print(f"  -> test_rmse={tm['rmse']:.4f}, {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"  FAILED {st}: {e}")
        traceback.print_exc()
