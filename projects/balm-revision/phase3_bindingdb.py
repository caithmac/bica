#!/usr/bin/env python3
"""BindingDB benchmark — precompute features once, then run all models."""
import csv, os, sys, time, warnings, pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
import torch, torch.nn as nn

warnings.filterwarnings('ignore')
sys.path.insert(0, "E:/Drug Discovery")
from harness.featurizers import ecfp, amino_acid_composition, concat
from harness.trainer import train_torch

SEEDS = [42, 123, 456]
SPLIT_TYPES = ['random', 'scaffold', 'cold_target', 'seq_clustered/30pct', 'seq_clustered/40pct']
SPLIT_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/splits_bindingdb")
RESULTS_CSV = "E:/Drug Discovery/projects/balm-revision/phase3_bindingdb_results.csv"
CACHE_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/cache_bindingdb")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

# ── Step 1: Collect ALL unique SMILES + sequences across all splits ──
print("\n=== Collecting unique molecules and sequences ===")
all_smiles = set()
all_seqs = set()
for st in SPLIT_TYPES:
    for sd in SEEDS:
        for fname in ['train.csv','val.csv','test.csv']:
            df = pd.read_csv(SPLIT_DIR/st/f"seed_{sd}"/fname)
            all_smiles.update(df['Drug_canonical'].tolist())
            all_seqs.update(df['Target'].tolist())

all_smiles = sorted(all_smiles)
all_seqs = sorted(all_seqs)
print(f"  Unique SMILES: {len(all_smiles):,}")
print(f"  Unique sequences: {len(all_seqs):,}")

# ── Step 2: Precompute features once ──
print("\n=== Precomputing ECFP4 fingerprints ===")
t0 = time.time()
ecfp_dict = {}
# ECFP returns numpy array per molecule — batch them
batch_size = 5000
for i in range(0, len(all_smiles), batch_size):
    batch = all_smiles[i:i+batch_size]
    fps = ecfp(batch)
    for j, smi in enumerate(batch):
        ecfp_dict[smi] = fps[j]
    print(f"  {min(i+batch_size, len(all_smiles)):,}/{len(all_smiles):,} ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== Precomputing AAC compositions ===")
t0 = time.time()
aac_dict = {}
for i in range(0, len(all_seqs), batch_size):
    batch = all_seqs[i:i+batch_size]
    aacs = amino_acid_composition(batch)
    for j, seq in enumerate(batch):
        aac_dict[seq] = aacs[j]
    print(f"  {min(i+batch_size, len(all_seqs)):,}/{len(all_seqs):,} ({time.time()-t0:.0f}s)", flush=True)

# Save cache
with open(CACHE_DIR/"ecfp_cache.pkl",'wb') as f: pickle.dump(ecfp_dict, f)
with open(CACHE_DIR/"aac_cache.pkl",'wb') as f: pickle.dump(aac_dict, f)
print("Caches saved.")

# ── Step 3: Build features for a split ──
def build_features(df):
    ecfp_rows = np.stack([ecfp_dict[s] for s in df['Drug_canonical']])
    aac_rows = np.stack([aac_dict[s] for s in df['Target']])
    return ecfp_rows, concat(ecfp_rows, aac_rows)

# ── Model definitions ──
class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []; prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]); prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def _m(y,yp): return float(np.sqrt(mean_squared_error(y,yp))), float(mean_absolute_error(y,yp)), float(pearsonr(y,yp)[0]), float(spearmanr(y,yp)[0])

def run_xgb(Xtr,ytr,Xv,yv,Xte,yte,sd):
    t0=time.time()
    m=xgb.XGBRegressor(n_estimators=500,max_depth=8,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,tree_method='hist',device=DEVICE,random_state=sd,verbosity=0)
    m.fit(Xtr,ytr); t=time.time()-t0
    vr,vm,vp,vs=_m(yv,m.predict(Xv)); tr_,tm,tp,ts=_m(yte,m.predict(Xte))
    return {'val_rmse':vr,'val_mae':vm,'val_pearson':vp,'test_rmse':tr_,'test_mae':tm,'test_pearson':tp,'test_spearman':ts,'train_time_s':round(t,1)}

def run_rf(Xtr,ytr,Xv,yv,Xte,yte,sd):
    t0=time.time()
    m=RandomForestRegressor(n_estimators=500,max_depth=20,random_state=sd,n_jobs=-1); m.fit(Xtr,ytr); t=time.time()-t0
    vr,vm,vp,vs=_m(yv,m.predict(Xv)); tr_,tm,tp,ts=_m(yte,m.predict(Xte))
    return {'val_rmse':vr,'val_mae':vm,'val_pearson':vp,'test_rmse':tr_,'test_mae':tm,'test_pearson':tp,'test_spearman':ts,'n_params':sum(x.tree_.node_count for x in m.estimators_),'train_time_s':round(t,1)}

def run_knn(Xtr,ytr,Xv,yv,Xte,yte,sd):
    t0=time.time(); m=KNeighborsRegressor(n_neighbors=5,weights='distance',n_jobs=-1); m.fit(Xtr,ytr); t=time.time()-t0
    vr,vm,vp,vs=_m(yv,m.predict(Xv)); tr_,tm,tp,ts=_m(yte,m.predict(Xte))
    return {'val_rmse':vr,'val_mae':vm,'val_pearson':vp,'test_rmse':tr_,'test_mae':tm,'test_pearson':tp,'test_spearman':ts,'train_time_s':round(t,1)}

def run_ridge(Xtr,ytr,Xv,yv,Xte,yte,sd):
    t0=time.time(); m=Ridge(alpha=1.0); m.fit(Xtr,ytr); t=time.time()-t0
    vr,vm,vp,vs=_m(yv,m.predict(Xv)); tr_,tm,tp,ts=_m(yte,m.predict(Xte))
    return {'val_rmse':vr,'val_mae':vm,'val_pearson':vp,'test_rmse':tr_,'test_mae':tm,'test_pearson':tp,'test_spearman':ts,'train_time_s':round(t,1)}

def run_mlp(Xtr,ytr,Xv,yv,Xte,yte,hidden,sd):
    torch.manual_seed(sd); np.random.seed(sd)
    m=MLPHead(Xtr.shape[1],hidden); np_=sum(p.numel() for p in m.parameters())
    t0=time.time(); vm,tm,_,ep,_=train_torch(m,Xtr,ytr,Xv,yv,Xte,yte,lr=5e-4,batch_size=BATCH_SIZE,max_epochs=200,patience=15,weight_decay=0.01); t=time.time()-t0
    return {'val_rmse':float(vm['rmse']),'val_mae':float(vm.get('mae',0)),'val_pearson':float(vm.get('pearson_r',0)),
            'test_rmse':float(tm['rmse']),'test_mae':float(tm.get('mae',0)),'test_pearson':float(tm.get('pearson_r',0)),
            'test_spearman':float(tm.get('spearman_r',0)),'n_params':np_,'train_time_s':round(t,1),'epochs_trained':ep}

MODELS = [
    ('XGBoost_ECFP4+AAC','xgb','aac',run_xgb),
    ('RF_ECFP4+AAC','tree','aac',run_rf),
    ('RF_ECFP4-only','tree','ecfp',run_rf),
    ('KNN_ECFP4+AAC','knn','aac',run_knn),
    ('Ridge_ECFP4+AAC','linear','aac',run_ridge),
    ('MLP_shallow_ECFP4+AAC','mlp','aac',lambda *a:run_mlp(*a,[256])),
    ('MLP_shallow_ECFP4-only','mlp','ecfp',lambda *a:run_mlp(*a,[256])),
    ('MLP_deep_ECFP4+AAC','mlp','aac',lambda *a:run_mlp(*a,[512,256,128])),
    ('MLP_deep_ECFP4-only','mlp','ecfp',lambda *a:run_mlp(*a,[512,256,128])),
]

# ── Run ──
FIELDS = ['timestamp','model','model_family','split_type','seed','n_train','n_val','n_test',
          'val_rmse','val_mae','val_pearson','test_rmse','test_mae','test_pearson','test_spearman',
          'n_params','train_time_s','epochs_trained','device']

existing=set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f): existing.add((row['model'],row['split_type'],row['seed']))
    print(f"\nResuming: {len(existing)} completed")

total=len(MODELS)*len(SPLIT_TYPES)*len(SEEDS); cnt=0

with open(RESULTS_CSV,'a',newline='') as f:
    w=csv.DictWriter(f,fieldnames=FIELDS)
    if not existing: w.writeheader()

    for st in SPLIT_TYPES:
        for sd in SEEDS:
            sdir=SPLIT_DIR/st/f"seed_{sd}"
            print(f"\n  Building features for {st} seed={sd}...",flush=True)
            tr=pd.read_csv(sdir/"train.csv"); va=pd.read_csv(sdir/"val.csv"); te=pd.read_csv(sdir/"test.csv")
            nt,nv,nte=len(tr),len(va),len(te)
            ytr=tr['Y'].values.astype(np.float32); yv=va['Y'].values.astype(np.float32); yte=te['Y'].values.astype(np.float32)
            ef_tr,Xa_tr=build_features(tr); ef_va,Xa_va=build_features(va); ef_te,Xa_te=build_features(te)

            for name,fam,feat,fn in MODELS:
                cnt+=1
                if (name,st,str(sd)) in existing: continue
                Xtr_=Xa_tr if feat=='aac' else ef_tr; Xv_=Xa_va if feat=='aac' else ef_va; Xte_=Xa_te if feat=='aac' else ef_te
                print(f"  [{cnt}/{total}] {name} | {st} | seed={sd} | dim={Xtr_.shape[1]}",flush=True)
                try:
                    r=fn(Xtr_,ytr,Xv_,yv,Xte_,yte,sd)
                    w.writerow({'timestamp':datetime.utcnow().isoformat(),'model':name,'model_family':fam,'split_type':st,'seed':sd,
                                'n_train':nt,'n_val':nv,'n_test':nte,**r,'device':DEVICE,
                                'epochs_trained':r.get('epochs_trained',''),'n_params':r.get('n_params','')})
                    f.flush()
                    print(f"    -> test_rmse={r['test_rmse']:.4f} pearson={r['test_pearson']:.4f} time={r['train_time_s']:.0f}s",flush=True)
                except Exception as e:
                    print(f"    FAILED: {e}",flush=True)

print(f"\n[DONE] {RESULTS_CSV}")
