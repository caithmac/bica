# -*- coding: utf-8 -*-
"""E201 — Learning Curves. Ponytail: use harness directly, skip run_experiment.py."""
import sys, time, logging, os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(EXP_DIR / "run.log", mode="w"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("E201")

# ── Config ──────────────────────────────────────────────────────────────────────
CONFIG = {
    "experiment": "E201",
    "subset_sizes": [500, 1000, 2500, 5000, 10000, 17312],
    "models": [
        {"id": "rf_ecfp4_aac", "type": "sklearn", "family": "tree",
         "ligand_repr": "ecfp4_1024", "protein_repr": "aac_20"},
        {"id": "mlp_chemberta_esm2_8M", "type": "torch", "family": "mlp",
         "ligand_repr": "chemberta_600", "protein_repr": "esm2_8M_320", "mlp_arch": "deep",
         "freeze_esm": True},
        {"id": "mlp_chemberta_esm2_8M_ft3", "type": "torch", "family": "mlp",
         "ligand_repr": "chemberta_600", "protein_repr": "esm2_8M_320", "mlp_arch": "deep",
         "freeze_esm": False, "unfreeze_layers": 3},
    ],
    "seeds": [42, 123, 456],
    "split_seed": 42,
}
with open(EXP_DIR / "config.yaml", "w") as f:
    yaml.dump(CONFIG, f)

# ── Imports ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path("E:/Drug Discovery")))
os.chdir("E:/Drug Discovery")
from harness.config import CACHE_DIR, SPLIT_DIR, BATCH_SIZE, LEARNING_RATE, SPLIT_SEED, LABEL_COL, SMILES_COL, PROTEIN_COL
from harness.data import get_splits_for_seed
import harness.featurizers as F
from harness.trainer import train_sklearn, train_torch, count_parameters
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch, torch.nn as nn
from models.mlp import MLP

# ── Get full splits once ────────────────────────────────────────────────────────
log.info("Loading splits...")
train_df, val_df, test_df = get_splits_for_seed(CONFIG["split_seed"])
log.info(f"Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")

# ── Build features for full data (once, cached) ──────────────────────────────────
def build_ecfp_aac(df):
    return F.concat(F.ecfp(df[SMILES_COL].tolist(), radius=2, nbits=1024),
                     F.amino_acid_composition(df[PROTEIN_COL].tolist()))

# Prebuild full features
X_full_train = build_ecfp_aac(train_df)
X_full_test  = build_ecfp_aac(test_df)
y_full_train = train_df[LABEL_COL].values.astype(np.float32)
y_full_test  = test_df[LABEL_COL].values.astype(np.float32)

# For MLP: cached ChemBERTa + ESM-2 features
from pathlib import Path as P
FEAT_CACHE = P("cache/features")
FEAT_CACHE.mkdir(parents=True, exist_ok=True)

def get_chemberta_feats(smiles_list, tag):
    fpath = FEAT_CACHE / f"{tag}.npy"
    if fpath.exists():
        return np.load(fpath)
    arr = F.chemberta_embeddings(smiles_list, model_name="seyonec/ChemBERTa-zinc-base-v1")
    np.save(fpath, arr)
    return arr

def get_esm2_feats(seq_list, tag):
    fpath = FEAT_CACHE / f"{tag}.npy"
    if fpath.exists():
        return np.load(fpath)
    arr = F.esm2_embeddings(seq_list, model_size="8M")
    np.save(fpath, arr)
    return arr

# Build DL features (ChemBERTa + ESM-2)
log.info("Building ChemBERTa + ESM-2 features (cached)...")
L_train_dl = get_chemberta_feats(train_df[SMILES_COL].tolist(), "cb600_train")
L_test_dl  = get_chemberta_feats(test_df[SMILES_COL].tolist(), "cb600_test")
P_train_dl = get_esm2_feats(train_df[PROTEIN_COL].tolist(), "esm2_8M_train")
P_test_dl  = get_esm2_feats(test_df[PROTEIN_COL].tolist(), "esm2_8M_test")
X_train_dl = F.concat(L_train_dl, P_train_dl)
X_test_dl  = F.concat(L_test_dl, P_test_dl)
log.info(f"DL features: {X_train_dl.shape[1]} dim")

# ── Run learning curves ─────────────────────────────────────────────────────────
results = []

for model_cfg in CONFIG["models"]:
    mid = model_cfg["id"]
    mtype = model_cfg["type"]
    log.info(f"\n{'='*50}\n  {mid}\n{'='*50}")
    
    for n_train in CONFIG["subset_sizes"]:
        if n_train > len(train_df):
            continue
        
        for seed in CONFIG["seeds"]:
            rng = np.random.default_rng(seed)
            subset_idx = rng.choice(len(train_df), size=n_train, replace=False)
            tag = f"{mid}_n{n_train}_s{seed}"
            
            if mtype == "sklearn":
                X_sub = X_full_train[subset_idx]
                y_sub = y_full_train[subset_idx]
                
                t0 = time.time()
                model = RandomForestRegressor(n_estimators=500, max_depth=None,
                                              min_samples_split=2, n_jobs=-1, random_state=42)
                model.fit(X_sub, y_sub)
                y_pred = model.predict(X_full_test)
                train_time = time.time() - t0
            else:
                # Torch MLP
                X_sub = X_train_dl[subset_idx]
                y_sub = y_full_train[subset_idx]
                
                input_dim = X_train_dl.shape[1]
                model = MLP(input_dim=input_dim, hidden_dims=[512, 256, 128], dropout=0.2)
                n_params = count_parameters(model)
                
                t0 = time.time()
                val_m, test_m, train_time, epochs, test_pred = train_torch(
                    model, X_sub, y_sub, X_train_dl[:min(500, len(train_df))],
                    y_full_train[:min(500, len(train_df))], X_test_dl, y_full_test,
                    batch_size=min(64, n_train // 4 + 1), lr=5e-4,
                    max_epochs=100, patience=20,
                )
                y_pred = test_pred
            
            rmse = float(np.sqrt(np.mean((y_pred - y_full_test)**2)))
            pearson = float(np.corrcoef(y_pred, y_full_test)[0, 1])
            spearman = float(pd.Series(y_pred).corr(pd.Series(y_full_test), method="spearman"))
            
            results.append({
                "model": mid, "n_train": n_train, "seed": seed,
                "rmse": rmse, "pearson_r": pearson, "spearman_r": spearman,
                "train_time_s": round(train_time, 1),
            })
            log.info(f"  {mid:30s}  n={n_train:6,d}  s={seed}  "
                     f"RMSE={rmse:.4f}  R={pearson:.4f}  {train_time:.1f}s")

# ── Save ────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(EXP_DIR / "results.csv", index=False)

# Quick plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
for mid in results_df["model"].unique():
    sub = results_df[results_df["model"] == mid].groupby("n_train")
    means = sub["rmse"].mean()
    stds = sub["rmse"].std()
    ax.errorbar(means.index, means.values, yerr=stds.values,
                label=mid, marker="o", capsize=3, linewidth=2)
ax.set_xlabel("Training set size"); ax.set_ylabel("Test RMSE")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_title("E201: Learning Curves")
fig.savefig(EXP_DIR / "learning_curves.pdf", dpi=150, bbox_inches="tight")
fig.savefig(EXP_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"\nSaved {len(results_df)} results + learning_curves.pdf")
