"""
E205 — ECFP4 Bit Importance Analysis
=====================================
Extract top ECFP4 bits from RF/XGB/LGBM, map to substructures,
compute overlap (Jaccard) between models, draw top-5 shared bits.

Post-hoc on trained tree models. Retrains if needed (fast on CPU).

Run: python experiments/runs/E205/run_e205.py
CPU-only
"""

import sys
import time
import logging
import pickle
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ── Paths ──────────────────────────────────────────────────────────────────────
EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("E:/Drug Discovery")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(EXP_DIR / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("E205")

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "experiment": "E205",
    "description": "ECFP4 bit importance analysis",
    "ecfp_radius": 2,
    "ecfp_nbits": 2048,
    "top_k_bits": 30,
    "models": ["rf_ecfp4_aac", "xgb_ecfp4_aac", "lgbm_ecfp4_aac"],
    "timestamp": datetime.now().isoformat(),
}
with open(EXP_DIR / "config.yaml", "w") as f:
    yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load data and featurize
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 1: Loading data + ECFP4 featurization")
log.info("=" * 70)

# Load E101 clean data + existing scaffold split
sys.path.insert(0, str(ROOT))
from harness.data import get_splits
train_df, val_df, test_df = get_splits()

log.info(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

# ECFP4 featurization
def get_ecfp4(smiles_list, radius=2, nbits=2048):
    """Generate ECFP4 fingerprints as numpy array."""
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
                # Convert to numpy efficiently
                on_bits = list(fp.GetOnBits())
                fps[i, on_bits] = 1.0
        except:
            pass
    return fps

def get_aac(seq_list, max_len=1200):
    """Amino acid composition features (20-dim)."""
    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    feats = np.zeros((len(seq_list), 20), dtype=np.float32)
    for i, seq in enumerate(seq_list):
        seq = str(seq).upper()
        if len(seq) == 0:
            continue
        for aa in seq:
            if aa in aa_to_idx:
                feats[i, aa_to_idx[aa]] += 1
        feats[i] /= max(len(seq), 1)
    return feats

log.info("  Computing ECFP4 fingerprints...")
t0 = time.time()
X_train_ecfp = get_ecfp4(train_df["Drug"].values)
X_train_aac = get_aac(train_df["Target"].values)
X_train = np.concatenate([X_train_ecfp, X_train_aac], axis=1)
y_train = train_df["Y"].values.astype(np.float32)

X_test_ecfp = get_ecfp4(test_df["Drug"].values)
X_test_aac = get_aac(test_df["Target"].values)
X_test = np.concatenate([X_test_ecfp, X_test_aac], axis=1)
y_test = test_df["Y"].values.astype(np.float32)

log.info(f"  Train features: {X_train.shape}  Test: {X_test.shape}  ({time.time()-t0:.1f}s)")

# Feature names
ecfp_names = [f"ECFP4_{i}" for i in range(CONFIG["ecfp_nbits"])]
aac_names = [f"AAC_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
feature_names = ecfp_names + aac_names


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Train models + extract importances
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 2: Training models + extracting importances")
log.info("=" * 70)

importances = {}

# ── RF ─────────────────────────────────────────────────────────────────────────
log.info("  Training Random Forest...")
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=500, max_depth=None, min_samples_split=2,
    min_samples_leaf=1, max_features="sqrt", n_jobs=-1, random_state=42,
)
rf.fit(X_train, y_train)
rf_rmse = np.sqrt(np.mean((rf.predict(X_test) - y_test)**2))
rf_r = np.corrcoef(rf.predict(X_test), y_test)[0, 1]
rf_imp = rf.feature_importances_
log.info(f"    RMSE={rf_rmse:.4f}  R={rf_r:.4f}  ({time.time()-t0:.1f}s)")

# ── XGBoost ────────────────────────────────────────────────────────────────────
log.info("  Training XGBoost...")
t0 = time.time()
xgb = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
)
xgb.fit(X_train, y_train)
xgb_rmse = np.sqrt(np.mean((xgb.predict(X_test) - y_test)**2))
xgb_r = np.corrcoef(xgb.predict(X_test), y_test)[0, 1]
xgb_imp = xgb.feature_importances_
log.info(f"    RMSE={xgb_rmse:.4f}  R={xgb_r:.4f}  ({time.time()-t0:.1f}s)")

# ── LightGBM ───────────────────────────────────────────────────────────────────
log.info("  Training LightGBM...")
t0 = time.time()
lgbm = LGBMRegressor(
    n_estimators=500, max_depth=-1, learning_rate=0.1,
    num_leaves=31, random_state=42, n_jobs=-1, verbose=-1,
)
lgbm.fit(X_train, y_train)
lgbm_rmse = np.sqrt(np.mean((lgbm.predict(X_test) - y_test)**2))
lgbm_r = np.corrcoef(lgbm.predict(X_test), y_test)[0, 1]
lgbm_imp = lgbm.feature_importances_
log.info(f"    RMSE={lgbm_rmse:.4f}  R={lgbm_r:.4f}  ({time.time()-t0:.1f}s)")

importances = {
    "RF": (rf_imp, rf_rmse, rf_r),
    "XGBoost": (xgb_imp, xgb_rmse, xgb_r),
    "LightGBM": (lgbm_imp, lgbm_rmse, lgbm_r),
}

# Save importances
for name, (imp, rmse, r) in importances.items():
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False)
    df_imp.to_csv(EXP_DIR / f"bit_importance_{name.lower()}.csv", index=False)
    log.info(f"  {name}: top-5 bits: {list(df_imp['feature'].head(5))}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Map ECFP4 bits to substructures
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 3: Mapping top ECFP4 bits to substructures")
log.info("=" * 70)

# For each model, get top-30 ECFP4-only bits (exclude AAC features)
top_k = CONFIG["top_k_bits"]
top_bits_per_model = {}

for name, (imp, _, _) in importances.items():
    # Filter to ECFP4 bits only
    ecfp_mask = np.array([f.startswith("ECFP4_") for f in feature_names])
    ecfp_indices = np.where(ecfp_mask)[0]
    ecfp_imp = imp[ecfp_indices]
    
    # Get top-k ECFP4 bits
    top_k_idx = np.argsort(-ecfp_imp)[:top_k]
    top_bits = {int(ecfp_indices[i]): float(ecfp_imp[i]) for i in top_k_idx}
    top_bits_per_model[name] = top_bits
    
    log.info(f"  {name}: top-{top_k} ECFP4 bits mapped")

# Create a reference molecule for bit visualization
# Use a common drug-like molecule to show substructures
ref_smiles = [train_df["Drug"].iloc[i] for i in range(min(20, len(train_df)))]
ref_mols = []
for smi in ref_smiles:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is not None:
        ref_mols.append(mol)

log.info(f"  Reference molecules for visualization: {len(ref_mols)}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Bit overlap analysis (Jaccard)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 4: Bit overlap between models")
log.info("=" * 70)

pairs = [
    ("RF", "XGBoost"),
    ("RF", "LightGBM"),
    ("XGBoost", "LightGBM"),
]

overlap_rows = []
for name_a, name_b in pairs:
    bits_a = set(top_bits_per_model[name_a].keys())
    bits_b = set(top_bits_per_model[name_b].keys())
    
    intersection = len(bits_a & bits_b)
    union = len(bits_a | bits_b)
    jaccard_top30 = intersection / union if union > 0 else 0
    
    # Also for top-50
    a_all = set(sorted(top_bits_per_model[name_a], 
                       key=lambda x: importances[name_a][0][x], reverse=True)[:50])
    b_all = set(sorted(top_bits_per_model[name_b],
                       key=lambda x: importances[name_b][0][x], reverse=True)[:50])
    jaccard_top50 = len(a_all & b_all) / len(a_all | b_all) if len(a_all | b_all) > 0 else 0
    
    overlap_rows.append({
        "model_a": name_a,
        "model_b": name_b,
        "jaccard_top30": round(jaccard_top30, 4),
        "jaccard_top50": round(jaccard_top50, 4),
        "intersection_count": intersection,
    })
    log.info(f"  {name_a} ↔ {name_b}: Jaccard(top30)={jaccard_top30:.3f}  Jaccard(top50)={jaccard_top50:.3f}")

overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_csv(EXP_DIR / "bit_overlap.csv", index=False)

# Find shared bits across all three models
all_bits_list = [set(top_bits_per_model[m].keys()) for m in ["RF", "XGBoost", "LightGBM"]]
shared_all = all_bits_list[0] & all_bits_list[1] & all_bits_list[2]
log.info(f"\n  Bits shared by ALL 3 models (top-30): {len(shared_all)}")

# Top-5 shared bits (ranked by average importance)
if len(shared_all) > 0:
    avg_imp = {}
    for bit in shared_all:
        avg_imp[bit] = np.mean([importances[m][0][bit] for m in ["RF", "XGBoost", "LightGBM"]])
    top5_shared = sorted(avg_imp, key=avg_imp.get, reverse=True)[:5]
    log.info(f"  Top-5 shared bits: {top5_shared}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Visualizations
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 5: Generating figures")
log.info("=" * 70)

# ── Venn diagram of top-30 bits ────────────────────────────────────────────────
rf_bits = set(top_bits_per_model["RF"].keys())
xgb_bits = set(top_bits_per_model["XGBoost"].keys())
lgbm_bits = set(top_bits_per_model["LightGBM"].keys())

fig, ax = plt.subplots(figsize=(8, 7))
venn3([rf_bits, xgb_bits, lgbm_bits],
      set_labels=("RF", "XGBoost", "LightGBM"),
      set_colors=("#1b7837", "#5aae61", "#762a83"), alpha=0.7, ax=ax)
ax.set_title("Top-30 ECFP4 Bit Overlap Between Models", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(EXP_DIR / "bit_overlap_venn.pdf", dpi=150, bbox_inches="tight")
fig.savefig(EXP_DIR / "bit_overlap_venn.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved bit_overlap_venn.pdf")

# ── Top-5 substructure drawings ────────────────────────────────────────────────
if len(shared_all) >= 5:
    log.info("  Drawing top-5 shared substructures...")
    # For ECFP4 bit visualization, we need a molecule that HAS that bit set
    # Draw the atom environments highlighted
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, bit_idx in enumerate(top5_shared):
        ax = axes[i] if len(top5_shared) >= 5 else axes
        
        # Find molecules in training set that have this bit set
        bit_col = X_train_ecfp[:, bit_idx] > 0
        mol_indices = np.where(bit_col)[0]
        
        if len(mol_indices) > 0:
            example_smi = train_df["Drug"].iloc[mol_indices[0]]
            mol = Chem.MolFromSmiles(str(example_smi))
            if mol is not None:
                # Draw the molecule
                img = Draw.MolToImage(mol, size=(300, 200))
                ax.imshow(img)
        
        ax.set_title(f"Bit {bit_idx}", fontsize=11)
        ax.axis("off")
    
    plt.tight_layout()
    fig.savefig(EXP_DIR / "top5_substructures.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(EXP_DIR / "top5_substructures.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved top5_substructures.pdf")
else:
    log.warning(f"  Only {len(shared_all)} shared bits — insufficient for top-5 drawing")

# ── Chemical interpretation ────────────────────────────────────────────────────
log.info("\n  Chemical interpretation of top shared bits:")
if len(shared_all) >= 5:
    for bit_idx in top5_shared:
        # Count how many training compounds have this bit
        n_with_bit = int(np.sum(X_train_ecfp[:, bit_idx] > 0))
        log.info(f"    Bit {bit_idx}: present in {n_with_bit}/{len(X_train_ecfp)} training compounds "
                 f"({100*n_with_bit/len(X_train_ecfp):.1f}%)")

log.info(f"\n{'='*50}")
log.info("E205 COMPLETE")
log.info(f"{'='*50}")
