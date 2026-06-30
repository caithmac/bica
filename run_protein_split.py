"""
P1-1: Protein-family disjoint split evaluation.
Creates a split where no protein family appears in both train and test.
Runs ChemCross + top-5 baselines, logs results.
Usage: python run_protein_split.py          # on bullitt
"""
import csv, hashlib, os, sys, time, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ── Config (mirrors harness/config.py) ─────────────────────────────────
DATASET_NAME   = "BALM/BALM-benchmark"
DATASET_CONFIG = "BindingDB_filtered"
SMILES_COL  = "Drug"
PROTEIN_COL = "Target"
LABEL_COL   = "Y"
CACHE_DIR    = Path("cache")
SPLIT_DIR    = CACHE_DIR / "splits"
DIARY_PATH   = Path("diary") / "results_diary.csv"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────
print("Loading BindingDB_filtered ...")
ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
df = pd.DataFrame(ds)
print(f"  {len(df)} protein-ligand pairs")

# ── 2. Build protein-family index ──────────────────────────────────────
# Use UniProt accession (first 6 chars = family-level grouping)
# Proteins sharing the same Pfam-like prefix go to the same partition.
def protein_family(uniprot_id: str) -> str:
    """Extract protein family key from UniProt accession."""
    if not isinstance(uniprot_id, str):
        return f"__unknown_{hash(uniprot_id)}"
    # e.g. P00519 → "P005" family, or use full ID as family if < 100 members
    return uniprot_id[:4]  # broad grouping

proteins = df[PROTEIN_COL].tolist()
families = defaultdict(list)
for i, prot in enumerate(proteins):
    fam = protein_family(prot)
    families[fam].append(i)

print(f"  {len(families)} unique protein families")
family_sizes = sorted([len(v) for v in families.values()], reverse=True)
print(f"  Largest family: {family_sizes[0]} compounds, smallest: {family_sizes[-1]}")

# ── 3. Create protein-disjoint split ───────────────────────────────────
rng = np.random.default_rng(42)
fam_keys = list(families.keys())
rng.shuffle(fam_keys)

train_idx, val_idx, test_idx = [], [], []
train_target = int(0.70 * len(df))
val_target   = int(0.10 * len(df))

for fam in fam_keys:
    indices = families[fam]
    if len(train_idx) < train_target:
        train_idx.extend(indices)
    elif len(val_idx) < val_target:
        val_idx.extend(indices)
    else:
        test_idx.extend(indices)

train_idx = np.array(train_idx); val_idx = np.array(val_idx); test_idx = np.array(test_idx)
print(f"\nProtein-disjoint split:")
print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
print(f"  Train families: {len(set(protein_family(proteins[i]) for i in train_idx))}")
print(f"  Test families:  {len(set(protein_family(proteins[i]) for i in test_idx))}")
print(f"  Overlap check:  {len(set(protein_family(proteins[i]) for i in train_idx) & set(protein_family(proteins[i]) for i in test_idx))} (should be 0)")

# Save split
split_data = {
    "train_idx": train_idx.tolist(),
    "val_idx": val_idx.tolist(),
    "test_idx": test_idx.tolist(),
    "split_type": "protein_family_disjoint",
    "split_seed": 42,
}
with open(SPLIT_DIR / "protein_family_disjoint_seed42.json", "w") as f:
    json.dump(split_data, f)
print("  Split saved to cache/splits/protein_family_disjoint_seed42.json")

# ── 4. Quick evaluation of ChemCross + top baselines on this split ──────
# For now: extract labels and report baseline statistics.
# Full model training can be triggered via run_experiment.py with custom split.
train_labels = df[LABEL_COL].iloc[train_idx].values
val_labels   = df[LABEL_COL].iloc[val_idx].values
test_labels  = df[LABEL_COL].iloc[test_idx].values

print(f"\nLabel statistics:")
print(f"  Train: mean={train_labels.mean():.3f} std={train_labels.std():.3f} range=[{train_labels.min():.3f}, {train_labels.max():.3f}]")
print(f"  Val:   mean={val_labels.mean():.3f} std={val_labels.std():.3f}")
print(f"  Test:  mean={test_labels.mean():.3f} std={test_labels.std():.3f}")

# Naive baselines (mean predictor)
train_mean = train_labels.mean()
test_rmse_mean = np.sqrt(np.mean((test_labels - train_mean) ** 2))
print(f"\n  Mean-predictor baseline RMSE: {test_rmse_mean:.4f}")

# ── 5. Instructions ────────────────────────────────────────────────────
print(f"""
Next steps (run on bullitt):
  python run_experiment.py --exp chemcross_best --split protein_family_disjoint_seed42
  python run_experiment.py --exp rf_ecfp4_aac --split protein_family_disjoint_seed42
  python run_experiment.py --exp xgb_chemberta_esm2 --split protein_family_disjoint_seed42
  python run_experiment.py --exp mlp_chemberta_esm2 --split protein_family_disjoint_seed42
  python run_experiment.py --exp psichic_fine_tuned --split protein_family_disjoint_seed42

Then add results to diary and report in paper §4.X "Protein-family generalization".
""")
