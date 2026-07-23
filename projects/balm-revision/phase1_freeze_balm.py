#!/usr/bin/env python3
"""Phase 1.1: Freeze BALM BindingDB_filtered dataset with full provenance metadata."""
import hashlib, json, os, sys, time
from datetime import datetime
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from datasets import load_dataset

OUT_DIR = "E:/Drug Discovery/projects/balm-revision/data/frozen"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load BALM dataset ---
print("[Phase 1.1] Loading BALM/BALM-benchmark BindingDB_filtered...")
t0 = time.time()
ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered", split="train")
df = ds.to_pandas()
load_time = time.time() - t0

# --- Apply BALM preprocessing (max Y per drug-target pair) ---
if 'Drug_ID' in df.columns and 'Target_ID' in df.columns:
    n_before = len(df)
    df = df.groupby(['Drug_ID', 'Drug', 'Target_ID', 'Target'])['Y'].agg('max').reset_index()
    print(f"  BALM aggregation: {n_before} -> {len(df)} rows")

df = df.dropna(subset=['Drug', 'Target', 'Y']).reset_index(drop=True)

# --- Compute canonical SMILES ---
print("  Canonicalizing SMILES...")
def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

df['Drug_canonical'] = df['Drug'].apply(canonicalize)
valid = df['Drug_canonical'].notna()
df = df[valid].reset_index(drop=True)

# --- Compute Bemis-Murcko scaffolds ---
print("  Computing Bemis-Murcko scaffolds...")
try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    def murcko_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaf) if scaf else None
except ImportError:
    def murcko_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf else None

df['scaffold'] = df['Drug_canonical'].apply(murcko_scaffold)

# --- Compute SHA256 ---
print("  Computing SHA256...")
parquet_path = os.path.join(OUT_DIR, "balm_filtered.parquet")
df.to_parquet(parquet_path, index=False)
with open(parquet_path, 'rb') as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

# --- Metadata ---
n_compounds = df['Drug_canonical'].nunique()
n_targets = df['Target'].nunique()
n_scaffolds = df['scaffold'].nunique()
n_pairs = len(df)

# Get protein length stats
seq_lens = df['Target'].str.len()
pKd_stats = df['Y'].describe().to_dict()

metadata = {
    "dataset": "BALM BindingDB_filtered",
    "source": "BALM/BALM-benchmark on HuggingFace",
    "config": "BindingDB_filtered",
    "frozen_at": datetime.utcnow().isoformat() + "Z",
    "sha256": sha256,
    "rows": n_pairs,
    "unique_compounds": n_compounds,
    "unique_targets": n_targets,
    "unique_scaffolds": n_scaffolds,
    "columns": list(df.columns),
    "protein_sequence_stats": {
        "min_len": int(seq_lens.min()),
        "max_len": int(seq_lens.max()),
        "mean_len": float(seq_lens.mean()),
        "median_len": float(seq_lens.median()),
    },
    "pKd_stats": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in pKd_stats.items()},
    "preprocessing": "groupby Drug_ID+Target_ID max Y, RDKit canonical SMILES, drop invalid",
    "load_time_seconds": round(load_time, 1),
    "parquet_path": parquet_path,
}

with open(os.path.join(OUT_DIR, "balm_filtered_metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[Phase 1.1] COMPLETE")
print(f"  Rows: {n_pairs}")
print(f"  Compounds: {n_compounds}")
print(f"  Targets: {n_targets}")
print(f"  Scaffolds: {n_scaffolds}")
print(f"  pKd: {pKd_stats.get('mean', '?'):.2f} ± {pKd_stats.get('std', '?'):.2f}")
print(f"  SHA256: {sha256[:16]}...")
print(f"  Saved: {parquet_path}")
