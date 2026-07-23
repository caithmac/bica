#!/usr/bin/env python3
"""Phase 1.2: Freeze current BindingDB release (July 2026, Kd-only) with provenance."""
import hashlib, json, os, time
from datetime import datetime
import pandas as pd
import numpy as np
from rdkit import Chem

OUT_DIR = "E:/Drug Discovery/projects/balm-revision/data/frozen"
os.makedirs(OUT_DIR, exist_ok=True)

SOURCE = "E:/Drug Discovery/experiments/runs/E105_BindingDB_202607/new_kd_cleaned.csv"
RAW_SOURCE = "E:/Drug Discovery/experiments/runs/E105_BindingDB_202607/bindingdb_raw.tsv"

# --- Load ---
print("[Phase 1.2] Loading E105b Kd-only clean dataset...")
t0 = time.time()
df = pd.read_csv(SOURCE)
load_time = time.time() - t0
print(f"  Loaded {len(df)} rows in {load_time:.1f}s")

# --- Canonical SMILES ---
print("  Canonicalizing SMILES...")
def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

df['smiles_canonical'] = df['smiles'].apply(canonicalize)
valid = df['smiles_canonical'].notna()
df = df[valid].reset_index(drop=True)

# --- Bemis-Murcko scaffolds ---
print("  Computing scaffolds...")
try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaf) if scaf else None
except ImportError:
    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf else None

df['scaffold'] = df['smiles_canonical'].apply(get_scaffold)

# --- Save ---
parquet_path = os.path.join(OUT_DIR, "bindingdb_202607_kd.parquet")
df.to_parquet(parquet_path, index=False)
with open(parquet_path, 'rb') as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

# --- Metadata ---
n_compounds = df['smiles_canonical'].nunique()
n_targets = df['protein_seq'].nunique()
n_scaffolds = df['scaffold'].nunique()
seq_lens = df['protein_seq'].str.len()
pKd_stats = df['pkd'].describe().to_dict()

# Raw source stats
raw_size_gb = os.path.getsize(RAW_SOURCE) / 1e9 if os.path.exists(RAW_SOURCE) else None

metadata = {
    "dataset": "BindingDB July 2026 — Kd-only",
    "source": "bindingdb.org BindingDB_All_202607_tsv.zip",
    "pipeline": "E105b: line-by-line csv parser, Kd-only (no Ki fallback), RDKit canonical, dedup max pKd per (SMILES,protein), protein QC ≤5% non-standard AA, min-10 compounds per target",
    "frozen_at": datetime.utcnow().isoformat() + "Z",
    "sha256": sha256,
    "raw_source_size_gb": round(raw_size_gb, 2) if raw_size_gb else None,
    "raw_rows": 3285265,  # from E105b log
    "rows_after_pipeline": len(df),
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
    "load_time_seconds": round(load_time, 1),
    "parquet_path": parquet_path,
}

with open(os.path.join(OUT_DIR, "bindingdb_202607_kd_metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[Phase 1.2] COMPLETE")
print(f"  Rows: {len(df)}")
print(f"  Compounds: {n_compounds}")
print(f"  Targets: {n_targets}")
print(f"  Scaffolds: {n_scaffolds}")
print(f"  pKd: {pKd_stats.get('mean', '?'):.2f} ± {pKd_stats.get('std', '?'):.2f}")
print(f"  SHA256: {sha256[:16]}...")
print(f"  Saved: {parquet_path}")
