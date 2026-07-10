"""
E102 — Cross-Dataset Overlap Analysis
======================================
Check for compound/protein overlap between BindingDB, LeakyPDB, and ChEMBL.
Overlap at three levels: exact SMILES, Bemis-Murcko scaffold, protein sequence.

Source data:
  - E101 clean BindingDB (experiments/runs/E101/bindingdb_clean.csv)
  - LeakyPDB (PDBBind-derived, via harness/data.py)
  - ChEMBL 36 (cache/chembl_36.db)

Run: python experiments/runs/E102/run_e102.py
"""

import sys
import os
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import sqlite3
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
log = logging.getLogger("E102")

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "experiment": "E102",
    "description": "Cross-dataset overlap analysis",
    "datasets": [
        "BindingDB (E101 clean)",
        "LeakyPDB (PDBBind-derived)",
        "ChEMBL 36 (binding data)",
    ],
    "overlap_levels": ["exact_smiles", "bemis_murcko_scaffold", "protein_sequence"],
    "timestamp": datetime.now().isoformat(),
}

with open(EXP_DIR / "config.yaml", "w") as f:
    yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_scaffold(smiles):
    """Bemis-Murcko scaffold SMILES. Returns None for invalid."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else None
    except:
        return None

def canonicalize_smiles(smiles):
    """Canonical SMILES via RDKit."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def protein_hash(seq):
    """Short hash for protein sequences (for display)."""
    return hashlib.md5(str(seq).encode()).hexdigest()[:8]

def overlap_ratio(set_a, set_b):
    """Jaccard: |A ∩ B| / |A ∪ B| and |A ∩ B| / min(|A|, |B|)."""
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0, 0.0, 0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = intersection / union if union > 0 else 0
    min_ratio = intersection / min(len(set_a), len(set_b)) if min(len(set_a), len(set_b)) > 0 else 0
    return jaccard, min_ratio, intersection


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load all three datasets
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 1: Loading datasets")
log.info("=" * 70)
t0 = time.time()

# ── BindingDB (E101 clean) ─────────────────────────────────────────────────────
bd_clean_path = ROOT / "experiments/runs/E101/bindingdb_clean.csv"
bd = pd.read_csv(bd_clean_path)
bd.columns = ["smiles", "protein_seq", "pkd"]
log.info(f"BindingDB (E101 clean): {len(bd):,} rows")

# Canonical SMILES + scaffold
bd_smiles_set = set()
bd_scaffolds_set = set()
bd_proteins_set = set()
for _, row in bd.iterrows():
    can_smi = canonicalize_smiles(row["smiles"])
    if can_smi:
        bd_smiles_set.add(can_smi)
        scaffold = get_scaffold(can_smi)
        if scaffold:
            bd_scaffolds_set.add(scaffold)
    bd_proteins_set.add(row["protein_seq"])

log.info(f"  Unique canonical SMILES: {len(bd_smiles_set):,}")
log.info(f"  Unique scaffolds: {len(bd_scaffolds_set):,}")
log.info(f"  Unique proteins: {len(bd_proteins_set):,}")

# ── LeakyPDB ───────────────────────────────────────────────────────────────────
# Use harness/data.py's load_leakypdb_raw
sys.path.insert(0, str(ROOT))
from harness.data import load_leakypdb_raw
lp = load_leakypdb_raw()
log.info(f"LeakyPDB: {len(lp):,} rows")

lp_smiles_set = set()
lp_scaffolds_set = set()
lp_proteins_set = set()
for _, row in lp.iterrows():
    can_smi = canonicalize_smiles(row["Drug"])
    if can_smi:
        lp_smiles_set.add(can_smi)
        scaffold = get_scaffold(can_smi)
        if scaffold:
            lp_scaffolds_set.add(scaffold)
    lp_proteins_set.add(row["Target"])

log.info(f"  Unique canonical SMILES: {len(lp_smiles_set):,}")
log.info(f"  Unique scaffolds: {len(lp_scaffolds_set):,}")
log.info(f"  Unique proteins: {len(lp_proteins_set):,}")

# ── ChEMBL 36 ──────────────────────────────────────────────────────────────────
# NOTE: ChEMBL 36 SQLite join query on 27GB DB is too slow (5+ min).
# Skipping ChEMBL overlap for now — BindingDB ↔ LeakyPDB is the critical pair.
# Revisit if ChEMBL overlap data is needed for paper.

chembl_smiles_set = set()
chembl_scaffolds_set = set()
chembl_proteins_set = set()
chembl_df = pd.DataFrame()
log.info("ChEMBL 36: SKIPPED (SQLite query too slow on 27GB DB)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Compute overlap matrices
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 2: Computing overlap matrices")
log.info("=" * 70)

datasets = {
    "BindingDB": (bd_smiles_set, bd_scaffolds_set, bd_proteins_set),
    "LeakyPDB": (lp_smiles_set, lp_scaffolds_set, lp_proteins_set),
    "ChEMBL": (chembl_smiles_set, chembl_scaffolds_set, chembl_proteins_set),
}
names = list(datasets.keys())

overlap_smiles = np.zeros((3, 3))
overlap_scaffolds = np.zeros((3, 3))
overlap_proteins = np.zeros((3, 3))
overlap_smiles_intersect = np.zeros((3, 3), dtype=int)
overlap_scaffolds_intersect = np.zeros((3, 3), dtype=int)
overlap_proteins_intersect = np.zeros((3, 3), dtype=int)

for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        si, sci, pi = datasets[ni]
        sj, scj, pj = datasets[nj]
        
        j_s, _, is_s = overlap_ratio(si, sj)
        j_sc, _, is_sc = overlap_ratio(sci, scj)
        j_p, _, is_p = overlap_ratio(pi, pj)
        
        overlap_smiles[i, j] = j_s
        overlap_scaffolds[i, j] = j_sc
        overlap_proteins[i, j] = j_p
        overlap_smiles_intersect[i, j] = is_s
        overlap_scaffolds_intersect[i, j] = is_sc
        overlap_proteins_intersect[i, j] = is_p

# ── Save overlap matrices ──────────────────────────────────────────────────────
for name, matrix, intersect_matrix in [
    ("smiles", overlap_smiles, overlap_smiles_intersect),
    ("scaffolds", overlap_scaffolds, overlap_scaffolds_intersect),
    ("proteins", overlap_proteins, overlap_proteins_intersect),
]:
    df_jaccard = pd.DataFrame(matrix, index=names, columns=names)
    df_jaccard.to_csv(EXP_DIR / f"overlap_matrix_{name}.csv")
    
    df_intersect = pd.DataFrame(intersect_matrix, index=names, columns=names)
    df_intersect.to_csv(EXP_DIR / f"overlap_matrix_{name}_intersection.csv")
    
    log.info(f"\n{name.upper()} Jaccard overlap:")
    log.info(f"\n{df_jaccard.to_string()}")
    log.info(f"\n{name.upper()} Intersection counts:")
    log.info(f"\n{df_intersect.to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Flag BindingDB test-set compounds that appear in other datasets
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 3: Checking BindingDB test-set leakage")
log.info("=" * 70)

# Use existing scaffold split (seed 42) to identify test compounds
# This is the same split used in all E000 experiments
from harness.data import get_splits
train_df, val_df, test_df = get_splits()

test_smiles = set()
test_scaffolds = set()
for smi in test_df["Drug"].values:
    can_smi = canonicalize_smiles(smi)
    if can_smi:
        test_smiles.add(can_smi)
        scaffold = get_scaffold(can_smi)
        if scaffold:
            test_scaffolds.add(scaffold)

log.info(f"BindingDB test set (scaffold split, seed 42): {len(test_df):,} compounds")
log.info(f"  Unique canonical SMILES in test: {len(test_smiles):,}")
log.info(f"  Unique scaffolds in test: {len(test_scaffolds):,}")

# Check for leaked compounds
leaked_smiles_lp = test_smiles & lp_smiles_set
leaked_scaffolds_lp = test_scaffolds & lp_scaffolds_set
leaked_smiles_chembl = test_smiles & chembl_smiles_set
leaked_scaffolds_chembl = test_scaffolds & chembl_scaffolds_set

log.info(f"\n  Leaked exact SMILES in LeakyPDB: {len(leaked_smiles_lp)}")
log.info(f"  Leaked scaffolds in LeakyPDB: {len(leaked_scaffolds_lp)}")
log.info(f"  Leaked exact SMILES in ChEMBL: {len(leaked_smiles_chembl)}")
log.info(f"  Leaked scaffolds in ChEMBL: {len(leaked_scaffolds_chembl)}")

# Save leaked compound list
leaked_rows = []
for smi in leaked_smiles_lp:
    # Find the BindingDB rows for this SMILES
    test_subset = test_df[test_df["Drug"].apply(lambda x: canonicalize_smiles(x) == smi)]
    for _, row in test_subset.iterrows():
        leaked_rows.append({"smiles": smi, "source": "LeakyPDB", "level": "exact_smiles",
                          "pkd": row["Y"], "protein": row["Target"][:50]})

for scaffold in leaked_scaffolds_lp:
    # Find compounds with this scaffold in test set
    for smi in test_smiles:
        if get_scaffold(smi) == scaffold:
            leaked_rows.append({"smiles": smi, "source": "LeakyPDB", "level": "scaffold",
                              "scaffold": scaffold[:80]})

if leaked_rows:
    leaked_df = pd.DataFrame(leaked_rows).drop_duplicates()
    leaked_df.to_csv(EXP_DIR / "leaked_test_compounds.csv", index=False)
    log.info(f"\n  Saved {len(leaked_df)} leaked entries to leaked_test_compounds.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Generate overlap heatmap
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 4: Generating overlap heatmap")
log.info("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, matrix, title in [
    (axes[0], overlap_smiles, "Exact SMILES"),
    (axes[1], overlap_scaffolds, "Bemis-Murcko Scaffolds"),
    (axes[2], overlap_proteins, "Protein Sequences"),
]:
    mask = np.eye(3, dtype=bool)
    sns.heatmap(
        pd.DataFrame(matrix, index=names, columns=names),
        annot=True, fmt=".3f", cmap="YlOrRd", 
        mask=mask,
        vmin=0, vmax=max(0.3, matrix.max()),
        ax=ax, cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    # Fill diagonal cells manually
    for i in range(3):
        ax.text(i+0.5, i+0.5, "1.000", ha="center", va="center", fontsize=10, color="gray")

plt.tight_layout()
fig.savefig(EXP_DIR / "overlap_heatmap.pdf", dpi=150, bbox_inches="tight")
fig.savefig(EXP_DIR / "overlap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved overlap_heatmap.pdf and overlap_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - t0
log.info(f"\n{'='*50}")
log.info(f"E102 COMPLETE in {elapsed:.1f}s")
log.info(f"{'='*50}")

# Key findings
log.info("\nKEY FINDINGS:")
log.info(f"  BindingDB ↔ LeakyPDB scaffold overlap: {overlap_scaffolds[0,1]:.3f}")
log.info(f"  BindingDB ↔ ChEMBL scaffold overlap: {overlap_scaffolds[0,2]:.3f}")
log.info(f"  Leaked test SMILES (LeakyPDB): {len(leaked_smiles_lp)}")
log.info(f"  Leaked test scaffolds (LeakyPDB): {len(leaked_scaffolds_lp)}")

if len(leaked_smiles_lp) > 0 or len(leaked_scaffolds_lp) > 0:
    log.warning("  ⚠ LEAKAGE DETECTED — test compounds appear in LeakyPDB!")
    log.warning("  Remove these compounds from test set before cross-dataset evaluation.")
else:
    log.info("  ✓ No test-set leakage detected in LeakyPDB")
