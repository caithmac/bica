"""Naive overlap: BindingDB vs LeakyPDB — exact SMILES and scaffold level."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path("E:/Drug Discovery")
sys.path.insert(0, str(ROOT))

# ── Load BindingDB clean + splits ──────────────────────────────────────────────
bd = pd.read_csv(ROOT / "experiments/runs/E101/bindingdb_clean.csv")
bd.columns = ["smiles", "protein_seq", "pkd"]
print(f"BindingDB clean: {len(bd):,} rows")

from harness.data import get_splits
train_df, val_df, test_df = get_splits()
print(f"Split sizes: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

# ── Load LeakyPDB ──────────────────────────────────────────────────────────────
from harness.data import load_leakypdb_raw
lp = load_leakypdb_raw()
print(f"LeakyPDB: {len(lp):,} rows")

# ── Helpers ─────────────────────────────────────────────────────────────────────
def canon(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else None
    except:
        return None

# ── Build LeakyPDB sets ─────────────────────────────────────────────────────────
lp_smiles = set()
lp_scaffolds = set()
for smi in lp["Drug"]:
    c = canon(smi)
    if c:
        lp_smiles.add(c)
        s = get_scaffold(c)
        if s:
            lp_scaffolds.add(s)
print(f"\nLeakyPDB: {len(lp_smiles):,} unique SMILES, {len(lp_scaffolds):,} unique scaffolds")

# ── Per-split overlap ──────────────────────────────────────────────────────────
for split_name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
    total = len(df)
    smi_match = 0
    scaffold_match = 0
    for smi in df["Drug"]:
        c = canon(smi)
        if c and c in lp_smiles:
            smi_match += 1
        if c:
            s = get_scaffold(c)
            if s and s in lp_scaffolds:
                scaffold_match += 1
    pct_smi = 100 * smi_match / total
    pct_scf = 100 * scaffold_match / total
    print(f"\n{split_name} ({total:,} compounds):")
    print(f"  Exact SMILES in LeakyPDB:   {smi_match:5d}  ({pct_smi:.2f} pct)")
    print(f"  Same scaffold in LeakyPDB:  {scaffold_match:5d}  ({pct_scf:.2f} pct)")

# ── Global overlap (full BindingDB) ────────────────────────────────────────────
all_smiles = set()
all_scaffolds = set()
for smi in bd["smiles"]:
    c = canon(smi)
    if c:
        all_smiles.add(c)
        s = get_scaffold(c)
        if s:
            all_scaffolds.add(s)

smi_overlap = all_smiles & lp_smiles
scf_overlap = all_scaffolds & lp_scaffolds
pct_smi = 100 * len(smi_overlap) / len(all_smiles)
pct_scf = 100 * len(scf_overlap) / len(all_scaffolds)

print(f"\n=== GLOBAL OVERLAP ===")
print(f"BindingDB total unique SMILES:     {len(all_smiles):,}")
print(f"BindingDB total unique scaffolds:  {len(all_scaffolds):,}")
print(f"Overlap SMILES:   {len(smi_overlap):,}  ({pct_smi:.2f} pct of BD)")
print(f"Overlap scaffolds: {len(scf_overlap):,}  ({pct_scf:.2f} pct of BD)")
print(f"Overlap SMILES as pct of LP:       {100*len(smi_overlap)/len(lp_smiles):.2f} pct")

# ── Test set leakage detail ────────────────────────────────────────────────────
print(f"\n=== TEST SET LEAKAGE DETAIL ===")
leaked = []
for _, row in test_df.iterrows():
    c = canon(row["Drug"])
    if c and c in lp_smiles:
        leaked.append((c, row["Target"][:50], row["Y"]))
print(f"Test compounds with exact SMILES match in LeakyPDB: {len(leaked)}")
for smi, target, pkd in leaked[:15]:
    print(f"  {smi[:55]:55s}  {target:45s}  pKd={pkd:.2f}")
if len(leaked) > 15:
    print(f"  ... and {len(leaked)-15} more")
