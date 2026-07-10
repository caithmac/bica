"""Cross-dataset leakage: LP compounds found in BD train."""
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path("E:/Drug Discovery")
sys.path.insert(0, str(ROOT))

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

from harness.data import get_splits, load_leakypdb_raw

# BindingDB TRAIN only (what a BD-trained model sees)
train_df, _, _ = get_splits()
bd_train_smiles = set()
bd_train_scaffolds = set()
for smi in train_df["Drug"]:
    c = canon(smi)
    if c:
        bd_train_smiles.add(c)
        s = get_scaffold(c)
        if s:
            bd_train_scaffolds.add(s)

print(f"BindingDB TRAIN unique SMILES:     {len(bd_train_smiles):,}")
print(f"BindingDB TRAIN unique scaffolds:  {len(bd_train_scaffolds):,}")

# LeakyPDB
lp = load_leakypdb_raw()
lp_smiles = set()
lp_scaffolds = set()
for smi in lp["Drug"]:
    c = canon(smi)
    if c:
        lp_smiles.add(c)
        s = get_scaffold(c)
        if s:
            lp_scaffolds.add(s)

print(f"LeakyPDB unique SMILES:     {len(lp_smiles):,}")
print(f"LeakyPDB unique scaffolds:  {len(lp_scaffolds):,}")

# Unique-level overlap
smi_ov = lp_smiles & bd_train_smiles
scf_ov = lp_scaffolds & bd_train_scaffolds
print()
print("=== LeakyPDB compounds already in BD TRAIN (unique) ===")
print(f"SMILES:   {len(smi_ov):,} / {len(lp_smiles):,}  ({100*len(smi_ov)/len(lp_smiles):.1f} pct of LP)")
print(f"Scaffolds: {len(scf_ov):,} / {len(lp_scaffolds):,}  ({100*len(scf_ov)/len(lp_scaffolds):.1f} pct of LP)")

# Row-level (pair level): LP rows where compound is in BD train
lp_total = len(lp)
lp_smi_hit = 0
lp_scf_hit = 0
for _, row in lp.iterrows():
    c = canon(row["Drug"])
    if c and c in bd_train_smiles:
        lp_smi_hit += 1
    if c:
        s = get_scaffold(c)
        if s and s in bd_train_scaffolds:
            lp_scf_hit += 1

print()
print("=== LeakyPDB rows found in BD TRAIN (pair level) ===")
print(f"Total LP rows: {lp_total:,}")
print(f"SMILES in BD train:   {lp_smi_hit:,}  ({100*lp_smi_hit/lp_total:.1f} pct)")
print(f"Scaffold in BD train: {lp_scf_hit:,}  ({100*lp_scf_hit/lp_total:.1f} pct)")
