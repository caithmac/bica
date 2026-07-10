"""Just count: how many unique SMILES appear in BOTH BindingDB and LeakyPDB."""
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem

ROOT = Path("E:/Drug Discovery")
sys.path.insert(0, str(ROOT))

def canon(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

from harness.data import load_raw, load_leakypdb_raw

bd = load_raw()
lp = load_leakypdb_raw()

bd_smiles = set()
for smi in bd["Drug"]:
    c = canon(smi)
    if c:
        bd_smiles.add(c)

lp_smiles = set()
for smi in lp["Drug"]:
    c = canon(smi)
    if c:
        lp_smiles.add(c)

both = bd_smiles & lp_smiles

print(f"BindingDB unique SMILES:  {len(bd_smiles):,}")
print(f"LeakyPDB unique SMILES:   {len(lp_smiles):,}")
print(f"In BOTH:                  {len(both):,}")
print(f"  as pct of BD: {100*len(both)/len(bd_smiles):.1f}%")
print(f"  as pct of LP: {100*len(both)/len(lp_smiles):.1f}%")
