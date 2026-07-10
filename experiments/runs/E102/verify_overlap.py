"""Double-check: print actual overlapping SMILES, verify counts manually."""
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

print(f"BindingDB raw rows: {len(bd):,}")
print(f"BindingDB cols: {list(bd.columns)}")
print(f"LeakyPDB raw rows: {len(lp):,}")
print(f"LeakyPDB cols: {list(lp.columns)}")

# Count invalid SMILES
bd_invalid = 0
bd_good = 0
bd_smiles = set()
for smi in bd["Drug"]:
    c = canon(smi)
    if c:
        bd_smiles.add(c)
        bd_good += 1
    else:
        bd_invalid += 1

lp_invalid = 0
lp_good = 0
lp_smiles = set()
for smi in lp["Drug"]:
    c = canon(smi)
    if c:
        lp_smiles.add(c)
        lp_good += 1
    else:
        lp_invalid += 1

print(f"\nBD: {bd_good:,} valid, {bd_invalid:,} invalid SMILES ({len(bd_smiles):,} unique)")
print(f"LP: {lp_good:,} valid, {lp_invalid:,} invalid SMILES ({len(lp_smiles):,} unique)")

both = sorted(bd_smiles & lp_smiles)
print(f"\nOverlap: {len(both)} SMILES")
print("First 20:")
for s in both[:20]:
    print(f"  {s}")

# Also check: are these just trivial molecules?
print("\nLength distribution of overlapping SMILES:")
from collections import Counter
lens = Counter(len(s) for s in both)
for l, c in sorted(lens.items()):
    print(f"  len={l}: {c}")

# Average length
avg_len = sum(len(s) for s in both) / len(both)
print(f"  Average length: {avg_len:.0f} chars")
