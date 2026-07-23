#!/usr/bin/env python3
"""Phase 1.3: Overlap analysis — BALM old vs BindingDB new, plus LeakyPDB."""
import json, os, time
from datetime import datetime
import pandas as pd
import numpy as np
from datasets import load_dataset

OUT_DIR = "E:/Drug Discovery/projects/balm-revision/data/"
os.makedirs(OUT_DIR, exist_ok=True)

print("[Phase 1.3] Loading frozen datasets...")
t0 = time.time()

# Load frozen datasets
old = pd.read_parquet(f"{OUT_DIR}frozen/balm_filtered.parquet")
new = pd.read_parquet(f"{OUT_DIR}frozen/bindingdb_202607_kd.parquet")

print(f"  OLD (BALM): {len(old)} rows, {old['Drug_canonical'].nunique()} compounds, {old['Target'].nunique()} targets")
print(f"  NEW (BindingDB 2026/07): {len(new)} rows, {new['smiles_canonical'].nunique()} compounds, {new['protein_seq'].nunique()} targets")

# Load LeakyPDB
print("  Loading LeakyPDB from BALM benchmark...")
lp = load_dataset("BALM/BALM-benchmark", "LeakyPDB", split="train").to_pandas()
# Canonicalize LeakyPDB SMILES
from rdkit import Chem
def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

lp['Drug_canonical'] = lp['Drug'].apply(canonicalize)
lp = lp[lp['Drug_canonical'].notna()].reset_index(drop=True)

# Scaffolds on LeakyPDB
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

lp['scaffold'] = lp['Drug_canonical'].apply(get_scaffold)
print(f"  LeakyPDB: {len(lp)} rows, {lp['Drug_canonical'].nunique()} compounds, {lp['Target'].nunique()} targets")

# --- Overlap analysis at PAIR level ---
print("\n--- Overlap Analysis (PAIR level) ---")

def overlap_report(name_a, df_a, smiles_a, target_a, scaffold_a,
                   name_b, df_b, smiles_b, target_b, scaffold_b):
    """Compute overlap percentages at the pair (row) level."""
    
    # Exact SMILES overlap
    smiles_a_set = set(df_a[smiles_a])
    smiles_b_set = set(df_b[smiles_b])
    exact_overlap_smiles = smiles_a_set & smiles_b_set
    
    a_rows_in_b = df_a[smiles_a].isin(exact_overlap_smiles).sum()
    b_rows_in_a = df_b[smiles_b].isin(exact_overlap_smiles).sum()
    
    # Scaffold overlap
    scaff_a_set = set(df_a[scaffold_a].dropna())
    scaff_b_set = set(df_b[scaffold_b].dropna())
    scaff_overlap = scaff_a_set & scaff_b_set
    
    a_scaff_rows_in_b = df_a[scaffold_a].isin(scaff_overlap).sum()
    b_scaff_rows_in_a = df_b[scaffold_b].isin(scaff_overlap).sum()
    
    # Protein sequence overlap
    prot_a_set = set(df_a[target_a].dropna())
    prot_b_set = set(df_b[target_b].dropna())
    prot_overlap = prot_a_set & prot_b_set
    
    a_prot_rows_in_b = df_a[target_a].isin(prot_overlap).sum()
    b_prot_rows_in_a = df_b[target_b].isin(prot_overlap).sum()
    
    report = {
        f"{name_a} → {name_b}": {
            "exact_smiles": {
                "compounds_in_overlap": len(exact_overlap_smiles),
                f"rows_in_{name_a.lower()}_that_share": int(a_rows_in_b),
                f"pct_of_{name_a.lower()}": round(100 * a_rows_in_b / len(df_a), 1),
                f"rows_in_{name_b.lower()}_that_share": int(b_rows_in_a),
                f"pct_of_{name_b.lower()}": round(100 * b_rows_in_a / len(df_b), 1),
            },
            "scaffold": {
                "scaffolds_in_overlap": len(scaff_overlap),
                f"rows_in_{name_a.lower()}_that_share": int(a_scaff_rows_in_b),
                f"pct_of_{name_a.lower()}": round(100 * a_scaff_rows_in_b / len(df_a), 1),
                f"rows_in_{name_b.lower()}_that_share": int(b_scaff_rows_in_a),
                f"pct_of_{name_b.lower()}": round(100 * b_scaff_rows_in_a / len(df_b), 1),
            },
            "protein_sequence": {
                "proteins_in_overlap": len(prot_overlap),
                f"rows_in_{name_a.lower()}_that_share": int(a_prot_rows_in_b),
                f"pct_of_{name_a.lower()}": round(100 * a_prot_rows_in_b / len(df_a), 1),
                f"rows_in_{name_b.lower()}_that_share": int(b_prot_rows_in_a),
                f"pct_of_{name_b.lower()}": round(100 * b_prot_rows_in_a / len(df_b), 1),
            },
        }
    }
    return report

# OLD vs NEW
results = {}
results.update(overlap_report("OLD", old, 'Drug_canonical', 'Target', 'scaffold',
                                "NEW", new, 'smiles_canonical', 'protein_seq', 'scaffold'))

# OLD vs LeakyPDB
results.update(overlap_report("OLD", old, 'Drug_canonical', 'Target', 'scaffold',
                                "LeakyPDB", lp, 'Drug_canonical', 'Target', 'scaffold'))

# NEW vs LeakyPDB
results.update(overlap_report("NEW", new, 'smiles_canonical', 'protein_seq', 'scaffold',
                                "LeakyPDB", lp, 'Drug_canonical', 'Target', 'scaffold'))

# --- Create NEW delta (compounds in new not in old) ---
print("\n--- Creating NEW BindingDB delta ---")
old_smiles = set(old['Drug_canonical'])
old_pairs = set(zip(old['Drug_canonical'], old['Target']))

new['in_old_by_smiles'] = new['smiles_canonical'].isin(old_smiles)
new['in_old_by_pair'] = new.apply(lambda r: (r['smiles_canonical'], r['protein_seq']) in old_pairs, axis=1)

delta = new[~new['in_old_by_smiles']].copy()
delta_pairs = new[~new['in_old_by_pair']].copy()

delta_path = f"{OUT_DIR}frozen/bindingdb_delta_new_compounds.parquet"
delta_pairs_path = f"{OUT_DIR}frozen/bindingdb_delta_new_pairs.parquet"
delta.to_parquet(delta_path, index=False)
delta_pairs.to_parquet(delta_pairs_path, index=False)

results["delta"] = {
    "total_new_rows": len(new),
    "rows_with_new_compounds": int((~new['in_old_by_smiles']).sum()),
    "pct_new_compounds": round(100 * (~new['in_old_by_smiles']).sum() / len(new), 1),
    "rows_with_new_pairs": int((~new['in_old_by_pair']).sum()),
    "pct_new_pairs": round(100 * (~new['in_old_by_pair']).sum() / len(new), 1),
    "new_compound_delta_path": delta_path,
    "new_pair_delta_path": delta_pairs_path,
}

# --- Save ---
results["_metadata"] = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "runtime_seconds": round(time.time() - t0, 1),
    "datasets": {
        "old_balm": {"rows": len(old), "compounds": old['Drug_canonical'].nunique(), "targets": old['Target'].nunique()},
        "new_bindingdb": {"rows": len(new), "compounds": new['smiles_canonical'].nunique(), "targets": new['protein_seq'].nunique()},
        "leakypdb": {"rows": len(lp), "compounds": lp['Drug_canonical'].nunique(), "targets": lp['Target'].nunique()},
    },
    "methodology": "PAIR-level overlap (not unique compound). A compound appearing 100 times against 100 targets counts 100 times."
}

with open(f"{OUT_DIR}overlap_analysis.json", 'w') as f:
    json.dump(results, f, indent=2)

# --- Print summary ---
print("\n" + "="*70)
print("OVERLAP SUMMARY")
print("="*70)

for key in ["OLD → NEW", "OLD → LeakyPDB", "NEW → LeakyPDB"]:
    for k, v in results.items():
        if key in k:
            print(f"\n{k}:")
            for level in ["exact_smiles", "scaffold", "protein_sequence"]:
                d = v[level]
                # Find the percentage keys
                pct_keys = [k for k in d if k.startswith('pct_')]
                for pk in pct_keys:
                    print(f"  {level}: {pk} = {d[pk]}%")
            break

print(f"\nDelta: {results['delta']['pct_new_compounds']}% new compounds, "
      f"{results['delta']['pct_new_pairs']}% new pairs")
print(f"Saved: {OUT_DIR}overlap_analysis.json")
