#!/usr/bin/env python3
"""
Evaluate Random Forest (ECFP4 + AAC protein features) on all 4 BALM datasets.
Uses scaffold split for BindingDB, Mpro, USP7; new_split column for LeakyPDB.

Datasets are loaded from HuggingFace: BALM/BALM-benchmark (configs: BindingDB_filtered, LeakyPDB, Mpro, USP7).
"""
import os, sys, json, warnings, traceback
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datasets import load_dataset
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
RDLogger.logger().setLevel(RDLogger.ERROR)

# ---------------------------------------------------------------------------
# AAC computation
# ---------------------------------------------------------------------------
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def compute_aac(sequence):
    if not isinstance(sequence, str) or len(sequence) == 0:
        return [0]*20
    counts = [sequence.count(aa) for aa in AMINO_ACIDS]
    total = sum(counts)
    return [c/total for c in counts] if total > 0 else [0]*20

def ecfp4_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    'BindingDB_filtered': {
        'hf_config': 'BindingDB_filtered',
        'smiles_col': 'Drug',
        'target_col': 'Target',
        'label_col': 'Y',
        'split': 'scaffold'
    },
    'LeakyPDB': {
        'hf_config': 'LeakyPDB',
        'smiles_col': 'Drug',
        'target_col': 'Target',
        'label_col': 'Y',
        'split': 'new_split'
    },
    'Mpro': {
        'hf_config': 'Mpro',
        'smiles_col': 'Drug',
        'target_col': 'Target',
        'label_col': 'Y',
        'split': 'scaffold'
    },
    'USP7': {
        'hf_config': 'USP7',
        'smiles_col': 'Drug',
        'target_col': 'Target',
        'label_col': 'Y',
        'split': 'scaffold'
    }
}

def load_balm_dataset(hf_config):
    """Load a BALM-benchmark dataset from HuggingFace."""
    print(f"  Loading {hf_config} from HuggingFace (BALM/BALM-benchmark)...")
    ds = load_dataset("BALM/BALM-benchmark", hf_config)
    # Use the first available split (usually 'train')
    split_key = list(ds.keys())[0]
    df = ds[split_key].to_pandas()
    print(f"  Loaded {len(df)} rows from split '{split_key}'")
    return df

def get_scaffold_split(df, smiles_col, test_ratio=0.2):
    """Deterministic scaffold split using Bemis-Murcko scaffolds."""
    from collections import defaultdict
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = defaultdict(list)
    for i, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except:
                scaffold = smi
        else:
            scaffold = smi
        scaffolds[scaffold].append(i)

    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

    train_idx, test_idx = [], []
    test_count = int(len(df) * test_ratio)
    for sset in scaffold_sets:
        if len(test_idx) + len(sset) <= test_count:
            test_idx.extend(sset)
        else:
            train_idx.extend(sset)
    return train_idx, test_idx

def evaluate_dataset(name, cfg):
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    df = load_balm_dataset(cfg['hf_config'])

    # Ensure columns exist
    for col in [cfg['smiles_col'], cfg['target_col'], cfg['label_col']]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {name}. Available: {list(df.columns)}")

    print(f"  Columns: {list(df.columns)}")

    # Drop rows with missing values in key columns
    df = df.dropna(subset=[cfg['smiles_col'], cfg['target_col'], cfg['label_col']]).reset_index(drop=True)

    print(f"  Total samples after dropna: {len(df)}")

    # ---- feature engineering ----
    print("  Computing ECFP4 fingerprints (2048 bits)...")
    ecfp4_feats = np.array([ecfp4_fingerprint(smi) for smi in df[cfg['smiles_col']]])

    print("  Computing AAC protein features (20 dims)...")
    aac_feats = np.array([compute_aac(seq) for seq in df[cfg['target_col']]])

    X = np.concatenate([ecfp4_feats, aac_feats], axis=1)
    y = df[cfg['label_col']].values.astype(np.float64)

    print(f"  Feature matrix: {X.shape}")

    # ---- split ----
    if cfg['split'] == 'new_split':
        if 'new_split' not in df.columns:
            raise KeyError(f"'new_split' column required for {name} but not found.")
        train_idx = np.where(df['new_split'] == 'train')[0].tolist()
        test_idx  = np.where(df['new_split'] == 'test')[0].tolist()
        print(f"  Using 'new_split' column: {len(train_idx)} train, {len(test_idx)} test")
    else:
        train_idx, test_idx = get_scaffold_split(df, cfg['smiles_col'])
        print(f"  Scaffold split: {len(train_idx)} train, {len(test_idx)} test")

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Empty train ({len(train_idx)}) or test ({len(test_idx)}) for {name}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- train Random Forest ----
    print("  Training Random Forest (500 trees, max_depth=20, n_jobs=-1)...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)

    # ---- evaluate ----
    y_pred = rf.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r_val, p_val = pearsonr(y_test, y_pred)
    r_val = r_val if not np.isnan(r_val) else 0.0

    result = {
        'dataset': name,
        'N': int(len(y)),
        'N_train': int(len(y_train)),
        'N_test': int(len(y_test)),
        'RMSE': round(rmse, 4),
        'Pearson_R': round(r_val, 4),
        'p_value': round(float(p_val), 6),
        'split': cfg['split']
    }
    print(f"  Results: RMSE={rmse:.4f}, Pearson R={r_val:.4f} (p={p_val:.6f}, test N={len(y_test)})")
    return result

def main():
    results = []
    for name, cfg in DATASETS.items():
        try:
            res = evaluate_dataset(name, cfg)
            results.append(res)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                'dataset': name,
                'N': -1,
                'N_train': -1,
                'N_test': -1,
                'RMSE': None,
                'Pearson_R': None,
                'error': str(e)
            })

    # ---- summary table ----
    print("\n\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Dataset':<25} {'N':<8} {'RMSE':<12} {'Pearson R':<12} {'N_test':<8}")
    print("-"*80)
    for r in results:
        rmse_str = f"{r['RMSE']:.4f}" if r.get('RMSE') is not None else "ERROR"
        r_str = f"{r['Pearson_R']:.4f}" if r.get('Pearson_R') is not None else "ERROR"
        print(f"{r['dataset']:<25} {r['N']:<8} {rmse_str:<12} {r_str:<12} {r['N_test']:<8}")
    print("="*80)

    out_path = '/home/rajeev/bica/xdataset_aac_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
