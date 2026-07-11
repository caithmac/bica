#!/usr/bin/env python3
"""
Evaluate Random Forest (ECFP4 + AAC) on BALM Benchmark using COLD-DRUG split.
Uses the exact split implementation from BALM repo (_create_fold_setting_cold with entities="Drug").

Datasets: BindingDB_filtered, Mpro, USP7, LeakyPDB
Cold-drug split: unique Drug IDs sampled for test — no drug molecule appears in both train/test.
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
# COLD-DRUG split (from BALM repo: balm/datasets/bindingdb_filtered.py)
# ---------------------------------------------------------------------------
def create_cold_drug_split(df, fold_seed, test_frac=0.2, val_frac=0.1, entity_col="Drug"):
    """
    Cold-drug split: sample unique Drug values for test, ensuring no drug appears in
    both train and test. Validation drawn from remaining unique drugs.
    
    Returns train_idx, test_idx (val_idx discarded — RF doesn't need validation).
    """
    rng = np.random.RandomState(fold_seed)
    
    unique_drugs = df[entity_col].drop_duplicates().values
    n_test_drugs = int(len(unique_drugs) * test_frac)
    
    # Sample test drugs
    test_drugs = set(rng.choice(unique_drugs, size=n_test_drugs, replace=False))
    
    test_idx = df[df[entity_col].isin(test_drugs)].index.tolist()
    train_idx = df[~df[entity_col].isin(test_drugs)].index.tolist()
    
    return train_idx, test_idx

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------
DATASETS = {
    'BindingDB_filtered': {
        'hf_config': 'BindingDB_filtered',
        'smiles_col': 'Drug',
        'target_col': 'Target',
        'label_col': 'Y',
        'split_type': 'cold_drug',
    },
    'Mpro': {
        'hf_config': 'Mpro',
        'smiles_col': 'Drug',
        'target_col': 'Target',
        'label_col': 'Y',
        'split_type': 'cold_drug',
    },
    'USP7': {
        'hf_config': 'USP7',
        'smiles_col': 'Drug',
        'target_col': 'Target',
        'label_col': 'Y',
        'split_type': 'cold_drug',
    },
    'LeakyPDB': {
        'hf_config': 'LeakyPDB',
        'smiles_col': 'Drug',
        'target_col': 'Target',
        'label_col': 'Y',
        'split_type': 'cold_drug',
    },
}

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_dataset(name, cfg, seeds=[42, 123, 1234]):
    print(f"\n{'='*60}")
    print(f"Processing: {name} (cold-drug split)")
    print(f"{'='*60}")
    
    # Load
    print(f"  Loading {cfg['hf_config']} from BALM/BALM-benchmark...")
    ds = load_dataset("BALM/BALM-benchmark", cfg['hf_config'])
    split_key = list(ds.keys())[0]
    df = ds[split_key].to_pandas()
    print(f"  Loaded {len(df)} rows (split: '{split_key}')")
    print(f"  Columns: {list(df.columns)}")
    
    # Drop missing
    df = df.dropna(subset=[cfg['smiles_col'], cfg['target_col'], cfg['label_col']]).reset_index(drop=True)
    
    # --- BALM preprocessing: group by Drug_ID+Drug+Target_ID+Target, take max Y ---
    # This is what BindingDBDataset.__init__ does before splitting
    if 'Drug_ID' in df.columns and 'Target_ID' in df.columns:
        n_before = len(df)
        df = df.groupby(['Drug_ID', cfg['smiles_col'], 'Target_ID', cfg['target_col']])[cfg['label_col']].agg('max').reset_index()
        print(f"  BALM aggregation: {n_before} -> {len(df)} rows (max Y per drug-target pair)")
    else:
        print(f"  No Drug_ID/Target_ID columns — skipping BALM aggregation")
    
    print(f"  After dropna+aggregation: {len(df)} rows")
    print(f"  Unique drugs: {df[cfg['smiles_col']].nunique()}")
    print(f"  Unique targets: {df[cfg['target_col']].nunique()}")
    
    # Features
    print("  Computing ECFP4 fingerprints (2048 bits)...")
    ecfp4_feats = np.array([ecfp4_fingerprint(smi) for smi in df[cfg['smiles_col']]])
    
    print("  Computing AAC protein features (20 dims)...")
    aac_feats = np.array([compute_aac(seq) for seq in df[cfg['target_col']]])
    
    X = np.concatenate([ecfp4_feats, aac_feats], axis=1)
    y = df[cfg['label_col']].values.astype(np.float64)
    print(f"  Feature matrix: {X.shape}")
    
    # Multi-seed evaluation
    seed_results = []
    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        train_idx, test_idx = create_cold_drug_split(df, seed)
        print(f"  Cold-drug split: {len(train_idx)} train, {len(test_idx)} test")
        print(f"  Train unique drugs: {df.iloc[train_idx][cfg['smiles_col']].nunique()}")
        print(f"  Test unique drugs: {df.iloc[test_idx][cfg['smiles_col']].nunique()}")
        
        # Check no drug overlap
        train_drugs = set(df.iloc[train_idx][cfg['smiles_col']])
        test_drugs = set(df.iloc[test_idx][cfg['smiles_col']])
        overlap = train_drugs & test_drugs
        if overlap:
            print(f"  WARNING: {len(overlap)} drugs appear in both train and test!")
        else:
            print(f"  √ No drug overlap between train and test")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train RF
        print("  Training RF (500 trees, max_depth=20)...")
        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            random_state=seed,
            n_jobs=-1,
            verbose=0
        )
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r_val, p_val = pearsonr(y_test, y_pred)
        r_val = r_val if not np.isnan(r_val) else 0.0
        
        seed_results.append({
            'seed': seed,
            'RMSE': rmse,
            'Pearson_R': r_val,
            'N_train': len(train_idx),
            'N_test': len(test_idx),
        })
        print(f"  RMSE={rmse:.4f}, Pearson R={r_val:.4f}, N_test={len(test_idx)}")
    
    # Aggregate
    rmse_vals = [r['RMSE'] for r in seed_results]
    r_vals = [r['Pearson_R'] for r in seed_results]
    
    return {
        'dataset': name,
        'N': int(len(y)),
        'n_unique_drugs': int(df[cfg['smiles_col']].nunique()),
        'RMSE_mean': round(np.mean(rmse_vals), 4),
        'RMSE_std': round(np.std(rmse_vals, ddof=1), 4),
        'Pearson_R_mean': round(np.mean(r_vals), 4),
        'Pearson_R_std': round(np.std(r_vals, ddof=1), 4),
        'seeds': seed_results,
    }

def main():
    results = []
    for name, cfg in DATASETS.items():
        try:
            res = evaluate_dataset(name, cfg, seeds=[42, 123, 1234])
            results.append(res)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                'dataset': name,
                'error': str(e)
            })
    
    # Summary
    print("\n\n" + "="*100)
    print("SUMMARY — RF (ECFP4 + AAC) on BALM COLD-DRUG split (3 seeds)")
    print("="*100)
    print(f"{'Dataset':<25} {'N':<8} {'N_drugs':<10} {'RMSE':<16} {'Pearson R':<16}")
    print("-"*100)
    for r in results:
        if 'error' in r:
            print(f"{r['dataset']:<25} ERROR: {r['error']}")
        else:
            rmse_str = f"{r['RMSE_mean']:.4f} ± {r['RMSE_std']:.4f}"
            r_str = f"{r['Pearson_R_mean']:.4f} ± {r['Pearson_R_std']:.4f}"
            print(f"{r['dataset']:<25} {r['N']:<8} {r['n_unique_drugs']:<10} {rmse_str:<16} {r_str:<16}")
    print("="*100)
    
    # Save
    out_path = 'E:/Drug Discovery/balm_colddrug_rf_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
