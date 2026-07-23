#!/usr/bin/env python3
"""Phase 7: RF TreeSHAP interpretability on best RF across all split types.
Computes SHAP values, maps ECFP4 bits to substructures, computes consistency across splits.
"""
import json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import IPythonConsole
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import shap

warnings.filterwarnings('ignore')

sys.path.insert(0, "E:/Drug Discovery")
from harness.featurizers import ecfp, amino_acid_composition, concat

SEEDS = [42, 123, 456]
SPLIT_TYPES = ['random', 'scaffold', 'cold_target']
OUT_DIR = "E:/Drug Discovery/projects/balm-revision/results/shap"
os.makedirs(OUT_DIR, exist_ok=True)

print("[Phase 7] TreeSHAP on RF_ECFP4+AAC across splits")

# --- Load data ---
df = pd.read_parquet("E:/Drug Discovery/projects/balm-revision/data/frozen/balm_filtered.parquet")

all_shap = {}

for split_type in SPLIT_TYPES:
    for seed in SEEDS:
        key = f"{split_type}_seed{seed}"
        split_dir = Path(f"E:/Drug Discovery/projects/balm-revision/data/splits/{split_type}/seed_{seed}")
        
        print(f"\n  {key}...")
        
        train_df = pd.read_csv(split_dir / "train.csv")
        test_df = pd.read_csv(split_dir / "test.csv")
        
        # Featurize
        X_train = concat(ecfp(train_df['Drug_canonical'].tolist()),
                         amino_acid_composition(train_df['Target'].tolist()))
        X_test = concat(ecfp(test_df['Drug_canonical'].tolist()),
                        amino_acid_composition(test_df['Target'].tolist()))
        y_train = train_df['Y'].values
        y_test = test_df['Y'].values
        
        # Fit RF
        rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # TreeSHAP (use background sampling for speed)
        n_background = min(500, len(X_train))
        bg_idx = np.random.RandomState(seed).choice(len(X_train), n_background, replace=False)
        
        print(f"    Computing SHAP values on {len(X_test)} test samples...")
        explainer = shap.TreeExplainer(rf, X_train[bg_idx])
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        
        # Global importance (mean |SHAP|)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Separate ECFP4 (1024 bits) vs AAC (20 features)
        ecfp4_shap = mean_abs_shap[:1024]
        aac_shap = mean_abs_shap[1024:1044]
        
        # Top 20 ECFP4 bits
        top20_idx = np.argsort(ecfp4_shap)[-20:][::-1]
        top20_values = ecfp4_shap[top20_idx]
        
        # Map ECFP4 bits to substructures on representative molecules
        bit_substructures = {}
        for bit_idx in top20_idx[:10]:
            # Find molecules where this bit is set
            bit_set = X_test[:, bit_idx] > 0
            if bit_set.sum() > 0:
                # Take first example
                idx = np.where(bit_set)[0][0]
                smi = test_df.iloc[idx]['Drug_canonical']
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    # Get the substructure for this bit
                    bi = {}
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo=bi)
                    if bit_idx in bi:
                        bit_substructures[str(bit_idx)] = f"bit_{bit_idx}"
        
        # AAC importance
        amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        aac_importance = {aa: float(aac_shap[i]) for i, aa in enumerate(amino_acids)}
        
        all_shap[key] = {
            'n_train': len(train_df),
            'n_test': len(test_df),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))),
            'test_pearson': float(pearsonr(y_test, rf.predict(X_test))[0]),
            'top20_ecfp4_bits': {str(idx): float(val) for idx, val in zip(top20_idx, top20_values)},
            'bit_substructures': bit_substructures,
            'aac_importance': aac_importance,
            'mean_abs_shap_ecfp4': float(ecfp4_shap.mean()),
            'mean_abs_shap_aac': float(aac_shap.mean()),
        }
        
        # Save SHAP values separately (large)
        shap_out = f"{OUT_DIR}/shap_{key}.npy"
        np.save(shap_out, shap_values)
        all_shap[key]['shap_file'] = shap_out
        
        print(f"    Top ECFP4 bit: {top20_idx[0]} (SHAP={top20_values[0]:.4f})")
        top_aac = sorted(aac_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"    Top AAC: {top_aac}")

# --- Cross-split consistency ---
print("\n--- Cross-Split Consistency ---")
# Check top bits across all splits
bit_ranks = {}
for key, data in all_shap.items():
    split_type = key.split('_seed')[0]
    if split_type not in bit_ranks:
        bit_ranks[split_type] = {}
    for bit, val in data['top20_ecfp4_bits'].items():
        bit_ranks[split_type][bit] = float(val)

# Find bits important across multiple split types
all_top_bits = set()
for bits in bit_ranks.values():
    all_top_bits.update(bits.keys())

consistent_bits = []
for bit in all_top_bits:
    splits_with_bit = [st for st, bits in bit_ranks.items() if bit in bits]
    if len(splits_with_bit) >= 2:
        mean_importance = np.mean([bit_ranks[st][bit] for st in splits_with_bit])
        consistent_bits.append((bit, len(splits_with_bit), mean_importance))

consistent_bits.sort(key=lambda x: x[1], reverse=True)

print(f"  Bits in 2+ split types: {len(consistent_bits)}")
for bit, n_splits, mean_imp in consistent_bits[:10]:
    print(f"    Bit {bit}: in {n_splits} split types, mean SHAP={mean_imp:.4f}")

# --- Save summary ---
summary = {
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'n_splits': len(all_shap),
    'splits': list(all_shap.keys()),
    'consistent_top_bits': [{'bit': b, 'n_splits': n, 'mean_shap': float(m)} 
                            for b, n, m in consistent_bits[:20]],
    'note': 'TreeSHAP on RF_ECFP4+AAC (500 trees, max_depth=20). ECFP4 bits 0-1023, AAC features 1024-1043.'
}

with open(f"{OUT_DIR}/phase7_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[Phase 7] COMPLETE — {OUT_DIR}/phase7_summary.json")
