"""Cross-dataset RF benchmark for 4 meaningful BALM datasets."""
import json, numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# Dataset configs: (huggingface_name, smiles_col, label_col, has_presplit)
CONFIGS = {
    'BindingDB_filtered': ('Drug', 'Y', False),
    'LeakyPDB': ('Drug', 'Y', True),   # has new_split column
    'Mpro': ('Drug', 'Y', False),
    'USP7': ('Drug', 'Y', False),
}

results = {}
for ds_name, (smiles_col, label_col, has_presplit) in CONFIGS.items():
    print(f'\n=== {ds_name} ===')
    try:
        ds = load_dataset("BALM/BALM-benchmark", ds_name, split="train")
        df = ds.to_pandas()
        print(f'  Rows: {len(df)}')

        # Parse SMILES
        smiles_list = df[smiles_col].tolist()
        y = df[label_col].values.astype(float)
        ligs = []
        valid_idx = []
        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is not None:
                    fp = GetMorganFingerprintAsBitVect(mol, 2, 1024)
                    ligs.append(list(fp))
                    valid_idx.append(i)
            except: pass
        print(f'  Valid: {len(ligs)}/{len(smiles_list)}')

        if len(ligs) < 100:
            print(f'  SKIP: too few valid molecules')
            continue

        X = np.array(ligs, dtype=np.float32)
        y_valid = y[valid_idx]

        if has_presplit:
            # Use LeakyPDB's built-in split
            splits = df['new_split'].values[valid_idx]
            train_mask = splits == 'train'
            val_mask = splits == 'val'
            test_mask = splits == 'test'
            print(f'  Using presplit: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}')
        else:
            # Scaffold split
            scaffolds = {}
            for idx, orig_i in enumerate(valid_idx):
                smi = str(smiles_list[orig_i])
                try:
                    mol = Chem.MolFromSmiles(smi)
                    sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else str(orig_i)
                except: sc = str(orig_i)
                scaffolds.setdefault(sc, []).append(idx)
            sc_keys = list(scaffolds.keys())
            np.random.RandomState(42).shuffle(sc_keys)
            train_idx, val_idx, test_idx = [], [], []
            for sc in sc_keys:
                if len(train_idx) < 0.7 * len(ligs): train_idx.extend(scaffolds[sc])
                elif len(val_idx) < 0.1 * len(ligs): val_idx.extend(scaffolds[sc])
                else: test_idx.extend(scaffolds[sc])
            train_mask = np.zeros(len(ligs), dtype=bool); train_mask[train_idx] = True
            val_mask = np.zeros(len(ligs), dtype=bool); val_mask[val_idx] = True
            test_mask = np.zeros(len(ligs), dtype=bool); test_mask[test_idx] = True
            print(f'  Scaffold split: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}')

        # Train RF
        rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(X[train_mask], y_valid[train_mask])
        pred = rf.predict(X[test_mask])
        rmse = np.sqrt(mean_squared_error(y_valid[test_mask], pred))
        pearson = np.corrcoef(pred, y_valid[test_mask])[0, 1]
        print(f'  TEST_RMSE={rmse:.4f}  PEARSON={pearson:.4f}  N_test={test_mask.sum()}')
        results[ds_name] = {'rmse': float(rmse), 'pearson': float(pearson), 'n_test': int(test_mask.sum()), 'n_total': int(len(ligs))}

    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback; traceback.print_exc()

print('\n\n=== CROSS-DATASET RESULTS ===')
print(f'{"Dataset":25s} {"N":>6s} {"RMSE":>8s} {"Pearson R":>10s} {"N test":>8s}')
print('-'*60)
for ds in CONFIGS:
    if ds in results:
        r = results[ds]
        print(f'{ds:25s} {r["n_total"]:6d} {r["rmse"]:8.4f} {r["pearson"]:10.4f} {r["n_test"]:8d}')
    else:
        print(f'{ds:25s} {"FAILED":>6s}')

with open('cross_dataset_results.json','w') as f:
    json.dump(results, f, indent=2)
print('\nSaved locally.')
