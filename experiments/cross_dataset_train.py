"""Cross-dataset RF benchmark. Run ON bullitt."""
import json, sys, numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

DATASETS = ['BindingDB_filtered','HIF2A','LeakyPDB','MCL1','Mpro','SYK','USP7']
results = {}

for ds_name in DATASETS:
    print(f'\n=== {ds_name} ===')
    try:
        ds = load_dataset("BALM/BALM-benchmark", ds_name, split="train")
        df = ds.to_pandas()
        print(f'  Rows: {len(df)}, Cols: {list(df.columns)}')

        # Ligand column is always first, label is last
        smiles_col = df.columns[0]
        label_col = df.columns[-1]
        smiles_list = df[smiles_col].tolist()
        y = df[label_col].values.astype(float)
        print(f'  Label range: {y.min():.2f} - {y.max():.2f}, mean: {y.mean():.2f}')

        # Compute ECFP4 fingerprints
        ligs = []
        valid = []
        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = GetMorganFingerprintAsBitVect(mol, 2, 1024)
                    ligs.append(list(fp))
                    valid.append(i)
            except:
                pass
        print(f'  Valid ligands: {len(ligs)}/{len(smiles_list)}')

        if len(ligs) < 100:
            print(f'  SKIP: too few valid molecules')
            continue

        X = np.array(ligs, dtype=np.float32)
        y = y[valid]

        # Scaffold split
        scaffolds = {}
        for idx, i in enumerate(valid):
            smi = smiles_list[i]
            try:
                mol = Chem.MolFromSmiles(smi)
                sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else str(i)
            except:
                sc = str(i)
            scaffolds.setdefault(sc, []).append(idx)

        sc_keys = list(scaffolds.keys())
        np.random.RandomState(42).shuffle(sc_keys)
        train_idx, val_idx, test_idx = [], [], []
        for sc in sc_keys:
            if len(train_idx) < 0.7 * len(ligs):
                train_idx.extend(scaffolds[sc])
            elif len(val_idx) < 0.1 * len(ligs):
                val_idx.extend(scaffolds[sc])
            else:
                test_idx.extend(scaffolds[sc])

        # Train RF
        rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        pred = rf.predict(X[test_idx])
        rmse = np.sqrt(mean_squared_error(y[test_idx], pred))
        pearson = np.corrcoef(pred, y[test_idx])[0, 1]

        print(f'  TEST_RMSE={rmse:.4f}  PEARSON={pearson:.4f}  N_test={len(test_idx)}')
        results[ds_name] = {'rmse': float(rmse), 'pearson': float(pearson), 'n': len(test_idx)}

    except Exception as e:
        print(f'  ERROR: {e}')

print('\n\n=== CROSS-DATASET RESULTS ===')
print(f'{"Dataset":25s} {"RMSE":>8s} {"Pearson R":>10s} {"N test":>8s}')
print('-'*55)
for ds in DATASETS:
    if ds in results:
        r = results[ds]
        print(f'{ds:25s} {r["rmse"]:8.4f} {r["pearson"]:10.4f} {r["n"]:8d}')
    else:
        print(f'{ds:25s} {"FAILED":>8s}')

with open('cross_dataset_results.json','w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to cross_dataset_results.json')
