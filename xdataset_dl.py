"""Run MLP (best DL) on all 7 BALM datasets for cross-dataset comparison."""
import json, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from datasets import load_dataset

CONFIGS = {
    'BindingDB_filtered': ('Drug', 'Y', False),
    'HIF2A': ('Drug', 'Y', False),
    'LeakyPDB': ('Drug', 'Y', True),
    'MCL1': ('Drug', 'Y', False),
    'Mpro': ('Drug', 'Y', False),
    'SYK': ('Drug', 'Y', False),
    'USP7': ('Drug', 'Y', False),
}

results = {}
for ds_name, (smiles_col, label_col, has_presplit) in CONFIGS.items():
    print(f'\n=== {ds_name} ===')
    try:
        ds = load_dataset("BALM/BALM-benchmark", ds_name, split="train")
        df = ds.to_pandas()
        print(f'  Rows: {len(df)}')

        smiles_list = df[smiles_col].tolist()
        y = df[label_col].values.astype(float)
        ligs, valid_idx = [], []
        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is not None:
                    fp = GetMorganFingerprintAsBitVect(mol, 2, 1024)
                    ligs.append(list(fp)); valid_idx.append(i)
            except: pass
        print(f'  Valid: {len(ligs)}/{len(smiles_list)}')
        if len(ligs) < 30: print('  SKIP'); continue

        X = np.array(ligs, dtype=np.float32); yv = y[valid_idx]
        if has_presplit:
            splits = df['new_split'].values[valid_idx]
            train_mask = splits == 'train'; test_mask = splits == 'test'
        else:
            scaffolds = {}
            for idx, orig_i in enumerate(valid_idx):
                smi = str(smiles_list[orig_i])
                try: mol = Chem.MolFromSmiles(smi); sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else str(orig_i)
                except: sc = str(orig_i)
                scaffolds.setdefault(sc, []).append(idx)
            sc_keys = list(scaffolds.keys()); np.random.RandomState(42).shuffle(sc_keys)
            all_idx = []; [all_idx.extend(scaffolds[sc]) for sc in sc_keys]
            n_train = int(0.8 * len(ligs))
            train_mask = np.zeros(len(ligs), dtype=bool); train_mask[all_idx[:n_train]] = True
            test_mask = np.zeros(len(ligs), dtype=bool); test_mask[all_idx[n_train:]] = True

        # RF
        rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(X[train_mask], yv[train_mask]); rf_pred = rf.predict(X[test_mask])
        rf_rmse = np.sqrt(mean_squared_error(yv[test_mask], rf_pred))
        rf_r = np.corrcoef(rf_pred, yv[test_mask])[0,1]

        # MLP
        mlp = MLPRegressor(hidden_layer_sizes=(256,256), activation='relu', alpha=1e-4,
                           batch_size=128, learning_rate_init=1e-3, max_iter=200,
                           early_stopping=True, random_state=42)
        mlp.fit(X[train_mask], yv[train_mask]); mlp_pred = mlp.predict(X[test_mask])
        mlp_rmse = np.sqrt(mean_squared_error(yv[test_mask], mlp_pred))
        mlp_r = np.corrcoef(mlp_pred, yv[test_mask])[0,1]

        print(f'  RF:  RMSE={rf_rmse:.4f} R={rf_r:.4f}')
        print(f'  MLP: RMSE={mlp_rmse:.4f} R={mlp_r:.4f}')
        results[ds_name] = {'n': int(len(ligs)), 'n_test': int(test_mask.sum()),
                            'rf_rmse': float(rf_rmse), 'rf_r': float(rf_r),
                            'mlp_rmse': float(mlp_rmse), 'mlp_r': float(mlp_r)}
    except Exception as e:
        print(f'  ERROR: {e}')

print('\n\n=== FULL RESULTS ===')
print(f'{"Dataset":22s} {"N":>6s} {"RF_RMSE":>8s} {"MLP_RMSE":>8s} {"RF_wins?":>10s}')
for ds in CONFIGS:
    if ds in results:
        r = results[ds]
        winner = 'YES' if r['rf_rmse'] < r['mlp_rmse'] else 'NO -> DL'
        print(f'{ds:22s} {r[\"n\"]:6d} {r[\"rf_rmse\"]:8.4f} {r[\"mlp_rmse\"]:8.4f} {winner:>10s}')
    else:
        print(f'{ds:22s} FAILED')

with open('xdataset_full_results.json', 'w') as f: json.dump(results, f, indent=2)
print('\nSaved.')
