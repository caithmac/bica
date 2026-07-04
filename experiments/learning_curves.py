"""Learning curves: RF vs MLP at different training sizes + Fine-tuned ESM-2.
Upload to bullitt and run."""
import json, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from datasets import load_dataset

print("Loading BindingDB_filtered...")
ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered", split="train")
df = ds.to_pandas()

# Parse ligands
smiles_list = df['Drug'].tolist()
y = df['Y'].values.astype(float)
X = []
for smi in smiles_list:
    try:
        mol = Chem.MolFromSmiles(smi)
        fp = GetMorganFingerprintAsBitVect(mol, 2, 1024)
        X.append(list(fp))
    except:
        X.append([0]*1024)
X = np.array(X, dtype=np.float32)

# Scaffold split
scaffolds = {}
for i, smi in enumerate(smiles_list):
    try:
        mol = Chem.MolFromSmiles(smi)
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else str(i)
    except: sc = str(i)
    scaffolds.setdefault(sc, []).append(i)
sc_keys = list(scaffolds.keys())
np.random.RandomState(42).shuffle(sc_keys)
all_idx = []
for sc in sc_keys: all_idx.extend(scaffolds[sc])
train_mask = np.zeros(len(X), dtype=bool)
train_mask[all_idx[:int(0.7*len(X))]] = True
test_mask = np.zeros(len(X), dtype=bool)
test_mask[all_idx[-int(0.2*len(X)):]] = True
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Learning curves
sizes = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
results = {'rf': [], 'mlp': []}

for frac in sizes:
    n = int(frac * len(X_train))
    idx = np.random.RandomState(42).choice(len(X_train), n, replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    # RF
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_sub, y_sub)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
    results['rf'].append({'frac': frac, 'n': n, 'rmse': float(rf_rmse)})
    print(f"  RF  n={n:6d}  RMSE={rf_rmse:.4f}")

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(256,256), activation='relu', alpha=1e-4,
                       batch_size=128, learning_rate_init=1e-3, max_iter=100,
                       early_stopping=True, random_state=42)
    mlp.fit(X_sub, y_sub)
    mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp.predict(X_test)))
    results['mlp'].append({'frac': frac, 'n': n, 'rmse': float(mlp_rmse)})
    print(f"  MLP n={n:6d}  RMSE={mlp_rmse:.4f}")

with open('learning_curve_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved learning_curve_results.json")
