"""
Learning curves: RF vs MLP with ECFP4-only, 80/20 random split.
Saves results to diary and generates values for paper.
"""
import numpy as np
import pandas as pd
import time, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem
from datasets import load_dataset

# ---- Load data ----
print("[1] Loading BindingDB_filtered...")
ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered", split="train")
df = ds.to_pandas()
df = df.dropna(subset=["Drug", "Y"])
smiles = df["Drug"].tolist()
y_all = df["Y"].values.astype(np.float32)
print(f"    {len(smiles):,} compounds")

# ---- ECFP4 ----
print("[2] Computing ECFP4 fingerprints...")
def ecfp(smiles_list, radius=2, nbits=1024):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(nbits, dtype=np.float32))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps)

X = ecfp(smiles)
print(f"    Shape: {X.shape}")

# ---- 80/20 split (random, seed 42) ----
np.random.seed(42)
n = len(X)
idx = np.random.permutation(n)
n_train = int(n * 0.80)
X_train_full = X[idx[:n_train]]
y_train_full = y_all[idx[:n_train]]
X_test = X[idx[n_train:]]
y_test = y_all[idx[n_train:]]
print(f"[3] 80/20 split: Train={len(X_train_full):,}  Test={len(X_test):,}")

# ---- Subset fractions ----
fractions = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
results = []

for frac in fractions:
    n_subset = int(len(X_train_full) * frac)
    X_sub = X_train_full[:n_subset]
    y_sub = y_train_full[:n_subset]
    
    # ---- RF ----
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_sub, y_sub)
    yp_rf = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, yp_rf))
    rf_r, _ = pearsonr(y_test, yp_rf)
    rf_time = time.time() - t0
    
    # ---- MLP ----
    t0 = time.time()
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128), activation='relu',
        alpha=0.0001, batch_size=128, learning_rate_init=0.001,
        max_iter=100, early_stopping=True, validation_fraction=0.1,
        random_state=42, verbose=False
    )
    mlp.fit(X_sub, y_sub)
    yp_mlp = mlp.predict(X_test)
    mlp_rmse = np.sqrt(mean_squared_error(y_test, yp_mlp))
    mlp_r, _ = pearsonr(y_test, yp_mlp)
    mlp_time = time.time() - t0
    
    gap = mlp_rmse - rf_rmse
    pct = int(frac * 100)
    print(f"  {pct:3d}% ({n_subset:5,}):  RF={rf_rmse:.4f}  MLP={mlp_rmse:.4f}  gap={gap:+.4f}  |  rf_t={rf_time:.1f}s  mlp_t={mlp_time:.1f}s")
    
    results.append({
        "frac": frac, "n_train": n_subset,
        "rf_rmse": rf_rmse, "rf_pearson": rf_r, "rf_time": rf_time,
        "mlp_rmse": mlp_rmse, "mlp_pearson": mlp_r, "mlp_time": mlp_time,
        "gap": gap
    })

# ---- Summary ----
print("\n[4] Summary for paper:")
print(f"    At 5%  ({results[0]['n_train']:,} cmpds): RF={results[0]['rf_rmse']:.3f}  MLP={results[0]['mlp_rmse']:.3f}  gap={results[0]['gap']:+.3f}")
print(f"    At 100% ({results[-1]['n_train']:,} cmpds): RF={results[-1]['rf_rmse']:.3f}  MLP={results[-1]['mlp_rmse']:.3f}  gap={results[-1]['gap']:+.3f}")

# ---- Save ----
df_res = pd.DataFrame(results)
df_res.to_csv("learning_curves.csv", index=False)
print("\n[5] Saved to learning_curves.csv")
