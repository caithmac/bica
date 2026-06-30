"""Train RF+ECFP4 on ASAP potency and evaluate per blind challenge protocol."""
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau

# ---- Load ----
df = pd.read_csv("potency.csv")
train = df[df["Set"] == "Train"].copy()
test  = df[df["Set"] == "Test"].copy()
TARGETS = ["pIC50 (MERS-CoV Mpro)", "pIC50 (SARS-CoV-2 Mpro)"]

# ---- Featurize ECFP4 ----
gen = GetMorganGenerator(radius=2, fpSize=1024)
def ecfp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return gen.GetFingerprintAsNumPy(mol).astype(np.float32) if mol else np.zeros(1024, dtype=np.float32)

X_train = np.stack([ecfp(s) for s in train["CXSMILES"]])
X_test  = np.stack([ecfp(s) for s in test["CXSMILES"]])

# ---- Train & predict per target ----
y_pred_all = {}
y_true_all = {}

for tgt in TARGETS:
    mask = train[tgt].notna()
    X_tr, y_tr = X_train[mask], train.loc[mask, tgt].values.astype(np.float32)
    
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    
    pred = rf.predict(X_test).astype(np.float64)
    true = test[tgt].values.astype(np.float64)
    
    valid = ~np.isnan(true)
    y_pred_all[tgt] = pred[valid]
    y_true_all[tgt] = true[valid]
    
    rmse = np.sqrt(mean_squared_error(true[valid], pred[valid]))
    r = pearsonr(true[valid], pred[valid])[0]
    print(f"{tgt}: RMSE={rmse:.3f}, Pearson r={r:.3f} (n={valid.sum()})")

# ---- Bootstrap evaluation (matching ASAP protocol) ----
N_BOOTSTRAP = 1000
rng = np.random.default_rng(42)
metrics = {"MAE": mean_absolute_error, "RMSE": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
           "Pearson r": lambda yt, yp: pearsonr(yt, yp)[0],
           "Spearman r": lambda yt, yp: spearmanr(yt, yp)[0],
           "R²": r2_score, "Kendall τ": lambda yt, yp: kendalltau(yt, yp).statistic}

print("\n=== Bootstrap Results (1000 samples) ===")
print(f"{'Metric':<15} {'SARS-CoV-2':>12} {'MERS-CoV':>12} {'Macro Avg':>12}")
print("-" * 55)

# Per-target and macro
all_scores = {}
for tgt in TARGETS:
    yt = y_true_all[tgt]
    yp = y_pred_all[tgt]
    n = len(yt)
    
    for name, fn in metrics.items():
        samples = np.array([fn(yt[bt_idx], yp[bt_idx]) for bt_idx in rng.choice(n, size=(N_BOOTSTRAP, n), replace=True)])
        all_scores.setdefault(name, {})[tgt] = (samples.mean(), samples.std())

# Macro: average of per-target means
for name in metrics:
    vals = [all_scores[name][tgt][0] for tgt in TARGETS]
    all_scores[name]["macro"] = (np.mean(vals), 0)

for name in metrics:
    sars = f"{all_scores[name][TARGETS[1]][0]:.3f} ± {all_scores[name][TARGETS[1]][1]:.3f}"
    mers = f"{all_scores[name][TARGETS[0]][0]:.3f} ± {all_scores[name][TARGETS[0]][1]:.3f}"
    macro = f"{all_scores[name]['macro'][0]:.3f}"
    print(f"{name:<15} {sars:>12} {mers:>12} {macro:>12}")

print("\n=== Head-to-head (point estimate) ===")
print(f"Method: RF + ECFP4 (500 trees)")
for tgt in TARGETS:
    yt, yp = y_true_all[tgt], y_pred_all[tgt]
    print(f"  {tgt}: MAE={mean_absolute_error(yt,yp):.3f}  RMSE={np.sqrt(mean_squared_error(yt,yp)):.3f}  Pearson r={pearsonr(yt,yp)[0]:.3f}  Spearman r={spearmanr(yt,yp)[0]:.3f}")
