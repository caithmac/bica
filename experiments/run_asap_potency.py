"""Train RF+ECFP4 on ASAP potency using official Polaris evaluation protocol.

Matches https://github.com/asapdiscovery/asap-polaris-blind-challenge-examples
- Excludes flagged compounds (5 per target, from official exclusions JSON)
- Bootstrap rng(0) matching official bootstrapping_sampler
- Macro metrics via mean across targets per bootstrap iteration
"""
import pandas as pd
import numpy as np
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

# ---- Official flag exclusions ----
EXCLUSIONS = {
    "pIC50 (SARS-CoV-2 Mpro)": [5, 8, 188, 194, 275],
    "pIC50 (MERS-CoV Mpro)":   [5, 8, 188, 194, 275],
}

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

    pred = rf.predict(X_test)
    true = test[tgt].values

    # NaN mask
    valid = ~np.isnan(true)
    pred_v = pred[valid]
    true_v = true[valid]

    # Map flag exclusion indices to post-NaN-filter positions
    flagged = EXCLUSIONS[tgt]
    exclude_positions = []
    for fi in flagged:
        nan_positions = np.where(~valid)[0]
        offset = sum(1 for np_ in nan_positions if np_ < fi)
        exclude_positions.append(fi - offset)

    keep = ~np.isin(np.arange(len(pred_v)), exclude_positions)
    y_pred_all[tgt] = pred_v[keep]
    y_true_all[tgt] = true_v[keep]

    rmse = np.sqrt(mean_squared_error(true_v[keep], pred_v[keep]))
    r = pearsonr(true_v[keep], pred_v[keep])[0]
    print(f"{tgt}: RMSE={rmse:.3f}, Pearson r={r:.3f} (n={keep.sum()})")

# ---- Bootstrap (matching official: rng(0), 1000 samples) ----
N_BOOTSTRAP = 1000
rng = np.random.default_rng(0)

METRICS = {
    "MAE": mean_absolute_error,
    "RMSE": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
    "Pearson r": lambda yt, yp: pearsonr(yt, yp)[0],
    "Spearman r": lambda yt, yp: spearmanr(yt, yp)[0],
    "R²": lambda yt, yp: r2_score(y_true=yt, y_pred=yp),
    "Kendall τ": lambda yt, yp: kendalltau(yt, yp).statistic,
}

# Generate all bootstrap indices once per target (matching official approach)
bootstrap_idx = {}
for tgt in TARGETS:
    n = len(y_true_all[tgt])
    bootstrap_idx[tgt] = rng.choice(n, size=(N_BOOTSTRAP, n), replace=True)

# Per-target + macro scores
per_target = {}
macro = {}

for name, fn in METRICS.items():
    per_target[name] = {}
    all_target_samples = []  # one array per target, shape (N_BOOTSTRAP,)

    for tgt in TARGETS:
        yt, yp = y_true_all[tgt], y_pred_all[tgt]
        idx = bootstrap_idx[tgt]
        samples = np.array([fn(yt[idx[i]], yp[idx[i]]) for i in range(N_BOOTSTRAP)])
        per_target[name][tgt] = (samples.mean(), samples.std())
        all_target_samples.append(samples)

    # Macro: mean across targets per iteration
    macro_samples = np.mean(all_target_samples, axis=0)
    macro[name] = (macro_samples.mean(), macro_samples.std())

print(f"\n=== Bootstrap Results ({N_BOOTSTRAP} samples, rng(0), flagged excluded) ===")
print(f"{'Metric':<15} {'SARS-CoV-2':>20} {'MERS-CoV':>20} {'Macro Avg':>20}")
print("-" * 80)
for name in METRICS:
    sars = f"{per_target[name][TARGETS[1]][0]:.3f} ± {per_target[name][TARGETS[1]][1]:.3f}"
    mers = f"{per_target[name][TARGETS[0]][0]:.3f} ± {per_target[name][TARGETS[0]][1]:.3f}"
    mac  = f"{macro[name][0]:.3f} ± {macro[name][1]:.3f}"
    print(f"{name:<15} {sars:>20} {mers:>20} {mac:>20}")

print("\n=== Head-to-head (point estimate) ===")
print("Method: RF + ECFP4 (500 trees)")
for tgt in TARGETS:
    yt, yp = y_true_all[tgt], y_pred_all[tgt]
    print(f"  {tgt}: MAE={mean_absolute_error(yt,yp):.3f}  RMSE={np.sqrt(mean_squared_error(yt,yp)):.3f}  "
          f"Pearson r={pearsonr(yt,yp)[0]:.3f}  Spearman r={spearmanr(yt,yp)[0]:.3f}  n={len(yt)}")
