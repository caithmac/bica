"""GAP-1 — Representation Ablation. RF + 4 feature sets, 3 seeds. CPU, fast."""
import sys, os, time, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("E:/Drug Discovery")
sys.path.insert(0, str(ROOT))
from harness.config import SMILES_COL, PROTEIN_COL, LABEL_COL
from harness.data import get_splits_for_seed
import harness.featurizers as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(EXP_DIR / "run.log", mode="w"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("GAP1")

train_df, val_df, test_df = get_splits_for_seed(42)
log.info(f"Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}")

# Feature sets
FEAT_SETS = {
    "ECFP4_only": lambda df: F.ecfp(df[SMILES_COL].tolist(), radius=2, nbits=1024),
    "AAC_only": lambda df: F.amino_acid_composition(df[PROTEIN_COL].tolist()),
    "ECFP4+AAC": lambda df: F.concat(
        F.ecfp(df[SMILES_COL].tolist(), radius=2, nbits=1024),
        F.amino_acid_composition(df[PROTEIN_COL].tolist())),
    "ECFP4+ESM2": lambda df: F.concat(
        F.ecfp(df[SMILES_COL].tolist(), radius=2, nbits=1024),
        F.esm2_embeddings(df[PROTEIN_COL].tolist(), model_size="8M")),
}
SEEDS = [42, 123, 456]

results = []
for name, feat_fn in FEAT_SETS.items():
    log.info(f"\n{name}...")
    X_train = feat_fn(train_df)
    X_test = feat_fn(test_df)
    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_test = test_df[LABEL_COL].values.astype(np.float32)
    
    for seed in SEEDS:
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=500, max_depth=None,
                                   min_samples_split=2, n_jobs=-1, random_state=seed)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        dt = time.time() - t0
        rmse = np.sqrt(np.mean((y_pred - y_test)**2))
        r = np.corrcoef(y_pred, y_test)[0, 1]
        results.append({"feats": name, "seed": seed, "rmse": rmse, "pearson_r": r,
                        "dim": X_train.shape[1], "time_s": dt})
        log.info(f"  {name:20s} seed={seed} dim={X_train.shape[1]:5d} RMSE={rmse:.4f} R={r:.4f}")

df = pd.DataFrame(results)
df.to_csv(EXP_DIR / "results.csv", index=False)

# Summary
log.info("\n=== SUMMARY ===")
for name in FEAT_SETS:
    sub = df[df["feats"] == name]
    log.info(f"  {name:15s}: RMSE={sub['rmse'].mean():.4f}±{sub['rmse'].std():.4f}  "
             f"dim={sub['dim'].iloc[0]}  best={sub['rmse'].min():.4f}")
