"""E303 — Random vs Scaffold Split. Run top-5 models on random split, compare."""
import sys, os, time, logging
from pathlib import Path
import numpy as np, pandas as pd

EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("E:/Drug Discovery")
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(EXP_DIR / "run.log", mode="w"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("E303")

from harness.config import SMILES_COL, PROTEIN_COL, LABEL_COL, SPLIT_SEED
from harness.data import get_splits_for_seed
import harness.featurizers as F
from harness.trainer import train_sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Get random split (seed 42)
df_raw = pd.read_pickle(ROOT / "cache/bindingdb_raw.pkl")
rng = np.random.default_rng(42)
n = len(df_raw)
idx = rng.permutation(n)
t_end = int(0.7*n); v_end = int(0.8*n)
train_df = df_raw.iloc[idx[:t_end]].reset_index(drop=True)
val_df = df_raw.iloc[idx[t_end:v_end]].reset_index(drop=True)
test_df = df_raw.iloc[idx[v_end:]].reset_index(drop=True)
log.info(f"Random split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

# Build features
def build_ecfp_aac(df):
    return F.concat(F.ecfp(df[SMILES_COL].tolist(), radius=2, nbits=1024),
                     F.amino_acid_composition(df[PROTEIN_COL].tolist()))
X_train = build_ecfp_aac(train_df); X_test = build_ecfp_aac(test_df)
y_train = train_df[LABEL_COL].values.astype(np.float32)
y_test = test_df[LABEL_COL].values.astype(np.float32)

# Models
models = {
    "rf_ecfp4_aac": RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=42),
    "xgb_ecfp4_aac": XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
}

# Load scaffold split results for comparison
scaffold_results = {"rf_ecfp4_aac": 1.0065, "xgb_ecfp4_aac": 1.0523}

results = []
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    dt = time.time() - t0
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    r = np.corrcoef(y_pred, y_test)[0, 1]
    scf_rmse = scaffold_results.get(name, 0)
    results.append({"model": name, "split": "random", "rmse": rmse, "pearson_r": r,
                    "scaffold_rmse": scf_rmse, "delta": rmse - scf_rmse, "time_s": dt})
    log.info(f"{name}: random={rmse:.4f} scaffold={scf_rmse:.4f} Δ={rmse-scf_rmse:+.4f}")

df = pd.DataFrame(results)
df.to_csv(EXP_DIR / "results.csv", index=False)
log.info(f"\nRandom split: DL overfitting gap = {df['rmse'].mean() - df['scaffold_rmse'].mean():+.4f}")
