"""E202 — Fine-Tuning Sweep (resume from interruption)."""
import sys, os, time, logging
from pathlib import Path
import numpy as np, pandas as pd

EXP_DIR = Path(__file__).parent
ROOT = Path("E:/Drug Discovery")
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(EXP_DIR / "run.log", mode="a"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("E202")

from harness.config import LABEL_COL
from harness.data import get_splits_for_seed
import harness.featurizers as F
from harness.trainer import train_torch
from models.mlp import MLP
import torch

train_df, val_df, test_df = get_splits_for_seed(42)
FEAT_CACHE = Path("E:/Drug Discovery/cache/features")

L_train = np.load(FEAT_CACHE / "cb600_train.npy")
P_train = np.load(FEAT_CACHE / "esm2_8M_train.npy")
L_test = np.load(FEAT_CACHE / "cb600_test.npy")
P_test = np.load(FEAT_CACHE / "esm2_8M_test.npy")
X_train = F.concat(L_train, P_train)
X_test = F.concat(L_test, P_test)
y_train = train_df[LABEL_COL].values.astype(np.float32)
y_test = test_df[LABEL_COL].values.astype(np.float32)
X_val = X_train[:500]; y_val = y_train[:500]

CONFIGS = [
    {"name": "shallow", "hidden": [256], "dropout": 0.2, "lr": 5e-4},
    {"name": "medium", "hidden": [512, 256], "dropout": 0.2, "lr": 5e-4},
    {"name": "deep", "hidden": [512, 256, 128], "dropout": 0.2, "lr": 5e-4},
    {"name": "deep_hidrop", "hidden": [512, 256, 128], "dropout": 0.5, "lr": 5e-4},
    {"name": "deep_lowlr", "hidden": [512, 256, 128], "dropout": 0.2, "lr": 1e-4},
]
SEEDS = [42, 123, 456]

# Check what's already done
done = set()
results_csv = EXP_DIR / "results.csv"
if results_csv.exists():
    existing = pd.read_csv(results_csv)
    done = set(zip(existing["name"], existing["seed"]))
    log.info(f"Resuming: {len(done)} fits already done, {len(CONFIGS)*len(SEEDS)-len(done)} remaining")
    results = existing.to_dict("records")
else:
    results = []

for cfg in CONFIGS:
    for seed in SEEDS:
        if (cfg["name"], seed) in done:
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        model = MLP(input_dim=X_train.shape[1], hidden_dims=cfg["hidden"], dropout=cfg["dropout"])
        vm, tm, tt, ep, yp = train_torch(model, X_train, y_train, X_val, y_val, X_test, y_test,
                                           batch_size=128, lr=cfg["lr"], max_epochs=100, patience=20)
        rmse = float(np.sqrt(np.mean((yp - y_test)**2)))
        r = float(np.corrcoef(yp, y_test)[0, 1])
        row = {**cfg, "seed": seed, "rmse": rmse, "pearson_r": r,
               "epochs": ep, "time_s": time.time()-t0}
        results.append(row)
        log.info(f"  {cfg['name']:15s} seed={seed} RMSE={rmse:.4f} R={r:.4f} ep={ep}")
        # Save incrementally
        pd.DataFrame(results).to_csv(results_csv, index=False)

df = pd.DataFrame(results)
df.to_csv(results_csv, index=False)

log.info("\n=== SUMMARY ===")
for name in df["name"].unique():
    sub = df[df["name"] == name]
    log.info(f"  {name:15s}: RMSE={sub['rmse'].mean():.4f}±{sub['rmse'].std():.4f}  "
             f"best={sub['rmse'].min():.4f}  ep={sub['epochs'].mean():.0f}")

rf_rmse = 1.0065
log.info(f"\n  RF baseline: {rf_rmse:.4f}")
log.info(f"  Best DL: {df['rmse'].min():.4f} (gap: {df['rmse'].min()-rf_rmse:+.4f})")
