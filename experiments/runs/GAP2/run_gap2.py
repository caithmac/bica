"""GAP-2 — HPO Sensitivity. DL default vs tuned. Just 2 MLP fits."""
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
log = logging.getLogger("GAP2")

from harness.config import SMILES_COL, PROTEIN_COL, LABEL_COL
from harness.data import get_splits_for_seed
import harness.featurizers as F
from harness.trainer import train_torch, count_parameters
from models.mlp import MLP
import torch

train_df, val_df, test_df = get_splits_for_seed(42)

# ChemBERTa + ESM-2 features (cached from E201)
FEAT_CACHE = Path("E:/Drug Discovery/cache/features")
L_train = np.load(FEAT_CACHE / "cb600_train.npy")
L_test = np.load(FEAT_CACHE / "cb600_test.npy")
P_train = np.load(FEAT_CACHE / "esm2_8M_train.npy")
P_test = np.load(FEAT_CACHE / "esm2_8M_test.npy")
X_train = F.concat(L_train, P_train)
X_test = F.concat(L_test, P_test)
y_train = train_df[LABEL_COL].values.astype(np.float32)
y_test = test_df[LABEL_COL].values.astype(np.float32)
# Small val set
X_val = X_train[:500]; y_val = y_train[:500]
log.info(f"Train={X_train.shape} Test={X_test.shape}")

# Default MLP (matching E201)
log.info("Default MLP (lr=5e-4, patience=20)...")
t0 = time.time()
m_default = MLP(input_dim=X_train.shape[1], hidden_dims=[512, 256, 128], dropout=0.2)
vm, tm, tt, ep, yp = train_torch(m_default, X_train, y_train, X_val, y_val, X_test, y_test,
                                   batch_size=128, lr=5e-4, max_epochs=100, patience=20)
rmse_def = float(np.sqrt(np.mean((yp - y_test)**2)))
log.info(f"  Default: RMSE={rmse_def:.4f} epochs={ep} time={time.time()-t0:.0f}s")

# Tuned MLP (lower lr, more patience, weight decay)
log.info("Tuned MLP (lr=1e-4, wd=1e-3, patience=40)...")
t0 = time.time()
m_tuned = MLP(input_dim=X_train.shape[1], hidden_dims=[512, 256, 128], dropout=0.3)
vm2, tm2, tt2, ep2, yp2 = train_torch(m_tuned, X_train, y_train, X_val, y_val, X_test, y_test,
                                        batch_size=128, lr=1e-4, max_epochs=200, patience=40,
                                        weight_decay=1e-3)
rmse_tuned = float(np.sqrt(np.mean((yp2 - y_test)**2)))
log.info(f"  Tuned:   RMSE={rmse_tuned:.4f} epochs={ep2} time={time.time()-t0:.0f}s")

# RF baseline for reference
rf_rmse = 1.0065  # from E000
log.info(f"\nRF baseline: {rf_rmse:.4f}")
log.info(f"DL default: {rmse_def:.4f} (gap: {rmse_def-rf_rmse:+.4f})")
log.info(f"DL tuned:   {rmse_tuned:.4f} (gap: {rmse_tuned-rf_rmse:+.4f})")

pd.DataFrame([
    {"config": "default", "rmse": rmse_def, "epochs": ep, "lr": 5e-4, "patience": 20, "dropout": 0.2},
    {"config": "tuned", "rmse": rmse_tuned, "epochs": ep2, "lr": 1e-4, "patience": 40, "dropout": 0.3},
]).to_csv(EXP_DIR / "results.csv", index=False)
