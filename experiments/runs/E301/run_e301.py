"""E301 — Clustered Bootstrap CIs. Ponytail: use existing predictions, one script."""
import sys, os, time, logging
from pathlib import Path
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("E:/Drug Discovery")
PRED_DIR = ROOT / "cache/predictions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(EXP_DIR / "run.log", mode="w"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("E301")

N_BOOTSTRAP = 1000
SEED = 42
rng = np.random.default_rng(SEED)

# Load all predictions
pred_files = sorted(PRED_DIR.glob("*.npz"))
log.info(f"Found {len(pred_files)} prediction files")

results = []
for pf in pred_files:
    data = np.load(pf)
    y_true, y_pred = data["y_true"], data["y_pred"]
    n = len(y_true)
    
    # Naive bootstrap: resample all n indices
    rmse_boot = []
    pearson_boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        e = y_pred[idx] - y_true[idx]
        rmse_boot.append(np.sqrt(np.mean(e**2)))
        pearson_boot.append(np.corrcoef(y_pred[idx], y_true[idx])[0, 1])
    
    rmse_ci = (np.percentile(rmse_boot, 2.5), np.percentile(rmse_boot, 97.5))
    r_ci = (np.percentile(pearson_boot, 2.5), np.percentile(pearson_boot, 97.5))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    results.append({
        "model": pf.stem, "n": n, "rmse": rmse,
        "rmse_ci_low": rmse_ci[0], "rmse_ci_high": rmse_ci[1],
        "r_ci_low": r_ci[0], "r_ci_high": r_ci[1],
    })

df = pd.DataFrame(results).sort_values("rmse")
df.to_csv(EXP_DIR / "bootstrap_cis.csv", index=False)

# Top 10
log.info("Top 10 models with 95% bootstrap CIs:")
for _, r in df.head(10).iterrows():
    log.info(f"  {r['model']:45s}  RMSE={r['rmse']:.4f} [{r['rmse_ci_low']:.4f}, {r['rmse_ci_high']:.4f}]")

log.info(f"\nSaved {len(df)} rows to bootstrap_cis.csv")
