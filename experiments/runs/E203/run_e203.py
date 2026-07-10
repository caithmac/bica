"""E203 — PSICHIC Diagnostic. Scaffold vs random split, track train/val RMSE."""
import sys, os, time, logging, subprocess, json
from pathlib import Path
import numpy as np, pandas as pd

EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("E:/Drug Discovery")
PSICHIC_DIR = ROOT / "PSICHIC"
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(EXP_DIR / "run.log", mode="w"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("E203")

from harness.data import get_splits, load_raw
import harness.featurizers as F

# ── Generate splits ────────────────────────────────────────────────────────────
# 1. Scaffold split (seed 42) — existing
train_s, val_s, test_s = get_splits()

# 2. Random split (seed 42)
df_raw = load_raw()
rng = np.random.default_rng(42)
idx = rng.permutation(len(df_raw))
t_end = int(0.7 * len(df_raw))
v_end = int(0.8 * len(df_raw))
train_r = df_raw.iloc[idx[:t_end]].reset_index(drop=True)
val_r = df_raw.iloc[idx[t_end:v_end]].reset_index(drop=True)
test_r = df_raw.iloc[idx[v_end:]].reset_index(drop=True)

log.info(f"Scaffold: train={len(train_s):,} val={len(val_s):,} test={len(test_s):,}")
log.info(f"Random:   train={len(train_r):,} val={len(val_r):,} test={len(test_r):,}")

# ── PSICHIC format: Ligand, Protein, regression_label ──────────────────────────
def to_psichic(df):
    return (df.rename(columns={"Drug": "Ligand", "Target": "Protein", "Y": "regression_label"})
            [["Ligand", "Protein", "regression_label"]].reset_index(drop=True))

# Save split CSVs
splits_dir = EXP_DIR / "splits"
splits_dir.mkdir(exist_ok=True)
for name, df in [("scaffold_train", train_s), ("scaffold_val", val_s), ("scaffold_test", test_s),
                  ("random_train", train_r), ("random_val", val_r), ("random_test", test_r)]:
    to_psichic(df).to_csv(splits_dir / f"{name}.csv", index=False)

# ── Run PSICHIC on both splits ─────────────────────────────────────────────────
CKPT = str(PSICHIC_DIR / "trained_weights/PDBv2020_PSICHIC")
CACHE_DIR = str(ROOT / "cache")
PROT_CACHE = CACHE_DIR + "/psichic_protein_feats.pt"
LIG_CACHE = CACHE_DIR + "/psichic_ligand_feats.pkl"

results = []

for split_name in ["scaffold", "random"]:
    log.info(f"\n{'='*50}\n  PSICHIC — {split_name} split\n{'='*50}")
    
    result_json = str(EXP_DIR / f"psichic_{split_name}_result.json")
    pred_npy = str(EXP_DIR / f"psichic_{split_name}_preds.npy")
    
    cmd = [
        sys.executable,
        str(PSICHIC_DIR / "psichic_runner.py"),
        "--train_csv", str(splits_dir / f"{split_name}_train.csv"),
        "--val_csv", str(splits_dir / f"{split_name}_val.csv"),
        "--test_csv", str(splits_dir / f"{split_name}_test.csv"),
        "--prot_cache", PROT_CACHE,
        "--lig_cache", LIG_CACHE,
        "--ckpt_path", CKPT,
        "--result_json", result_json,
        "--pred_npy", pred_npy,
        "--mode", "fine_tune",
        "--device", "cuda",
        "--batch_size", "64",
        "--ft_iters", "100",
        "--ft_lr", "1e-4",
    ]
    
    log.info(f"  Running PSICHIC fine_tune on {split_name} split...")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(PSICHIC_DIR))
    
    if proc.returncode == 0:
        with open(result_json) as f:
            result = json.load(f)
        results.append({"split": split_name, **result})
        log.info(f"  {split_name}: RMSE={result.get('test_rmse','?'):.4f}  "
                 f"R={result.get('test_pearson_r','?'):.4f}")
    else:
        log.error(f"  PSICHIC failed on {split_name}: {proc.stderr[:500]}")
    log.info(f"  Time: {(time.time()-t0)/60:.1f} min")

pd.DataFrame(results).to_csv(EXP_DIR / "results.csv", index=False)

# Compare
if len(results) == 2:
    scf = results[0]["test_rmse"]
    rnd = results[1]["test_rmse"]
    log.info(f"\nPSICHIC scaffold: {scf:.4f}  random: {rnd:.4f}  Δ={scf-rnd:+.4f}")
    if rnd < scf - 0.1:
        log.warning("MEMORIZATION DETECTED: random split RMSE significantly lower than scaffold!")
