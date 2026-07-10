"""
E203 v2 — PSICHIC learning curves with per-step logging.

Uses PSICHIC's native main.py pipeline which writes train/val/test metrics
to full_result-{seed}.txt at every evaluate_step interval.

Runs both scaffold and random splits with evaluate_step=200 for
25 data points over 5000 iterations.
"""
import sys, os, time, logging, subprocess, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("E:/Drug Discovery")
PSICHIC_DIR = ROOT / "PSICHIC"
CKPT_PATH = PSICHIC_DIR / "trained_weights/PDBv2020_PSICHIC"
SPLITS_DIR = Path("E:/Drug Discovery/experiments/runs/E203/splits")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(EXP_DIR / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("E203v2")

# ── Setup datafolders for main.py ──────────────────────────────────────────────
# main.py expects: datafolder/{train.csv, test.csv, valid.csv}
# And will auto-compute protein.pt + ligand.pt on first run.

for split_name in ["scaffold", "random"]:
    data_dir = EXP_DIR / f"{split_name}_data"
    data_dir.mkdir(exist_ok=True)
    for subset in ["train", "val", "test"]:
        src = SPLITS_DIR / f"{split_name}_{subset}.csv"
        # main.py looks for valid.csv (not val.csv)
        dst_name = "valid.csv" if subset == "val" else f"{subset}.csv"
        dst = data_dir / dst_name
        shutil.copy(src, dst)
    log.info(f"Prepared {split_name}_data/ with train.csv, val.csv, test.csv")

# ── Run PSICHIC main.py for both splits ────────────────────────────────────────
PYTHON = "C:/Users/ss864/AppData/Local/miniconda3/envs/drug_discovery/python.exe"

for split_name in ["scaffold", "random"]:
    log.info(f"\n{'='*60}\n  PSICHIC main.py — {split_name} split\n{'='*60}")
    
    data_dir = EXP_DIR / f"{split_name}_data"
    result_dir = EXP_DIR / f"{split_name}_result"
    result_dir.mkdir(exist_ok=True)
    
    cmd = [
        PYTHON,
        str(PSICHIC_DIR / "main.py"),
        "--datafolder", str(data_dir),
        "--result_path", str(result_dir) + "/",
        "--config_path", str(CKPT_PATH / "config.json"),
        "--trained_model_path", str(CKPT_PATH),
        "--regression_task", "True",
        "--total_iters", "5000",
        "--evaluate_step", "200",
        "--lrate", "1e-5",
        "--batch_size", "16",
        "--seed", "1",
        "--save_interpret", "False",
        "--device", "cuda",
    ]
    
    log.info(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=7200, cwd=str(PSICHIC_DIR),
        env={**os.environ, "PYTHONPATH": str(PSICHIC_DIR)},
    )
    
    elapsed = time.time() - t0
    
    # Save stdout/stderr
    (EXP_DIR / f"{split_name}_stdout.txt").write_text(proc.stdout)
    (EXP_DIR / f"{split_name}_stderr.txt").write_text(proc.stderr)
    
    if proc.returncode != 0:
        log.error(f"  PSICHIC failed on {split_name}: rc={proc.returncode}")
        log.error(f"  stderr tail: {proc.stderr[-1000:]}")
    else:
        log.info(f"  Completed in {elapsed/60:.1f} min")
    
    # Parse full_result
    full_result_path = result_dir / "full_result-1.txt"
    if full_result_path.exists():
        log.info(f"  full_result-1.txt: {full_result_path.stat().st_size:,} bytes")
    else:
        log.warning(f"  full_result-1.txt NOT FOUND at {full_result_path}")

log.info("\nE203 v2 complete")
