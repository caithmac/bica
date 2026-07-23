#!/usr/bin/env python3
"""Phase 6: PSICHIC reproduction on PDBBind-v2020 + transfer to BindingDB splits.
Requires PSICHIC submodule at E:/BICA/PSICHIC.
"""
import json, os, subprocess, sys, time, shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

PYTHON = "C:/Users/ss864/AppData/Local/miniconda3/envs/drug_discovery/python.exe"
PSICHIC_DIR = Path("E:/BICA/PSICHIC")
DATA_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/psichic")
RESULTS_DIR = Path("E:/Drug Discovery/projects/balm-revision/results/psichic")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load splits ---
SPLIT_TYPES = ['random', 'scaffold', 'cold_target']
SEED = 42
OUTPUTS = {}

for split_type in SPLIT_TYPES:
    split_dir = Path(f"E:/Drug Discovery/projects/balm-revision/data/splits/{split_type}/seed_{SEED}")
    
    if not split_dir.exists():
        print(f"  SKIP {split_type} — no split data")
        continue
    
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    
    # PSICHIC expects: Ligand (SMILES), Protein (sequence), regression_label (pKd)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        psichic_df = pd.DataFrame({
            'Ligand': df['Drug_canonical'],
            'Protein': df['Target'],
            'regression_label': df['Y'],
        })
        out_dir = DATA_DIR / f"{split_type}_seed{SEED}"
        os.makedirs(out_dir, exist_ok=True)
        psichic_df.to_csv(out_dir / f"{name}.csv", index=False)
    
    print(f"  Prepared {split_type}: {len(train_df)}/{len(val_df)}/{len(test_df)}")

# --- PSICHIC zero-shot on BindingDB splits ---
print("\n--- PSICHIC Zero-Shot ---")

CKPT_PATH = PSICHIC_DIR / "trained_weights/PDBv2020_PSICHIC"

for split_type in SPLIT_TYPES:
    datafolder = DATA_DIR / f"{split_type}_seed{SEED}"
    result_json = RESULTS_DIR / f"zero_shot_{split_type}.json"
    
    print(f"  {split_type}...")
    
    cmd = [
        PYTHON, str(PSICHIC_DIR / "psichic_runner.py"),
        "--mode", "zero_shot",
        "--train_csv", str(datafolder / "train.csv"),
        "--val_csv", str(datafolder / "val.csv"),
        "--test_csv", str(datafolder / "test.csv"),
        "--ckpt_path", str(CKPT_PATH),
        "--result_json", str(result_json),
        "--device", "cuda",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    if result.returncode == 0 and result_json.exists():
        with open(result_json) as f:
            data = json.load(f)
        OUTPUTS[f"zero_shot_{split_type}"] = data
        print(f"    RMSE={data.get('rmse', '?'):.4f}, Pearson={data.get('pearson_r', '?'):.4f}")
    else:
        print(f"    FAILED: {result.stderr[-500:]}")
        # Try with cached features
        print(f"    Attempting with cached features...")
        prot_cache = DATA_DIR / "psichic_protein_feats.pt"
        lig_cache = DATA_DIR / "psichic_ligand_feats.pkl"
        
        cmd2 = cmd + [
            "--prot_cache", str(prot_cache) if prot_cache.exists() else "",
            "--lig_cache", str(lig_cache) if lig_cache.exists() else "",
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=3600)
        if result2.returncode == 0:
            with open(result_json) as f:
                data = json.load(f)
            OUTPUTS[f"zero_shot_{split_type}"] = data
            print(f"    RMSE={data.get('rmse', '?'):.4f}, Pearson={data.get('pearson_r', '?'):.4f}")
        else:
            print(f"    FAILED again: {result2.stderr[-300:]}")

# --- PSICHIC fine-tune on scaffold split ---
print("\n--- PSICHIC Fine-Tune ---")

for split_type in SPLIT_TYPES:
    datafolder = DATA_DIR / f"{split_type}_seed{SEED}"
    result_path = RESULTS_DIR / f"finetune_{split_type}"
    os.makedirs(result_path, exist_ok=True)
    
    print(f"  {split_type} (5000 iters, lr=1e-5)...")
    
    cmd = [
        PYTHON, str(PSICHIC_DIR / "main.py"),
        "--datafolder", str(datafolder),
        "--result_path", str(result_path) + "/",
        "--config_path", str(PSICHIC_DIR / "trained_weights/PDBv2020_PSICHIC/config.json"),
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
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)  # 4h timeout
    
    full_result = result_path / "full_result-1.txt"
    if result.returncode == 0 and full_result.exists():
        # Parse final test result
        with open(full_result) as f:
            content = f.read()
        
        # Extract last test RMSE
        import re
        test_rmses = re.findall(r'"rmse": ([\d.]+)', content)
        test_pearsons = re.findall(r'"pearson_r": ([\d.]+)', content)
        
        print(f"    Final test RMSE: {test_rmses[-1] if test_rmses else '?'}")
        print(f"    Final test Pearson: {test_pearsons[-1] if test_pearsons else '?'}")
        print(f"    Learning curve: {full_result}")
    else:
        print(f"    FAILED: {result.stderr[-300:] if result.stderr else 'timeout/unknown'}")

# --- Save summary ---
summary = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "psichic_checkpoint": str(CKPT_PATH),
    "results": OUTPUTS,
}

with open(RESULTS_DIR / "phase6_summary.json", 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n[Phase 6] COMPLETE — Results: {RESULTS_DIR}/phase6_summary.json")
