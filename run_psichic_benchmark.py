"""
run_psichic_benchmark.py
========================
Evaluate PSICHIC on BindingDB_filtered using the same Bemis-Murcko scaffold
split (seed 42) as the rest of the benchmark.

How it works
------------
PSICHIC's models/ package clashes with our own models/ package, so we run
PSICHIC inside its own directory as a subprocess (PSICHIC/psichic_runner.py).
Results are returned via a JSON file and logged to the benchmark diary.

Two modes
---------
  zero_shot  – pretrained PDBv2020 checkpoint, no fine-tuning  (~minutes)
  fine_tune  – fine-tune from PDBv2020 on train split, evaluate on test (~hours)

Usage
-----
    conda run -n drug_discovery python run_psichic_benchmark.py --mode zero_shot
    conda run -n drug_discovery python run_psichic_benchmark.py --mode fine_tune
    conda run -n drug_discovery python run_psichic_benchmark.py --mode both
"""

import os, sys, json, subprocess, time, tempfile
import numpy as np
import pandas as pd
import argparse

ROOT      = os.path.dirname(os.path.abspath(__file__))
PSICHIC   = os.path.join(ROOT, "PSICHIC")
CACHE_DIR = os.path.join(ROOT, "cache")
PRED_DIR  = os.path.join(CACHE_DIR, "predictions")
CKPT_BASE = os.path.join(PSICHIC, "trained_weights", "PDBv2020_PSICHIC")
FT_OUT    = os.path.join(CACHE_DIR, "psichic_finetuned")
PROT_CACHE = os.path.join(CACHE_DIR, "psichic_protein_feats.pt")
LIG_CACHE  = os.path.join(CACHE_DIR, "psichic_ligand_feats.pkl")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FT_OUT,   exist_ok=True)

sys.path.insert(0, ROOT)
from harness.data    import get_splits
from harness.diary   import log_result, save_predictions


def _df_to_psichic(df: pd.DataFrame) -> pd.DataFrame:
    return (df.rename(columns={"Drug": "Ligand", "Target": "Protein", "Y": "regression_label"})
              [["Ligand", "Protein", "regression_label"]]
              .reset_index(drop=True))


def _run_mode(mode: str, device: str, batch_size: int,
              ft_iters: int, ft_lr: float,
              train_csv: str, val_csv: str, test_csv: str,
              n_train: int, n_val: int, n_test: int) -> dict:
    """Run PSICHIC runner subprocess, return result dict."""

    result_json = os.path.join(CACHE_DIR, f"psichic_{mode}_result.json")
    pred_npy    = os.path.join(PRED_DIR,  f"psichic_{mode}_preds.npy")
    ft_model_out = os.path.join(FT_OUT, "model_ft.pt") if mode == "fine_tune" else ""

    cmd = [
        sys.executable,
        os.path.join(PSICHIC, "psichic_runner.py"),
        "--train_csv",    train_csv,
        "--val_csv",      val_csv,
        "--test_csv",     test_csv,
        "--prot_cache",   PROT_CACHE,
        "--lig_cache",    LIG_CACHE,
        "--ckpt_path",    CKPT_BASE,
        "--result_json",  result_json,
        "--pred_npy",     pred_npy,
        "--mode",         mode,
        "--device",       device,
        "--batch_size",   str(batch_size),
        "--ft_iters",     str(ft_iters),
        "--ft_lr",        str(ft_lr),
        "--ft_model_out", ft_model_out,
    ]

    print(f"\n[PSICHIC] Launching {mode} subprocess …")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=PSICHIC)   # run from inside PSICHIC/
    elapsed = time.time() - t0

    if ret.returncode != 0:
        raise RuntimeError(f"PSICHIC runner failed (exit {ret.returncode})")

    with open(result_json) as f:
        result = json.load(f)

    # Load predictions for diary
    arr    = np.load(pred_npy)
    y_true = arr[:, 0]
    y_pred = arr[:, 1]

    # Experiment ID and logging
    exp_id = f"psichic_{mode}"
    log_result(
        experiment_id   = exp_id,
        model_name      = exp_id,
        model_family    = "psichic",
        ligand_repr     = "psichic_graph",
        protein_repr    = "esm2_650M_contact",
        fusion_strategy = "psichic_gnn",
        n_params        = result["n_params"],
        epochs_trained  = result["steps_trained"],
        batch_size      = result["batch_size"],
        learning_rate   = result["learning_rate"],
        split_type      = "scaffold_bemis_murcko",
        n_train         = n_train,
        n_val           = n_val,
        n_test          = n_test,
        val_metrics     = result["val_metrics"],
        test_metrics    = result["test_metrics"],
        train_time_sec  = result["train_time_sec"],
        notes           = (
            "PSICHIC pretrained PDBBind-v2020, zero-shot on BindingDB scaffold split"
            if mode == "zero_shot" else
            f"PSICHIC fine-tuned from PDBv2020; {result['steps_trained']} iters "
            f"lr={result['learning_rate']} on BindingDB scaffold split"
        ),
    )
    save_predictions(exp_id, y_true, y_pred)
    print(f"  → Logged '{exp_id}'  "
          f"RMSE={result['test_metrics']['rmse']:.4f}  "
          f"Pearson={result['test_metrics']['pearson_r']:.4f}  "
          f"Spearman={result['test_metrics']['spearman_r']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",       default="both",
                    choices=["zero_shot", "fine_tune", "both"])
    ap.add_argument("--device",     default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--ft_iters",   type=int, default=5000)
    ap.add_argument("--ft_lr",      type=float, default=1e-5)
    args = ap.parse_args()

    # ── 1. Load and write splits to temp CSVs ─────────────────────────────────
    print("[PSICHIC] Loading BindingDB_filtered scaffold splits …")
    train_df, val_df, test_df = get_splits()
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_csv = os.path.join(tmpdir, "train.csv")
        val_csv   = os.path.join(tmpdir, "val.csv")
        test_csv  = os.path.join(tmpdir, "test.csv")
        _df_to_psichic(train_df).to_csv(train_csv, index=False)
        _df_to_psichic(val_df  ).to_csv(val_csv,   index=False)
        _df_to_psichic(test_df ).to_csv(test_csv,  index=False)

        modes = (["zero_shot", "fine_tune"] if args.mode == "both"
                 else [args.mode])

        for mode in modes:
            _run_mode(
                mode       = mode,
                device     = args.device,
                batch_size = args.batch_size,
                ft_iters   = args.ft_iters,
                ft_lr      = args.ft_lr,
                train_csv  = train_csv,
                val_csv    = val_csv,
                test_csv   = test_csv,
                n_train    = len(train_df),
                n_val      = len(val_df),
                n_test     = len(test_df),
            )

    print("\n[PSICHIC] All done. Results logged to diary/results_diary.csv")


if __name__ == "__main__":
    main()
