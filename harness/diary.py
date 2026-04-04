"""
Experiment diary — appends results to a structured CSV.
Never overwrites existing entries.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

from harness.config import DIARY_PATH

FIELDNAMES = [
    # Identification
    "timestamp",
    "experiment_id",
    "model_name",
    "model_family",          # e.g. "linear", "tree", "mlp", "gnn", "transformer"
    # Representation
    "ligand_repr",           # e.g. "ecfp4_1024", "chemberta", "graph"
    "protein_repr",          # e.g. "esm2_320", "aac_580", "kmer3"
    "fusion_strategy",       # e.g. "concat", "cross_attention"
    # Hyperparameters (key ones)
    "n_params",              # trainable parameter count (or "N/A")
    "epochs_trained",
    "batch_size",
    "learning_rate",
    # Split info
    "split_type",
    "n_train",
    "n_val",
    "n_test",
    # Results — Validation set
    "val_rmse",
    "val_pearson_r",
    "val_spearman_r",
    # Results — Test set
    "test_rmse",
    "test_pearson_r",
    "test_spearman_r",
    # Timing
    "train_time_sec",
    # Notes
    "notes",
]


def save_predictions(experiment_id: str, y_true, y_pred):
    """
    Cache test-set predictions to cache/predictions/{experiment_id}.npz
    so bootstrap_ci.py can compute CIs without re-running the experiment.
    """
    import numpy as np
    pred_dir = DIARY_PATH.parent.parent / "cache" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f"{experiment_id}.npz",
             y_true=np.array(y_true, dtype=np.float32),
             y_pred=np.array(y_pred, dtype=np.float32))


def log_result(
    *,
    experiment_id: str,
    model_name: str,
    model_family: str,
    ligand_repr: str,
    protein_repr: str,
    fusion_strategy: str,
    n_params,
    epochs_trained,
    batch_size,
    learning_rate,
    split_type: str,
    n_train: int,
    n_val: int,
    n_test: int,
    val_metrics: dict,
    test_metrics: dict,
    train_time_sec: float,
    notes: str = "",
):
    """Append one row to the CSV diary. Creates the file with headers if new."""
    DIARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = DIARY_PATH.exists() and DIARY_PATH.stat().st_size > 0

    row = {
        "timestamp":       datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "experiment_id":   experiment_id,
        "model_name":      model_name,
        "model_family":    model_family,
        "ligand_repr":     ligand_repr,
        "protein_repr":    protein_repr,
        "fusion_strategy": fusion_strategy,
        "n_params":        n_params,
        "epochs_trained":  epochs_trained,
        "batch_size":      batch_size,
        "learning_rate":   learning_rate,
        "split_type":      split_type,
        "n_train":         n_train,
        "n_val":           n_val,
        "n_test":          n_test,
        "val_rmse":        val_metrics["rmse"],
        "val_pearson_r":   val_metrics["pearson_r"],
        "val_spearman_r":  val_metrics["spearman_r"],
        "test_rmse":       test_metrics["rmse"],
        "test_pearson_r":  test_metrics["pearson_r"],
        "test_spearman_r": test_metrics["spearman_r"],
        "train_time_sec":  round(train_time_sec, 1),
        "notes":           notes,
    }

    with open(DIARY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[diary] Logged: {experiment_id} | test_rmse={test_metrics['rmse']:.4f} "
          f"pearson={test_metrics['pearson_r']:.4f} spearman={test_metrics['spearman_r']:.4f}")


def print_leaderboard():
    """Print current leaderboard sorted by test RMSE."""
    if not DIARY_PATH.exists():
        print("[diary] No results yet.")
        return

    import pandas as pd
    df = pd.read_csv(DIARY_PATH)
    if df.empty:
        print("[diary] No results yet.")
        return

    cols = ["experiment_id", "model_name", "ligand_repr", "protein_repr",
            "test_rmse", "test_pearson_r", "test_spearman_r", "train_time_sec"]
    df_show = df[cols].sort_values("test_rmse")
    print("\n=== LEADERBOARD (sorted by test RMSE ↑ = worse) ===")
    print(df_show.to_string(index=False))
    print()
