"""
Evaluation metrics used across all experiments.
All models must report: RMSE, Pearson R, Spearman R.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics for binding affinity prediction.

    Returns
    -------
    dict with keys: rmse, pearson_r, pearson_p, spearman_r, spearman_p
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    assert len(y_true) == len(y_pred), "Length mismatch"

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)

    return {
        "rmse":       round(rmse, 4),
        "pearson_r":  round(float(pr), 4),
        "pearson_p":  round(float(pp), 6),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp), 6),
        "n_samples":  len(y_true),
    }


def format_metrics(metrics: dict) -> str:
    return (f"RMSE={metrics['rmse']:.4f}  "
            f"Pearson={metrics['pearson_r']:.4f}  "
            f"Spearman={metrics['spearman_r']:.4f}  "
            f"(n={metrics['n_samples']:,})")
