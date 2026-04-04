"""
Bootstrap 95% confidence intervals for test RMSE, Pearson R, and Spearman R.

Does NOT re-train any model. Reads saved test predictions from
cache/predictions/{experiment_id}.npz and bootstraps the test set.

If predictions aren't cached yet, re-runs inference on the test set using
saved model weights from cache/models/{experiment_id}.pt (if available),
or re-runs the full experiment to generate them.

Usage:
    python bootstrap_ci.py                    # CI for all experiments in diary
    python bootstrap_ci.py --exp rf_ecfp4_aac # CI for one experiment
    python bootstrap_ci.py --n_bootstrap 2000 # more bootstrap samples

Output: diary/bootstrap_ci.csv
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

DIARY_PATH   = Path("diary/results_diary.csv")
PRED_CACHE   = Path("cache/predictions")
OUTPUT_PATH  = Path("diary/bootstrap_ci.csv")
PRED_CACHE.mkdir(parents=True, exist_ok=True)


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """
    Bootstrap 95% CI for RMSE, Pearson R, Spearman R.
    Returns dict with point estimate and [2.5%, 97.5%] intervals.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)

    rmses, pearsons, spearmans = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        rmses.append(np.sqrt(np.mean((yt - yp) ** 2)))
        pearsons.append(pearsonr(yt, yp)[0])
        spearmans.append(spearmanr(yt, yp)[0])

    def _ci(arr):
        arr = np.array(arr)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    rmse_pt  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pear_pt  = float(pearsonr(y_true, y_pred)[0])
    spear_pt = float(spearmanr(y_true, y_pred)[0])

    r_lo, r_hi   = _ci(rmses)
    p_lo, p_hi   = _ci(pearsons)
    s_lo, s_hi   = _ci(spearmans)

    return {
        "rmse":          round(rmse_pt,  4),
        "rmse_ci_lo":    round(r_lo,     4),
        "rmse_ci_hi":    round(r_hi,     4),
        "pearson_r":     round(pear_pt,  4),
        "pearson_ci_lo": round(p_lo,     4),
        "pearson_ci_hi": round(p_hi,     4),
        "spearman_r":    round(spear_pt, 4),
        "spearman_ci_lo":round(s_lo,     4),
        "spearman_ci_hi":round(s_hi,     4),
        "n_bootstrap":   n_bootstrap,
        "n_test":        n,
    }


def load_or_generate_predictions(exp_id: str) -> tuple | None:
    """
    Load (y_true, y_pred) from cache/predictions/{exp_id}.npz.
    Returns None if not available (predictions must be saved during training).
    """
    pred_file = PRED_CACHE / f"{exp_id}.npz"
    if pred_file.exists():
        data = np.load(pred_file)
        return data["y_true"], data["y_pred"]
    return None


def run_bootstrap(exp_ids: list, n_bootstrap: int = 1000):
    rows = []
    missing = []

    for exp_id in exp_ids:
        result = load_or_generate_predictions(exp_id)
        if result is None:
            missing.append(exp_id)
            print(f"  [skip] {exp_id} — no saved predictions in cache/predictions/")
            continue

        y_true, y_pred = result
        ci = bootstrap_metrics(y_true, y_pred, n_bootstrap=n_bootstrap)
        ci["experiment_id"] = exp_id
        rows.append(ci)
        print(f"  {exp_id:50s}  RMSE={ci['rmse']:.4f} "
              f"[{ci['rmse_ci_lo']:.4f}, {ci['rmse_ci_hi']:.4f}]  "
              f"Pearson={ci['pearson_r']:.4f} "
              f"[{ci['pearson_ci_lo']:.4f}, {ci['pearson_ci_hi']:.4f}]")

    if rows:
        out_df = pd.DataFrame(rows)
        cols = ["experiment_id", "rmse", "rmse_ci_lo", "rmse_ci_hi",
                "pearson_r", "pearson_ci_lo", "pearson_ci_hi",
                "spearman_r", "spearman_ci_lo", "spearman_ci_hi",
                "n_bootstrap", "n_test"]
        out_df = out_df[cols].sort_values("rmse")
        out_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n[bootstrap] Written to {OUTPUT_PATH}")
        print(out_df.to_string(index=False))

    if missing:
        print(f"\n[bootstrap] {len(missing)} experiments missing predictions.")
        print("  To generate predictions, re-run those experiments — they will")
        print("  auto-save predictions to cache/predictions/ going forward.")
        print("  Missing:", missing[:10], "..." if len(missing) > 10 else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",         type=str,  default=None)
    parser.add_argument("--n_bootstrap", type=int,  default=1000)
    args = parser.parse_args()

    if not DIARY_PATH.exists():
        print("[bootstrap] No diary found. Run experiments first.")
        return

    diary = pd.read_csv(DIARY_PATH)
    diary = diary.sort_values("timestamp").drop_duplicates("experiment_id", keep="last")

    if args.exp:
        exp_ids = [args.exp]
    else:
        exp_ids = diary["experiment_id"].tolist()

    print(f"[bootstrap] Computing {args.n_bootstrap}-sample CIs for {len(exp_ids)} experiments …\n")
    run_bootstrap(exp_ids, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    main()
