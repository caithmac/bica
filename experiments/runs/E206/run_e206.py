"""
E206 — Enrichment / Ranking Analysis
=====================================
Post-hoc analysis: does the RMSE gap between RF and DL matter for virtual screening?
Results from saved E000 predictions.

Metrics:
  - Enrichment Factor at top 1%, 5%, 10% (active = pKd >= 7.0 and >= 6.0)
  - Kendall τ between model rankings
  - "In a top-100 virtual screen, RF finds X more actives than DL"

Run: python experiments/runs/E206/run_e206.py
CPU-only, post-hoc on E000 predictions
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("E:/Drug Discovery")
PRED_DIR = ROOT / "cache/predictions"
DIARY_PATH = ROOT / "diary/results_diary.csv"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(EXP_DIR / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("E206")

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "experiment": "E206",
    "description": "Enrichment and ranking analysis",
    "activity_thresholds": [6.0, 7.0],  # pKd >= 6 = Kd <= 1µM; pKd >= 7 = Kd <= 100nM
    "top_percentiles": [1, 5, 10],
    "timestamp": datetime.now().isoformat(),
}

with open(EXP_DIR / "config.yaml", "w") as f:
    yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)


# ── Load results diary ─────────────────────────────────────────────────────────
diary = pd.read_csv(DIARY_PATH)
log.info(f"Loaded {len(diary)} entries from results diary")

# Select representative models for comparison
# Best tree model + best DL model on scaffold split
model_pairs = [
    # (model_id, label, category)
    ("rf_ecfp4_aac", "RF + ECFP4 + AAC", "tree_best"),
    ("xgb_ecfp4_aac", "XGBoost + ECFP4 + AAC", "tree"),
    ("lgbm_ecfp4_aac", "LightGBM + ECFP4 + AAC", "tree"),
    ("mlp_chemberta_esm2_8M", "MLP + ChemBERTa-5M + ESM-2-8M", "dl"),
    ("xgb_chemberta_5M_esm2_650M", "XGBoost + ChemBERTa + ESM-2-650M", "tree_pretrained"),
    ("bica_chemberta_esm2_8M", "BiCA + ChemBERTa + ESM-2-8M", "dl"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load predictions
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 1: Loading predictions")
log.info("=" * 70)

predictions = {}
y_true = None

for model_id, label, category in model_pairs:
    pred_file = PRED_DIR / f"{model_id}.npz"
    if pred_file.exists():
        data = np.load(pred_file)
        if y_true is None:
            y_true = data["y_true"]
        predictions[model_id] = {
            "label": label,
            "category": category,
            "y_pred": data["y_pred"],
            "y_true": data["y_true"],
        }
        rmse = np.sqrt(np.mean((data["y_pred"] - data["y_true"])**2))
        r = np.corrcoef(data["y_pred"], data["y_true"])[0, 1]
        log.info(f"  {label:40s}  RMSE={rmse:.4f}  R={r:.4f}")
    else:
        log.warning(f"  {model_id}.npz not found — skipping")

# Verify all models use the same y_true
for mid, pred in predictions.items():
    assert np.allclose(pred["y_true"], y_true), f"{mid} has different y_true!"
log.info(f"\n  All predictions verified: same test set ({len(y_true):,} compounds)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Compute enrichment factors
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 2: Enrichment factor computation")
log.info("=" * 70)

enrichment_rows = []

for threshold in CONFIG["activity_thresholds"]:
    n_active_total = int(np.sum(y_true >= threshold))
    active_rate = n_active_total / len(y_true)
    log.info(f"\n  Active threshold: pKd ≥ {threshold} → {n_active_total} actives ({active_rate:.1%})")
    
    for model_id, pred in predictions.items():
        y_pred = pred["y_pred"]
        n = len(y_pred)
        
        # Rank compounds by predicted pKd (descending)
        ranked_indices = np.argsort(-y_pred)  # highest first
        
        for top_pct in CONFIG["top_percentiles"]:
            top_n = max(1, int(n * top_pct / 100))
            top_indices = ranked_indices[:top_n]
            n_actives_found = int(np.sum(y_true[top_indices] >= threshold))
            
            # Enrichment factor: (actives_in_top / total_in_top) / (total_actives / total_all)
            observed_rate = n_actives_found / top_n
            ef = observed_rate / active_rate if active_rate > 0 else 0.0
            
            enrichment_rows.append({
                "model_id": model_id,
                "label": pred["label"],
                "category": pred["category"],
                "activity_threshold": threshold,
                "top_percentile": top_pct,
                "top_n": top_n,
                "n_actives_found": n_actives_found,
                "enrichment_factor": round(ef, 3),
                "observed_rate": round(observed_rate, 4),
            })

enrichment_df = pd.DataFrame(enrichment_rows)
enrichment_df.to_csv(EXP_DIR / "enrichment.csv", index=False)

# Log the key comparison
log.info(f"\n{'='*50}")
log.info("KEY COMPARISON: RF vs DL at top-5%")
log.info(f"{'='*50}")
for threshold in CONFIG["activity_thresholds"]:
    subset = enrichment_df[(enrichment_df["activity_threshold"] == threshold) & 
                           (enrichment_df["top_percentile"] == 5)]
    rf_row = subset[subset["model_id"] == "rf_ecfp4_aac"]
    dl_row = subset[subset["model_id"] == "mlp_chemberta_esm2_8M"]
    
    if len(rf_row) > 0 and len(dl_row) > 0:
        rf_ef = rf_row["enrichment_factor"].values[0]
        dl_ef = dl_row["enrichment_factor"].values[0]
        rf_actives = rf_row["n_actives_found"].values[0]
        dl_actives = dl_row["n_actives_found"].values[0]
        delta = rf_actives - dl_actives
        log.info(f"  pKd ≥ {threshold}: RF EF={rf_ef:.2f} vs DL EF={dl_ef:.2f}")
        log.info(f"    RF finds {delta} more actives in top-5% ({rf_actives} vs {dl_actives})")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Kendall τ between model rankings
# ═══════════════════════════════════════════════════════════════════════════════
log.info(f"\n{'='*50}")
log.info("STEP 3: Kendall τ between rankings")
log.info(f"{'='*50}")

model_ids = list(predictions.keys())
kendall_rows = []

for i, mid_a in enumerate(model_ids):
    for j, mid_b in enumerate(model_ids):
        if i >= j:
            continue
        tau, pval = kendalltau(predictions[mid_a]["y_pred"], predictions[mid_b]["y_pred"])
        kendall_rows.append({
            "model_a": mid_a,
            "model_b": mid_b,
            "label_a": predictions[mid_a]["label"],
            "label_b": predictions[mid_b]["label"],
            "kendall_tau": round(tau, 4),
            "p_value": pval,
        })
        log.info(f"  {mid_a:30s} ↔ {mid_b:30s}  τ={tau:.4f}")

kendall_df = pd.DataFrame(kendall_rows)
kendall_df.to_csv(EXP_DIR / "kendall_tau.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Missed actives — compounds RF finds that DL misses
# ═══════════════════════════════════════════════════════════════════════════════
log.info(f"\n{'='*50}")
log.info("STEP 4: Missed actives (RF finds, DL misses)")
log.info(f"{'='*50}")

rf_pred = predictions["rf_ecfp4_aac"]["y_pred"]
dl_pred = predictions["mlp_chemberta_esm2_8M"]["y_pred"]
n = len(y_true)

# Top-100 by RF ranking
rf_ranked = np.argsort(-rf_pred)
top100_rf = set(rf_ranked[:100])
top100_dl = set(np.argsort(-dl_pred)[:100])

missed_by_dl = top100_rf - top100_dl
missed_by_rf = top100_dl - top100_rf
both = top100_rf & top100_dl

log.info(f"  Top-100 overlap: {len(both)} compounds")
log.info(f"  Found by RF but missed by DL: {len(missed_by_dl)}")
log.info(f"  Found by DL but missed by RF: {len(missed_by_rf)}")

# Among those missed by DL, how many are real actives?
for threshold in CONFIG["activity_thresholds"]:
    missed_active = sum(1 for idx in missed_by_dl if y_true[idx] >= threshold)
    log.info(f"  Of {len(missed_by_dl)} missed by DL: {missed_active} are real actives (pKd ≥ {threshold})")

# Save missed actives
missed_rows = []
for idx in missed_by_dl:
    missed_rows.append({
        "compound_index": int(idx),
        "true_pkd": float(y_true[idx]),
        "rf_rank": int(np.where(rf_ranked == idx)[0][0]) + 1,
        "dl_rank": int(np.where(np.argsort(-dl_pred) == idx)[0][0]) + 1,
        "rf_pred": float(rf_pred[idx]),
        "dl_pred": float(dl_pred[idx]),
        "is_active_7": bool(y_true[idx] >= 7.0),
        "is_active_6": bool(y_true[idx] >= 6.0),
    })

missed_df = pd.DataFrame(missed_rows).sort_values("true_pkd", ascending=False)
missed_df.to_csv(EXP_DIR / "missed_actives.csv", index=False)

# ── Also: full ranking comparison ──────────────────────────────────────────────
ranking_df = pd.DataFrame({
    "compound_index": np.arange(n),
    "true_pkd": y_true,
    "rf_rank": np.argsort(np.argsort(-rf_pred)) + 1,  # 1-indexed
    "rf_pred": rf_pred,
    "dl_rank": np.argsort(np.argsort(-dl_pred)) + 1,
    "dl_pred": dl_pred,
    "delta_rank": np.argsort(np.argsort(-rf_pred)) - np.argsort(np.argsort(-dl_pred)),
})
ranking_df.to_csv(EXP_DIR / "ranking_comparison.csv", index=False)
log.info(f"\n  Full ranking comparison saved ({len(ranking_df):,} compounds)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Enrichment plot
# ═══════════════════════════════════════════════════════════════════════════════
log.info(f"\n{'='*50}")
log.info("STEP 5: Enrichment plot")
log.info(f"{'='*50}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax_idx, threshold in enumerate(CONFIG["activity_thresholds"]):
    ax = axes[ax_idx]
    subset = enrichment_df[enrichment_df["activity_threshold"] == threshold]
    
    # Only plot key models
    plot_models = ["rf_ecfp4_aac", "xgb_ecfp4_aac", "mlp_chemberta_esm2_8M", "bica_chemberta_esm2_8M"]
    colors = {"rf_ecfp4_aac": "#1b7837", "xgb_ecfp4_aac": "#5aae61", 
              "mlp_chemberta_esm2_8M": "#762a83", "bica_chemberta_esm2_8M": "#af8dc3"}
    markers = {"rf_ecfp4_aac": "o-", "xgb_ecfp4_aac": "s--",
               "mlp_chemberta_esm2_8M": "^-.", "bica_chemberta_esm2_8M": "d:"}
    
    for mid in plot_models:
        model_data = subset[subset["model_id"] == mid]
        if len(model_data) == 0:
            continue
        label = predictions[mid]["label"]
        ax.plot(model_data["top_percentile"], model_data["enrichment_factor"],
                markers.get(mid, "o-"), color=colors.get(mid, "gray"),
                label=label, linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Top N%", fontsize=12)
    ax.set_ylabel("Enrichment Factor", fontsize=12)
    ax.set_title(f"Active: pKd ≥ {threshold}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    ax.set_xlim(0, 11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(EXP_DIR / "enrichment_plot.pdf", dpi=150, bbox_inches="tight")
fig.savefig(EXP_DIR / "enrichment_plot.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved enrichment_plot.pdf and enrichment_plot.png")


# ═══════════════════════════════════════════════════════════════════════════════
log.info(f"\n{'='*50}")
log.info("E206 COMPLETE")
log.info(f"{'='*50}")
