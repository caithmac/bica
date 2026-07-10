"""E302 — Calibration Analysis. RMSE good but are predictions trustworthy?"""
import sys, os, logging
from pathlib import Path
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = Path("E:/Drug Discovery/cache/predictions")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(EXP_DIR / "run.log", mode="w"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("E302")

# Top models to evaluate
MODELS = ["rf_ecfp4_aac", "xgb_ecfp4_aac", "lgbm_ecfp4_aac",
          "mlp_chemberta_esm2_8M", "xgb_chemberta_5M_esm2_650M"]

results = []
for mid in MODELS:
    pf = PRED_DIR / f"{mid}.npz"
    if not pf.exists():
        continue
    data = np.load(pf)
    y_true, y_pred = data["y_true"], data["y_pred"]
    errors = y_pred - y_true
    
    # Calibration metrics
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)  # systematic over/under prediction
    
    # Binned calibration: group true values into 10 bins, compute mean error per bin
    bins = np.percentile(y_true, np.linspace(0, 100, 11))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_errors, bin_stds = [], []
    for i in range(10):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if mask.sum() > 0:
            bin_errors.append(np.mean(errors[mask]))
            bin_stds.append(np.std(errors[mask]))
        else:
            bin_errors.append(np.nan)
            bin_stds.append(np.nan)
    
    # Std of residuals
    residual_std = np.std(errors)
    
    # Coverage: fraction within ±1 RMSE
    coverage_1rmse = np.mean(np.abs(errors) < rmse)
    
    results.append({
        "model": mid, "rmse": rmse, "mae": mae, "bias": bias,
        "residual_std": residual_std, "coverage_1rmse": coverage_1rmse,
    })
    
    log.info(f"{mid:40s} RMSE={rmse:.4f} MAE={mae:.4f} bias={bias:+.4f} "
             f"σ_resid={residual_std:.4f} cov_1σ={coverage_1rmse:.2%}")

df = pd.DataFrame(results)
df.to_csv(EXP_DIR / "calibration.csv", index=False)

# Key finding: check if bias is systematic
log.info(f"\nKey: {'RF shows negative bias (overpredicts pKd)' if df.iloc[0]['bias'] < -0.05 else 'No systematic bias'}")

# Quick calibration plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 6))
for i, mid in enumerate(MODELS):
    pf = PRED_DIR / f"{mid}.npz"
    if not pf.exists(): continue
    data = np.load(pf)
    yt, yp = data["y_true"], data["y_pred"]
    ax.scatter(yt[::10], yp[::10], alpha=0.3, s=3, label=mid)
ax.plot([2, 10], [2, 10], "k--", alpha=0.5)
ax.set_xlabel("True pKd"); ax.set_ylabel("Predicted pKd")
ax.legend(fontsize=7); ax.set_title("E302: Predicted vs True")
fig.savefig(EXP_DIR / "calibration_plot.pdf", dpi=150, bbox_inches="tight")
fig.savefig(EXP_DIR / "calibration_plot.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved calibration_plot.pdf")
