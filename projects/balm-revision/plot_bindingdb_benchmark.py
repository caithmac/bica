#!/usr/bin/env python3
"""Publication-style figure: BindingDB benchmark — all 9 models × 5 splits."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Data from verified CSV ────────────────────────────────────────────
models = [
    "XGBoost\n+AAC", "RF\n+AAC", "RF\n-only",
    "KNN\n+AAC", "Ridge\n+AAC",
    "MLP shallow\n+AAC", "MLP shallow\n-only",
    "MLP deep\n+AAC", "MLP deep\n-only",
]
splits = ["Random", "Scaffold", "Cold\nTarget", "Seq\n30%", "Seq\n40%"]

rmse = np.array([
    [0.858, 1.028, 1.369, 1.374, 1.374],
    [0.918, 1.089, 1.432, 1.430, 1.430],
    [1.117, 1.232, 1.367, 1.367, 1.367],
    [1.081, 1.381, 1.773, 1.773, 1.773],
    [1.240, 1.335, 1.519, 1.519, 1.519],
    [1.670, 1.680, 1.598, 1.614, 1.614],
    [1.670, 1.684, 1.599, 1.577, 1.577],
    [1.666, 1.692, 1.587, 1.584, 1.584],
    [1.666, 1.679, 1.600, 1.580, 1.580],
])

std = np.array([
    [0.012, 0.011, 0.024, 0.020, 0.020],
    [0.007, 0.008, 0.023, 0.020, 0.020],
    [0.008, 0.015, 0.013, 0.012, 0.012],
    [0.011, 0.021, 0.035, 0.035, 0.035],
    [0.007, 0.011, 0.029, 0.029, 0.029],
    [0.028, 0.029, 0.036, 0.037, 0.037],
    [0.025, 0.033, 0.034, 0.037, 0.037],
    [0.035, 0.030, 0.037, 0.038, 0.038],
    [0.028, 0.030, 0.039, 0.040, 0.040],
])

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 9,
    "legend.fontsize": 7.5, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

colors = [
    "#2E7D32", "#388E3C", "#66BB6A",
    "#1565C0", "#42A5F5",
    "#C62828", "#E53935", "#EF5350", "#EF9A9A",
]

fig, ax = plt.subplots(figsize=(9, 5.2))

x = np.arange(len(splits))
width = 0.085
n = len(models)

for i in range(n):
    offset = (i - (n - 1) / 2) * width
    ax.bar(x + offset, rmse[i], width,
           yerr=std[i], label=models[i],
           color=colors[i], edgecolor="white", linewidth=0.4,
           error_kw={"lw": 0.8, "capsize": 2, "capthick": 0.8})

ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.set_ylabel("Test RMSE (pKd)")
ax.set_ylim(0.7, 1.95)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.7, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
          frameon=True, fancybox=False, edgecolor="#cccccc",
          framealpha=0.95, borderpad=0.5)
ax.set_title("BindingDB pKd Benchmark (52,898 pairs, ECFP4 + AAC)",
             fontweight="bold", pad=12)
ax.text(0.02, 0.96, "3 seeds, mean ± SD", transform=ax.transAxes,
        fontsize=8, fontstyle="italic", color="#555555", va="top")

fig.tight_layout()
out = "E:/Drug Discovery/projects/balm-revision/results/phase3_bindingdb_benchmark"
fig.savefig(out + ".png", dpi=250, facecolor="white", edgecolor="none")
fig.savefig(out + ".pdf", facecolor="white", edgecolor="none")
print(f"Saved: {out}.png + .pdf")
