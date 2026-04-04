"""
Generate paper figures for the BiCA benchmark paper.

fig1_leaderboard.pdf  — Best-per-family bar chart (RMSE, colour-coded by model class)
fig2_architecture.pdf — BiCA v2 architecture schematic (text boxes + arrows, matplotlib)
fig3–fig7             — Copied / re-saved from diary/figures/per_compound/

Run from any directory; paths are relative to this file's location.
"""

import os
import sys
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

HERE   = Path(__file__).parent
SRC    = Path("e:/Drug Discovery/diary/figures/per_compound")
PAPER  = Path("e:/Drug Discovery/writing_outputs/20260404_bica_benchmark_paper")
FIGDIR = PAPER / "figures"

FIGDIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Fig 1 – Leaderboard bar chart
# ──────────────────────────────────────────────────────────────────────────────

families = [
    ("Linear baselines",   "Ridge (ChemBERTa+ESM-2)",    1.2544, "#b0b0b0"),
    ("Classical ML",       "RF (ECFP4+AAC)",              1.0065, "#2ca02c"),
    ("Classical ML",       "XGBoost (ChemBERTa+ESM-2)",  1.0434, "#98df8a"),
    ("Classical ML",       "LightGBM (ECFP4+AAC)",       1.0528, "#c5e8c5"),
    ("MLP",                "MLP (ChemBERTa+ESM-2)",       1.1007, "#ff7f0e"),
    ("CNN/LSTM/Transformer","CNN-1D (SMILES+ESM-2)",      1.3243, "#d62728"),
    ("CNN/LSTM/Transformer","DistMat CNN",                1.1092, "#e6a0a0"),
    ("CNN/LSTM/Transformer","LSTM (BPE)",                 1.1456, "#ff9896"),
    ("CNN/LSTM/Transformer","Transformer-flat",           1.1194, "#f7b6b6"),
    ("GNN",                "GCN (MolGraph+ESM-2)",        1.2024, "#9467bd"),
    ("GNN",                "GAT (MolGraph+ESM-2)",        1.1939, "#c5b0d5"),
    ("BiCA",               "BiCA v2 (ChemBERTa+ESMC)",   1.1020, "#1f77b4"),
]

labels   = [f[1] for f in families]
rmse     = [f[2] for f in families]
colours  = [f[3] for f in families]
families_label = [f[0] for f in families]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.barh(range(len(labels)), rmse, color=colours, edgecolor="white",
               linewidth=0.5, height=0.7)

# value labels
for i, v in enumerate(rmse):
    ax.text(v + 0.003, i, f"{v:.4f}", va="center", ha="left", fontsize=8)

# RF reference line
rf_rmse = 1.0065
ax.axvline(rf_rmse, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7,
           label=f"RF baseline ({rf_rmse:.4f})")

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Test RMSE (pKd) ↓", fontsize=11)
ax.set_title("Best model per family — BindingDB_filtered scaffold split",
             fontsize=11, pad=10)
ax.set_xlim(0.9, 1.42)
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
fig.savefig(FIGDIR / "fig1_leaderboard.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig1_leaderboard.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig1_leaderboard.pdf  ✓")

# ──────────────────────────────────────────────────────────────────────────────
# Fig 2 – BiCA v2 architecture schematic
# ──────────────────────────────────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis("off")

def box(ax, x, y, w, h, label, sub="", fc="#ddeeff", ec="#336699", fs=9, sfs=7.5):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.08",
                                    facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x, y + (0.12 if sub else 0), label,
            ha="center", va="center", fontsize=fs, fontweight="bold", color="#1a1a3a")
    if sub:
        ax.text(x, y - 0.25, sub,
                ha="center", va="center", fontsize=sfs, color="#444")

def arrow(ax, x0, y0, x1, y1, label="", bidirectional=False):
    style = "<->" if bidirectional else "->"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color="#555", lw=1.2))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.05, my, label, fontsize=7, color="#555", ha="left", va="center")

# ─── Input boxes ───
box(ax2, 2.0, 5.0, 2.6, 0.7, "Protein sequence",   "ESM-2 (480-dim) → proj 256",
    fc="#e8f4e8", ec="#2ca02c")
box(ax2, 8.0, 5.0, 2.6, 0.7, "Ligand SMILES",      "ChemBERTa-77M (384-dim) → proj 256",
    fc="#ffe8cc", ec="#e07c00")

# ─── Cross-attention layers ───
box(ax2, 5.0, 3.8, 3.8, 0.65, "Cross-Attention Block × 2",
    "p2l: A_p2l (L_p×L_l)   l2p: A_l2p (L_l×L_p)",
    fc="#e0e8ff", ec="#336699")

arrow(ax2, 2.0, 4.65,  3.1, 4.13)   # prot → CA block
arrow(ax2, 8.0, 4.65,  6.9, 4.13)   # lig  → CA block

# ─── Value-weighting annotation ───
ax2.text(5.0, 3.35, r"Value-weighted: $\tilde{A}_{ij} \propto A_{ij}\cdot\|\mathbf{v}_j\|_2$",
         ha="center", fontsize=7.5, color="#883300",
         bbox=dict(boxstyle="round,pad=0.2", fc="#fff3e0", ec="#e07c00", alpha=0.8))

# ─── Attention Pool ───
box(ax2, 2.5, 2.5, 2.4, 0.65, "AttentionPool (protein)",
    "S3: scalar α_i per residue", fc="#f0fff0", ec="#2ca02c")
box(ax2, 7.5, 2.5, 2.4, 0.65, "AttentionPool (ligand)",
    "S4: scalar β_j per token",  fc="#fff8e8", ec="#e07c00")

arrow(ax2, 3.1, 3.47, 2.5, 2.83)
arrow(ax2, 6.9, 3.47, 7.5, 2.83)

# ─── Pooled vectors ───
box(ax2, 2.5, 1.6, 2.2, 0.55, "Protein vec (256)", fc="#e8f8e8", ec="#2ca02c")
box(ax2, 7.5, 1.6, 2.2, 0.55, "Ligand vec (256)",  fc="#fff4e0", ec="#e07c00")

arrow(ax2, 2.5, 2.17, 2.5, 1.87)
arrow(ax2, 7.5, 2.17, 7.5, 1.87)

# ─── Concatenation & MLP ───
box(ax2, 5.0, 1.0, 3.0, 0.65, "Concat → Predictor MLP",
    "512 → 256 → 1  (pKd)", fc="#f0e8ff", ec="#6644aa")
arrow(ax2, 3.6, 1.6, 4.35, 1.0+0.2)
arrow(ax2, 6.4, 1.6, 5.65, 1.0+0.2)

# ─── Output ───
ax2.annotate("", xy=(5.0, 0.55), xytext=(5.0, 0.67),
             arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))
ax2.text(5.0, 0.38, "pKd̂", ha="center", fontsize=12, fontweight="bold", color="#1a1a3a")

ax2.set_title("BiCA v2 — Bidirectional Cross-Attention Architecture", fontsize=11, pad=6)
plt.tight_layout()
fig2.savefig(FIGDIR / "fig2_architecture.pdf", bbox_inches="tight")
fig2.savefig(FIGDIR / "fig2_architecture.png", dpi=200, bbox_inches="tight")
plt.close(fig2)
print("fig2_architecture.pdf ✓")

# ──────────────────────────────────────────────────────────────────────────────
# Fig 3–7 — copy from diary/figures/per_compound/
# ──────────────────────────────────────────────────────────────────────────────

mapping = {
    "fig3_protein_consensus.png":     SRC / "protein_consensus.png",
    "fig4_top_compounds_heatmap.png": SRC / "top_compounds_heatmap.png",
    "fig5_value_weighted.png":        SRC / "value_weighted_comparison.png",
    "fig6_token_importance.png":      SRC / "token_importance.png",
    "fig7_consensus_heatmap.png":     SRC / "consensus_heatmap.png",
}

for dest_name, src_path in mapping.items():
    dest = FIGDIR / dest_name
    shutil.copy2(src_path, dest)
    print(f"{dest_name:45s}  ✓  (copied from {src_path.name})")

print("\nAll 7 figures written to:", FIGDIR)
