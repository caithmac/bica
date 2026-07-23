#!/usr/bin/env python3
"""Phase 3 all-models benchmark plot — publication style."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "figure.dpi": 300,
})

df = pd.read_csv("E:/Drug Discovery/projects/balm-revision/phase3_results.csv")
random = df[df['split_type'] == 'random']

ORDER = [
    'XGBoost_ECFP4+AAC', 'RF_ECFP4+ESM2-8M', 'RF_ECFP4+AAC', 'RF_ECFP4-only',
    'KNN_ECFP4+AAC', 'Ridge_ECFP4+AAC',
    'MLP_deep_ECFP4+AAC', 'MLP_deep_ECFP4-only',
    'MLP_shallow_ECFP4+AAC', 'MLP_shallow_ECFP4-only',
]
NAMES = {
    'XGBoost_ECFP4+AAC': 'XGBoost', 'RF_ECFP4+ESM2-8M': 'RF+ESM2',
    'RF_ECFP4+AAC': 'RF+AAC', 'RF_ECFP4-only': 'RF-only',
    'KNN_ECFP4+AAC': 'KNN', 'Ridge_ECFP4+AAC': 'Ridge',
    'MLP_deep_ECFP4+AAC': 'MLP-d+AAC', 'MLP_deep_ECFP4-only': 'MLP-d',
    'MLP_shallow_ECFP4+AAC': 'MLP-s+AAC', 'MLP_shallow_ECFP4-only': 'MLP-s',
}
COLORS = ['#C55A11','#5B9BD5','#2E75B6','#A5C8E1',
          '#BF8F00','#806000',
          '#70AD47','#A9D18E','#548235','#C5E0B4']

means_rmse, stds_rmse = {}, {}
means_r, stds_r = {}, {}
for m in ORDER:
    sub = random[random['model'] == m]
    means_rmse[m] = sub['test_rmse'].mean()
    stds_rmse[m] = sub['test_rmse'].std()
    means_r[m] = sub['test_pearson'].mean()
    stds_r[m] = sub['test_pearson'].std()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2), gridspec_kw={'width_ratios': [1.2, 1]})

# Panel A: RMSE
x = np.arange(len(ORDER))
ax1.bar(x, [means_rmse[m] for m in ORDER],
        yerr=[stds_rmse[m] for m in ORDER],
        color=COLORS, edgecolor='#333333', linewidth=0.6,
        capsize=3, error_kw={'linewidth': 0.8, 'capthick': 0.8}, width=0.7)

for i, m in enumerate(ORDER):
    val = means_rmse[m]
    ax1.text(i, val + stds_rmse[m] + 0.025, f'{val:.3f}',
             ha='center', va='bottom', fontsize=7, fontweight='bold' if i<4 else 'normal',
             color='#222', path_effects=[pe.withStroke(linewidth=2, foreground='white')])

# Separators
for sep_x in [0.5, 3.5, 5.5]:
    ax1.axvline(x=sep_x, color='#999', linewidth=0.4, linestyle='--', alpha=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels([NAMES[m] for m in ORDER], fontsize=7, rotation=25, ha='right')
ax1.set_ylabel('Test RMSE ↓', fontsize=9)
ax1.set_title('A  Test RMSE (random split, 3 seeds)', fontsize=10, fontweight='bold', loc='left', pad=8)
ax1.set_ylim(0.65, 1.65)
ax1.grid(axis='y', color='#e0e0e0', linewidth=0.4, alpha=0.6)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Family labels
for pos, lbl, clr in [(0.0,'XGB', '#C55A11'),(2.0,'RF','#2E75B6'),(4.5,'KNN/\nRidge','#806000'),(7.5,'MLP','#70AD47')]:
    ax1.text(pos, 1.60, lbl, ha='center', fontsize=7, fontstyle='italic', color=clr, fontweight='bold')

# Panel B: Pearson r
ax2.bar(x, [means_r[m] for m in ORDER],
        yerr=[stds_r[m] for m in ORDER],
        color=COLORS, edgecolor='#333333', linewidth=0.6,
        capsize=3, error_kw={'linewidth': 0.8, 'capthick': 0.8}, width=0.7)

for i, m in enumerate(ORDER):
    val = means_r[m]
    ax2.text(i, val + stds_r[m] + 0.015, f'{val:.3f}',
             ha='center', va='bottom', fontsize=7, color='#222',
             path_effects=[pe.withStroke(linewidth=2, foreground='white')])

for sep_x in [0.5, 3.5, 5.5]:
    ax2.axvline(x=sep_x, color='#999', linewidth=0.4, linestyle='--', alpha=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels([NAMES[m] for m in ORDER], fontsize=7, rotation=25, ha='right')
ax2.set_ylabel('Pearson r ↑', fontsize=9)
ax2.set_title('B  Pearson correlation', fontsize=10, fontweight='bold', loc='left', pad=8)
ax2.set_ylim(0, 0.95)
ax2.grid(axis='y', color='#e0e0e0', linewidth=0.4, alpha=0.6)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=1.5)
out = "E:/Drug Discovery/projects/balm-revision/results/phase3_random_benchmark.png"
plt.savefig(out, bbox_inches='tight', pad_inches=0.04)
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.04)
print(f"Saved: {out}")
