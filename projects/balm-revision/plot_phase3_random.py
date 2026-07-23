#!/usr/bin/env python3
"""Phase 3 random split benchmark figure — publication style."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

# --- Scientific style setup ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "figure.dpi": 300,
})

# --- Load data ---
df = pd.read_csv("E:/Drug Discovery/projects/balm-revision/phase3_results.csv")
random = df[df['split_type'] == 'random']

# --- Aggregate (mean ± std across seeds) ---
MODEL_ORDER = [
    'RF_ECFP4+ESM2-8M',
    'RF_ECFP4+AAC',
    'RF_ECFP4-only',
    'MLP_deep_ECFP4+AAC',
    'MLP_deep_ECFP4-only',
    'MLP_shallow_ECFP4+AAC',
    'MLP_shallow_ECFP4-only',
]

SHORT_NAMES = {
    'RF_ECFP4+ESM2-8M':    'RF+ESM2',
    'RF_ECFP4+AAC':         'RF+AAC',
    'RF_ECFP4-only':        'RF-only',
    'MLP_deep_ECFP4+AAC':   'MLP-d+AAC',
    'MLP_deep_ECFP4-only':  'MLP-d',
    'MLP_shallow_ECFP4+AAC':'MLP-s+AAC',
    'MLP_shallow_ECFP4-only':'MLP-s',
}

FAMILY = {
    'RF_ECFP4+ESM2-8M': 'tree',
    'RF_ECFP4+AAC': 'tree',
    'RF_ECFP4-only': 'tree',
    'MLP_deep_ECFP4+AAC': 'mlp',
    'MLP_deep_ECFP4-only': 'mlp',
    'MLP_shallow_ECFP4+AAC': 'mlp',
    'MLP_shallow_ECFP4-only': 'mlp',
}

# Colors: rf blue, rf+esm2 lighter blue, mlp green shades
COLORS = {
    'RF_ECFP4+ESM2-8M':     '#5B9BD5',
    'RF_ECFP4+AAC':          '#2E75B6',
    'RF_ECFP4-only':         '#A5C8E1',
    'MLP_deep_ECFP4+AAC':    '#70AD47',
    'MLP_deep_ECFP4-only':   '#A9D18E',
    'MLP_shallow_ECFP4+AAC': '#548235',
    'MLP_shallow_ECFP4-only':'#C5E0B4',
}

means_rmse = {}
stds_rmse = {}
means_r = {}
stds_r = {}

for m in MODEL_ORDER:
    sub = random[random['model'] == m]
    means_rmse[m] = sub['test_rmse'].mean()
    stds_rmse[m] = sub['test_rmse'].std()
    means_r[m] = sub['test_pearson'].mean()
    stds_r[m] = sub['test_pearson'].std()

# --- Create figure: two panels (RMSE bars + Pearson bars) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2), gridspec_kw={'width_ratios': [1.2, 1]})

# --- Panel A: Test RMSE ---
x = np.arange(len(MODEL_ORDER))
bars = ax1.bar(x, [means_rmse[m] for m in MODEL_ORDER],
               yerr=[stds_rmse[m] for m in MODEL_ORDER],
               color=[COLORS[m] for m in MODEL_ORDER],
               edgecolor='#333333', linewidth=0.6,
               capsize=3, error_kw={'linewidth': 0.8, 'capthick': 0.8},
               width=0.65)

# Value labels above bars
for i, m in enumerate(MODEL_ORDER):
    val = means_rmse[m]
    ypos = val + stds_rmse[m] + 0.025
    color = '#222222'
    weight = 'bold' if m in ['RF_ECFP4+AAC', 'RF_ECFP4+ESM2-8M'] else 'normal'
    ax1.text(i, ypos, f'{val:.3f}', ha='center', va='bottom',
             fontsize=7.5, fontweight=weight, color=color,
             path_effects=[pe.withStroke(linewidth=2, foreground='white')])

# Separator line between RF and MLP
ax1.axvline(x=2.5, color='#888888', linewidth=0.5, linestyle='--', alpha=0.6)

ax1.set_xticks(x)
ax1.set_xticklabels([SHORT_NAMES[m] for m in MODEL_ORDER], fontsize=7)
ax1.set_ylabel('Test RMSE ↓', fontsize=9)
ax1.set_title('A  Test RMSE (random split, 3 seeds)', fontsize=10, fontweight='bold', loc='left', pad=8)
ax1.set_ylim(0.7, 1.65)
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax1.grid(axis='y', color='#e0e0e0', linewidth=0.4, alpha=0.7)
ax1.tick_params(axis='x', pad=3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Family labels
ax1.text(1.0, 1.56, 'Random Forest', ha='center', fontsize=7.5, fontstyle='italic',
         color='#2E75B6', fontweight='bold')
ax1.text(5.0, 1.56, 'MLP', ha='center', fontsize=7.5, fontstyle='italic',
         color='#548235', fontweight='bold')

# --- Panel B: Pearson r ---
bars2 = ax2.bar(x, [means_r[m] for m in MODEL_ORDER],
                yerr=[stds_r[m] for m in MODEL_ORDER],
                color=[COLORS[m] for m in MODEL_ORDER],
                edgecolor='#333333', linewidth=0.6,
                capsize=3, error_kw={'linewidth': 0.8, 'capthick': 0.8},
                width=0.65)

for i, m in enumerate(MODEL_ORDER):
    val = means_r[m]
    ypos = val + stds_r[m] + 0.015
    ax2.text(i, ypos, f'{val:.3f}', ha='center', va='bottom',
             fontsize=7.5, color='#222222',
             path_effects=[pe.withStroke(linewidth=2, foreground='white')])

ax2.axvline(x=2.5, color='#888888', linewidth=0.5, linestyle='--', alpha=0.6)

ax2.set_xticks(x)
ax2.set_xticklabels([SHORT_NAMES[m] for m in MODEL_ORDER], fontsize=7)
ax2.set_ylabel('Pearson r ↑', fontsize=9)
ax2.set_title('B  Pearson correlation', fontsize=10, fontweight='bold', loc='left', pad=8)
ax2.set_ylim(0, 0.95)
ax2.grid(axis='y', color='#e0e0e0', linewidth=0.4, alpha=0.7)
ax2.tick_params(axis='x', pad=3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Finish ---
plt.tight_layout(pad=1.5)
out = "E:/Drug Discovery/projects/balm-revision/results/phase3_random_benchmark.pdf"
plt.savefig(out, bbox_inches='tight', pad_inches=0.04)
plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.04, dpi=300)
print(f"Saved: {out}")
