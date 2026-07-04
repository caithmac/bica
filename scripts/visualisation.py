# Cell 1: Reconstruct full results from all 96 checkpoints
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import re

checkpoint_dir = Path('checkpoints')
rows = []

for ckpt_path in sorted(checkpoint_dir.glob('best_*.pt')):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    metrics = ckpt.get('metrics', {})
    config = ckpt.get('model_config', {})
    
    # Parse filename: best_{model}_{split}_seed{seed}.pt
    name = ckpt_path.stem.replace('best_', '')
    # Extract seed
    seed_match = re.search(r'_seed(\d+)$', name)
    seed = int(seed_match.group(1)) if seed_match else None
    name_no_seed = name[:seed_match.start()] if seed_match else name
    
    # Extract split type (last segment before seed)
    for split in ['random', 'scaffold', 'cold_drug', 'cold_target']:
        if name_no_seed.endswith(f'_{split}'):
            split_type = split
            model_name = name_no_seed[:-(len(split) + 1)]
            break
    else:
        split_type = config.get('split_type', 'unknown')
        model_name = config.get('model_name', name_no_seed)
    
    rows.append({
        'model_name': model_name,
        'split_type': split_type,
        'seed': seed,
        'num_heads': config.get('num_heads', None),
        'best_epoch': metrics.get('epoch', ckpt.get('epoch', None)),
        'val_rmse': metrics.get('val_rmse', None),
        'val_r2': metrics.get('val_r2', None),
        'val_pearson': metrics.get('val_pearson', None),
        'test_rmse': metrics.get('test_rmse', None),
        'test_r2': metrics.get('test_r2', None),
        'test_pearson': metrics.get('test_pearson', None),
        'test_spearman': metrics.get('test_spearman', None),
    })

df = pd.DataFrame(rows)

# Fill num_heads from model name where missing
def extract_heads(name):
    m = re.search(r'_(\d+)h$', name)
    return int(m.group(1)) if m else 8

df['num_heads'] = df.apply(lambda r: r['num_heads'] if pd.notna(r['num_heads']) else extract_heads(r['model_name']), axis=1)
df['num_heads'] = df['num_heads'].astype(int)

# Add a clean model type column
def get_model_type(name):
    if 'Unidirectional_P2L' in name:
        return 'Unidirectional P→L'
    elif 'Unidirectional_L2P' in name:
        return 'Unidirectional L→P'
    elif 'NoResidual' in name:
        return 'BiCA (No Residual)'
    elif 'VariableHeads' in name:
        return 'BiCA (Full)'
    return name

df['model_type'] = df['model_name'].apply(get_model_type)

# Save complete results
df.to_csv('results/all_experiments_complete.csv', index=False)
print(f"Loaded {len(df)} experiments from checkpoints")
print(f"\nModels: {df['model_name'].unique()}")
print(f"Splits: {df['split_type'].unique()}")
print(f"Seeds: {df['seed'].unique()}")
df.head()
# Cell 2: Summary table (mean ± std across seeds)
import warnings
warnings.filterwarnings('ignore')

summary = df.groupby(['model_type', 'model_name', 'num_heads', 'split_type']).agg(
    test_rmse_mean=('test_rmse', 'mean'),
    test_rmse_std=('test_rmse', 'std'),
    test_r2_mean=('test_r2', 'mean'),
    test_r2_std=('test_r2', 'std'),
    test_pearson_mean=('test_pearson', 'mean'),
    test_pearson_std=('test_pearson', 'std'),
    test_spearman_mean=('test_spearman', 'mean'),
    test_spearman_std=('test_spearman', 'std'),
    count=('seed', 'count')
).reset_index()

# Display formatted
def fmt(mean, std):
    return f"{mean:.3f}±{std:.3f}"

display_rows = []
for _, r in summary.iterrows():
    display_rows.append({
        'Model': r['model_name'],
        'Heads': int(r['num_heads']),
        'Split': r['split_type'],
        'N': int(r['count']),
        'RMSE': fmt(r['test_rmse_mean'], r['test_rmse_std']),
        'R²': fmt(r['test_r2_mean'], r['test_r2_std']),
        'Pearson': fmt(r['test_pearson_mean'], r['test_pearson_std']),
        'Spearman': fmt(r['test_spearman_mean'], r['test_spearman_std']),
    })

summary_display = pd.DataFrame(display_rows)
print(f"Summary across {len(df)} experiments ({len(summary)} model×split combos)\n")
summary_display
# Cell 3: Figure 1 — Model comparison across all splits (grouped bar chart)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 120})

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics_to_plot = [
    ('test_rmse_mean', 'test_rmse_std', 'Test RMSE ↓', True),
    ('test_r2_mean', 'test_r2_std', 'Test R²  ↑', False),
    ('test_pearson_mean', 'test_pearson_std', 'Pearson r ↑', False),
    ('test_spearman_mean', 'test_spearman_std', 'Spearman ρ ↑', False),
]

splits = ['random', 'scaffold', 'cold_drug', 'cold_target']
split_labels = ['Random', 'Scaffold', 'Cold Drug', 'Cold Target']

# Aggregate across splits: mean over all splits for each model
model_order = [
    'BiCA_Unidirectional_L2P_8h',
    'BiCA_Unidirectional_P2L_8h',
    'BiCA_NoResidual_8h',
    'BiCA_VariableHeads_2h',
    'BiCA_VariableHeads_4h',
    'BiCA_VariableHeads_8h',
    'BiCA_VariableHeads_16h',
    'BiCA_VariableHeads_32h',
]
model_labels = [
    'Uni L→P (8h)',
    'Uni P→L (8h)',
    'No Residual (8h)',
    'Full (2h)',
    'Full (4h)',
    'Full (8h)',
    'Full (16h)',
    'Full (32h)',
]

colors = plt.cm.Set2(np.linspace(0, 1, len(splits)))

for ax, (metric, std_col, title, lower_better) in zip(axes.flat, metrics_to_plot):
    x = np.arange(len(model_order))
    width = 0.2
    
    for i, (split, split_label) in enumerate(zip(splits, split_labels)):
        vals = []
        errs = []
        for model in model_order:
            row = summary[(summary['model_name'] == model) & (summary['split_type'] == split)]
            if len(row) > 0:
                vals.append(row[metric].values[0])
                errs.append(row[std_col].values[0])
            else:
                vals.append(np.nan)
                errs.append(0)
        
        ax.bar(x + i * width, vals, width, yerr=errs, label=split_label, 
               color=colors[i], edgecolor='white', linewidth=0.5, capsize=2)
    
    ax.set_xlabel('Model')
    ax.set_ylabel(title)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_labels, rotation=35, ha='right', fontsize=9)
    ax.legend(fontsize=8, loc='best')
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('BiCA Ablation Study — All Models × All Splits (3 seeds)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/fig1_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
# Cell 4: Figure 2 — Number of Heads ablation (line plot with error bands)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

heads_df = summary[summary['model_type'] == 'BiCA (Full)'].copy()
heads_df = heads_df.sort_values('num_heads')

metrics_line = [
    ('test_rmse_mean', 'test_rmse_std', 'Test RMSE ↓'),
    ('test_r2_mean', 'test_r2_std', 'Test R² ↑'),
    ('test_pearson_mean', 'test_pearson_std', 'Pearson r ↑'),
    ('test_spearman_mean', 'test_spearman_std', 'Spearman ρ ↑'),
]

colors_split = {'random': '#e74c3c', 'scaffold': '#3498db', 'cold_drug': '#2ecc71', 'cold_target': '#9b59b6'}
markers = {'random': 'o', 'scaffold': 's', 'cold_drug': 'D', 'cold_target': '^'}

for ax, (metric, std_col, title) in zip(axes, metrics_line):
    for split in splits:
        split_data = heads_df[heads_df['split_type'] == split].sort_values('num_heads')
        if len(split_data) == 0:
            continue
        heads = split_data['num_heads'].values
        means = split_data[metric].values
        stds = split_data[std_col].values
        
        ax.plot(heads, means, marker=markers[split], color=colors_split[split], 
                label=split.replace('_', ' ').title(), linewidth=2, markersize=7)
        ax.fill_between(heads, means - stds, means + stds, alpha=0.15, color=colors_split[split])
    
    ax.set_xlabel('Number of Attention Heads')
    ax.set_ylabel(title)
    ax.set_title(title, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle('Effect of Number of Attention Heads (BiCA Full Model)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/fig2_heads_ablation.png', dpi=300, bbox_inches='tight')
plt.show()
# Cell 5: Figure 3 — Architecture ablation (Bidirectional vs Unidirectional vs No Residual)
# Compare only the 8-head variants for fair comparison
arch_models = [
    ('BiCA_Unidirectional_L2P_8h', 'Uni L→P'),
    ('BiCA_Unidirectional_P2L_8h', 'Uni P→L'),
    ('BiCA_NoResidual_8h', 'Bi (No Res.)'),
    ('BiCA_VariableHeads_8h', 'Bi (Full)'),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

arch_colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

for ax, (split, split_label) in zip(axes, zip(splits, split_labels)):
    model_names_plot = []
    means = []
    stds = []
    
    for model, label in arch_models:
        row = summary[(summary['model_name'] == model) & (summary['split_type'] == split)]
        if len(row) > 0:
            model_names_plot.append(label)
            means.append(row['test_pearson_mean'].values[0])
            stds.append(row['test_pearson_std'].values[0])
    
    bars = ax.bar(model_names_plot, means, yerr=stds, capsize=4, 
                  color=arch_colors[:len(model_names_plot)], edgecolor='white', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_title(f'{split_label} Split', fontweight='bold')
    ax.set_ylabel('Pearson r')
    ax.set_ylim(0.55, 0.95)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=20)

fig.suptitle('Architecture Ablation — Pearson r by Split Type (8 heads, 3 seeds)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/fig3_architecture_ablation.png', dpi=300, bbox_inches='tight')
plt.show()
# Cell 6: Figure 4 — Heatmap of Pearson r (models × splits)
import seaborn as sns

# Pivot: rows = models, columns = splits
pivot_data = summary.pivot_table(
    index='model_name', columns='split_type', values='test_pearson_mean'
)

# Reorder rows and columns
pivot_data = pivot_data.reindex(index=model_order, columns=splits)
pivot_data.index = model_labels

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.65, vmax=0.90,
            linewidths=0.5, linecolor='white', ax=ax,
            cbar_kws={'label': 'Pearson r', 'shrink': 0.8})
ax.set_title('Test Pearson r — All Models × All Splits\n(mean over 3 seeds)', 
             fontsize=13, fontweight='bold')
ax.set_xlabel('Split Type')
ax.set_ylabel('Model')
ax.set_xticklabels(['Random', 'Scaffold', 'Cold Drug', 'Cold Target'], rotation=0)

plt.tight_layout()
plt.savefig('results/fig4_heatmap_pearson.png', dpi=300, bbox_inches='tight')
plt.show()
# Cell 7: Figure 5 — Heatmap of Test R² (models × splits)
pivot_r2 = summary.pivot_table(
    index='model_name', columns='split_type', values='test_r2_mean'
)
pivot_r2 = pivot_r2.reindex(index=model_order, columns=splits)
pivot_r2.index = model_labels

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.40, vmax=0.75,
            linewidths=0.5, linecolor='white', ax=ax,
            cbar_kws={'label': 'R²', 'shrink': 0.8})
ax.set_title('Test R² — All Models × All Splits\n(mean over 3 seeds)', 
             fontsize=13, fontweight='bold')
ax.set_xlabel('Split Type')
ax.set_ylabel('Model')
ax.set_xticklabels(['Random', 'Scaffold', 'Cold Drug', 'Cold Target'], rotation=0)

plt.tight_layout()
plt.savefig('results/fig5_heatmap_r2.png', dpi=300, bbox_inches='tight')
plt.show()
# Cell 8: Figure 6 — Box plots showing variance across seeds per model
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

box_metrics = [
    ('test_rmse', 'Test RMSE ↓'),
    ('test_r2', 'Test R² ↑'),
    ('test_pearson', 'Pearson r ↑'),
    ('test_spearman', 'Spearman ρ ↑'),
]

# Add clean label for boxplots
df['model_label'] = df['model_name'].map(dict(zip(model_order, model_labels)))

for ax, (metric, title) in zip(axes.flat, box_metrics):
    plot_df = df[df['model_label'].notna()].copy()
    plot_df['model_label'] = pd.Categorical(plot_df['model_label'], categories=model_labels, ordered=True)
    
    sns.boxplot(data=plot_df, x='model_label', y=metric, hue='split_type',
                palette=colors_split, ax=ax, linewidth=0.8, fliersize=3)
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(title)
    ax.tick_params(axis='x', rotation=35)
    ax.legend(title='Split', fontsize=7, title_fontsize=8, loc='best')
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Distribution of Metrics Across Seeds (3 seeds per configuration)', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/fig6_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()