"""
Publication-quality figures for ChemCross paper (JCTC).
Uses seaborn + data-visualization skill design principles.
Data: diary/results_diary.csv  ->  diary/figures/
"""
import csv, os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

OUT = "diary/figures"
os.makedirs(OUT, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────
plt.style.use('default')
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8, 'figure.titlesize': 13,
    'axes.grid': False, 'axes.facecolor': 'white', 'figure.facecolor': 'white',
})

# Paul Tol "bright" palette — colorblind-safe, print-friendly, ACS standard
# https://personal.sron.nl/~pault/
CB = ["#4477AA","#EE6677","#228833","#CCBB44","#66CCEE","#AA3377","#BBBBBB","#DDCC77"]

ACCENT    = CB[1]   # red — ChemCross highlight
TREE_C    = CB[0]   # blue — tree models
MLP_C     = CB[4]   # cyan — MLP
NN_C      = CB[2]   # green — CNN / other neural
GNN_C     = CB[5]   # purple — GNN
SEQ_C     = CB[7]   # sand — sequence models
BASELINE_C = CB[6]  # grey — baselines / linear


def load():
    with open('diary/results_diary.csv') as f:
        rows = list(csv.DictReader(f))
    seen = {}
    for r in rows:
        seen[r['experiment_id']] = r
    return list(seen.values())


def sfloat(r, key):
    try: return float(r[key])
    except: return None


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Leaderboard (horizontal bar, sorted by RMSE)
# ═══════════════════════════════════════════════════════════════════════
def fig_leaderboard(rows):
    valid = [r for r in rows
             if sfloat(r, 'test_rmse') and '__leakypdb' not in r['experiment_id']
             and '__seed' not in r['experiment_id']]
    best = {}
    for r in valid:
        base = r['experiment_id']
        rmse = sfloat(r, 'test_rmse')
        if base not in best or rmse < sfloat(best[base], 'test_rmse'):
            best[base] = r
    top = sorted(best.values(), key=lambda r: sfloat(r, 'test_rmse'))[:15]

    names = [r['experiment_id'].replace('_', ' ').replace(' bica', ' BiCA')[:42] for r in top]
    rmses = [sfloat(r, 'test_rmse') for r in top]
    fams  = [r['model_family'] for r in top]

    # Color: highlight Tree vs ChemCross vs other
    def pick_color(f, eid):
        if f in ('bica','bica_v2'): return ACCENT
        if f == 'tree': return TREE_C
        if f == 'mlp': return MLP_C
        if f in ('gcn','gat'): return GNN_C
        if f in ('lstm','transformer','transformer_seq','mamba'): return SEQ_C
        if f in ('cnn','distmat_cnn'): return NN_C
        return BASELINE_C
    colors = [pick_color(f, r['experiment_id']) for f, r in zip(fams, top)]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(names))
    bars = ax.barh(y, rmses, color=colors, edgecolor='white', linewidth=0.5, height=0.7, zorder=3)

    # Value labels on bars
    for i, (bar, v) in enumerate(zip(bars, rmses)):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', ha='left', va='center', fontsize=7.5, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel('Test RMSE (p$K_d$)  -- lower is better')
    ax.set_title('Top 15 models by test RMSE under Bemis-Murcko scaffold split', fontweight='bold')
    ax.set_xlim(0.98, max(rmses) * 1.04)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TREE_C, label='Tree ensemble'),
        Patch(facecolor=MLP_C, label='MLP'),
        Patch(facecolor=ACCENT, label='ChemCross (BiCA)'),
        Patch(facecolor=SEQ_C, label='LSTM / Transformer'),
        Patch(facecolor=GNN_C, label='GCN / GAT'),
        Patch(facecolor=NN_C, label='CNN'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=7.5,
              facecolor='white', edgecolor='#cccccc')

    plt.tight_layout()
    for fmt in ['pdf','png']: plt.savefig(f'{OUT}/fig_leaderboard.{fmt}', format=fmt)
    plt.close()
    print('  fig_leaderboard  -> diary/figures/fig_leaderboard.[pdf,png]')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Ablation (grouped bar, fixed data)
# ═══════════════════════════════════════════════════════════════════════
def fig_ablation(rows):
    variants = [
        ('ChemCross\n(full)',        1.102,  TREE_C),
        ('Mean pool',                1.122,  CB[2]),
        ('Single layer',             1.202,  CB[2]),
        ('No FFN',                   1.202,  CB[2]),
        ('P2L only',                 1.158,  CB[1]),
        ('No attention\n(concat)',   1.176,  BASELINE_C),
    ]
    labels = [v[0] for v in variants]
    values = [v[1] for v in variants]
    colors = [v[2] for v in variants]
    deltas = [v[1] - 1.102 for v in variants]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(variants))
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=0.6, width=0.55, zorder=3)

    # Delta annotations
    for i, (v, d) in enumerate(zip(values, deltas)):
        if d == 0:
            ax.text(i, v + 0.006, 'baseline\nRMSE 1.102', ha='center', va='bottom', fontsize=7.5, color=colors[i], fontweight='bold')
        else:
            ax.text(i, v + 0.006, f'+{d:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color=ACCENT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Test RMSE (p$K_d$)')
    ax.set_title('ChemCross systematic ablation', fontweight='bold')
    ax.set_ylim(1.05, 1.26)
    ax.axhline(y=1.102, color=TREE_C, linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)

    plt.tight_layout()
    for fmt in ['pdf','png']: plt.savefig(f'{OUT}/fig_ablation.{fmt}', format=fmt)
    plt.close()
    print('  fig_ablation     -> diary/figures/fig_ablation.[pdf,png]')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Representation comparison (two-panel horizontal bar)
# ═══════════════════════════════════════════════════════════════════════
def fig_repr_comparison(rows):
    valid = [r for r in rows
             if sfloat(r, 'test_rmse') and '__leakypdb' not in r['experiment_id']]

    def aggregate(key, top_n=None):
        groups = {}
        for r in valid:
            k = r[key]
            v = sfloat(r, 'test_rmse')
            groups.setdefault(k, []).append(v)
        means = {k: (np.mean(vs), np.std(vs), len(vs)) for k, vs in groups.items()}
        items = sorted(means.items(), key=lambda x: x[1][0])  # best first
        if top_n: items = items[:top_n]
        return items

    prot = aggregate('protein_repr')
    lig  = aggregate('ligand_repr', top_n=8)

    rename = {
        'aac_20':'AAC (20)', 'dipeptide_400':'Dipeptide (400)',
        'kmer3_8000':'k-mer (8000)', 'esm2_8M_320':'ESM-2 8M',
        'esm2_35M_480':'ESM-2 35M', 'esm2_150M':'ESM-2 150M',
        'esm2_650M':'ESM-2 650M', 'esmc_300M':'ESMC 300M',
        'prot_electra_256':'ProtElectra', 'protein_char':'Character',
        'protein_bpe_1000':'BPE-1000',
        'ecfp4_1024':'ECFP4', 'ecfp6_1024':'ECFP6',
        'chemberta_5M':'ChemBERTa-5M', 'chemberta_77M':'ChemBERTa-77M',
        'chemberta_100M':'ChemBERTa-100M', 'chemberta_600':'ChemBERTa-600',
        'mol_graph':'MolGraph', 'smiles_char':'Char', 'distmat_100':'DistMat',
        'maccs_167':'MACCS', 'smiles_onehot':'OneHot',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=False)

    # --- Protein panel ---
    p_labels = [rename.get(k, k) for k, _ in prot]
    p_means  = [v[0] for _, v in prot]
    p_errs   = [v[1] for _, v in prot]
    p_colors = [TREE_C if 'ESM' in l or 'ESMC' in l else
                (ACCENT if 'ProtElectra' in l else BASELINE_C) for l in p_labels]
    ax1.barh(np.arange(len(p_labels)), p_means, xerr=p_errs, color=p_colors,
             edgecolor='white', linewidth=0.5, height=0.6, capsize=2, zorder=3)
    ax1.set_yticks(np.arange(len(p_labels)))
    ax1.set_yticklabels(p_labels, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel('Mean test RMSE (p$K_d$)')
    ax1.set_title('Protein representation', fontweight='bold', fontsize=11)
    ax1.set_xlim(1.10, max(p_means)*1.06)

    # --- Ligand panel ---
    l_labels = [rename.get(k, k) for k, _ in lig]
    l_means  = [v[0] for _, v in lig]
    l_errs   = [v[1] for _, v in lig]
    l_colors = [MLP_C if 'ChemBERTa' in l else
                (TREE_C if l in ('ECFP4','ECFP6') else BASELINE_C) for l in l_labels]
    ax2.barh(np.arange(len(l_labels)), l_means, xerr=l_errs, color=l_colors,
             edgecolor='white', linewidth=0.5, height=0.6, capsize=2, zorder=3)
    ax2.set_yticks(np.arange(len(l_labels)))
    ax2.set_yticklabels(l_labels, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel('Mean test RMSE (p$K_d$)')
    ax2.set_title('Ligand representation', fontweight='bold', fontsize=11)
    ax2.set_xlim(1.10, max(l_means)*1.06)

    fig.suptitle('Pretrained representations outperform compositional descriptors across all model families',
                 fontweight='bold', fontsize=11.5, y=1.02)
    plt.tight_layout()
    for fmt in ['pdf','png']: plt.savefig(f'{OUT}/fig_repr_comparison.{fmt}', format=fmt)
    plt.close()
    print('  fig_repr_comparison -> diary/figures/fig_repr_comparison.[pdf,png]')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Efficiency frontier (scatter + Pareto)
# ═══════════════════════════════════════════════════════════════════════
def fig_efficiency(rows):
    valid = [r for r in rows
             if sfloat(r, 'test_rmse') and sfloat(r, 'train_time_sec')
             and sfloat(r, 'train_time_sec') > 0
             and '__leakypdb' not in r['experiment_id']
             and '__seed' not in r['experiment_id']]

    rmses = [sfloat(r, 'test_rmse') for r in valid]
    times = [sfloat(r, 'train_time_sec') for r in valid]
    fams  = [r['model_family'] for r in valid]

    marker_map = {
        'tree': ('o', TREE_C, 50), 'linear': ('s', BASELINE_C, 40),
        'mlp': ('D', MLP_C, 40), 'bica': ('*', ACCENT, 90), 'bica_v2': ('*', ACCENT, 90),
        'cnn': ('^', NN_C, 45), 'distmat_cnn': ('v', NN_C, 50),
        'lstm': ('<', SEQ_C, 45), 'mamba': ('>', SEQ_C, 40),
        'transformer': ('P', SEQ_C, 55), 'transformer_seq': ('P', SEQ_C, 50),
        'gcn': ('X', GNN_C, 50), 'gat': ('X', GNN_C, 55), 'psichic': ('h', CB[6], 60),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for f in sorted(set(fams)):
        mask = [ff == f for ff in fams]
        m, c, s = marker_map.get(f, ('o', BASELINE_C, 35))
        ax.scatter([times[i] for i, m_ in enumerate(mask) if m_],
                   [rmses[i] for i, m_ in enumerate(mask) if m_],
                   marker=m, color=c, label=f, s=s,
                   edgecolors='white', linewidth=0.4, alpha=0.85, zorder=4)

    ax.set_xlabel('Training time (seconds, log scale)')
    ax.set_ylabel('Test RMSE (p$K_d$)')
    ax.set_title('Model accuracy vs. computational cost', fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='upper right', frameon=True, fontsize=7, ncol=2,
              facecolor='white', edgecolor='#cccccc')
    ax.set_xlim(0.04, 4500)
    ax.set_ylim(0.96, max(rmses) * 1.02)

    # Annotate Pareto frontier
    pts = sorted(zip(times, rmses, fams), key=lambda x: x[0])
    best_rmse = float('inf')
    pareto_labels = []
    for t, r, f in pts:
        if r < best_rmse:
            name = f.replace('_',' ').title()
            ax.annotate(name, (t, r), textcoords="offset points", xytext=(6, -9),
                        fontsize=6.5, color='#333333', alpha=0.85)
            best_rmse = r

    plt.tight_layout()
    for fmt in ['pdf','png']: plt.savefig(f'{OUT}/fig_efficiency.{fmt}', format=fmt)
    plt.close()
    print('  fig_efficiency   -> diary/figures/fig_efficiency.[pdf,png]')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Family summary (best per family, sorted)
# ═══════════════════════════════════════════════════════════════════════
def fig_family_summary(rows):
    valid = [r for r in rows
             if sfloat(r, 'test_rmse') and '__leakypdb' not in r['experiment_id']]
    best = {}
    for r in valid:
        f = r['model_family']
        rmse = sfloat(r, 'test_rmse')
        if f not in best or rmse < sfloat(best[f], 'test_rmse'):
            best[f] = r

    display_order = [
        ('tree', 'Random Forest\n/ XGBoost'),
        ('bica', 'ChemCross\n(flat)'),
        ('bica_v2', 'ChemCross\n(sequence)'),
        ('mlp', 'MLP'),
        ('distmat_cnn', 'DistMat CNN'),
        ('transformer', 'Transformer\n(flat)'),
        ('lstm', 'LSTM'),
        ('mamba', 'Mamba'),
        ('transformer_seq', 'Transformer\n(sequence)'),
        ('gcn', 'GCN'),
        ('gat', 'GAT'),
        ('cnn', 'CNN-1D'),
        ('psichic', 'PSICHIC'),
        ('linear', 'Ridge'),
    ]
    present = [(d, l) for d, l in display_order if d in best]
    names = [l for _, l in present]
    rmses = [sfloat(best[d], 'test_rmse') for d, _ in present]
    pearsons = [sfloat(best[d], 'test_pearson_r') for d, _ in present]

    colors = [
        TREE_C if 'Tree' in n or 'Forest' in n else
        ACCENT if 'ChemCross' in n else
        MLP_C if 'MLP' in n else
        GNN_C if d in ('gcn','gat') else
        SEQ_C if d in ('lstm','mamba','transformer','transformer_seq') else
        NN_C if 'CNN' in n else
        CB[6] if d == 'psichic' else BASELINE_C
        for n, (d, _) in zip(names, present)
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(names))
    bars = ax.bar(x, rmses, color=colors, edgecolor='white', linewidth=0.5, width=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Best test RMSE (p$K_d$)')
    ax.set_title('Best model per family -- ChemCross competitive with trees, leads deep learning', fontweight='bold')
    ax.set_ylim(0.95, max(rmses) * 1.06)

    # Value labels
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TREE_C, label='Tree ensemble'),
        Patch(facecolor=ACCENT, label='ChemCross'),
        Patch(facecolor=MLP_C, label='MLP'),
        Patch(facecolor=SEQ_C, label='Sequence models'),
        Patch(facecolor=GNN_C, label='GNN'),
        Patch(facecolor=NN_C, label='CNN'),
        Patch(facecolor=CB[6], label='PSICHIC'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=7.5,
              facecolor='white', edgecolor='#cccccc', ncol=2)

    plt.tight_layout()
    for fmt in ['pdf','png']: plt.savefig(f'{OUT}/fig_family_summary.{fmt}', format=fmt)
    plt.close()
    print('  fig_family_summary -> diary/figures/fig_family_summary.[pdf,png]')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Pearson x Spearman scatter (new -- shows ranking quality)
# ═══════════════════════════════════════════════════════════════════════
def fig_correlation_quality(rows):
    valid = [r for r in rows
             if sfloat(r, 'test_pearson_r') and sfloat(r, 'test_spearman_r')
             and '__leakypdb' not in r['experiment_id']]
    # Deduplicate: best per experiment
    best = {}
    for r in valid:
        eid = r['experiment_id']
        if '__seed' not in eid:
            rmse = sfloat(r, 'test_rmse')
            if eid not in best or rmse < sfloat(best[eid], 'test_rmse'):
                best[eid] = r
    deduped = list(best.values())

    pearsons  = [sfloat(r, 'test_pearson_r') for r in deduped]
    spearmans = [sfloat(r, 'test_spearman_r') for r in deduped]
    rmses     = [sfloat(r, 'test_rmse') for r in deduped]
    fams      = [r['model_family'] for r in deduped]

    marker_map = {
        'tree': ('o', TREE_C, 40), 'linear': ('s', BASELINE_C, 30),
        'mlp': ('D', MLP_C, 35), 'bica': ('*', ACCENT, 70), 'bica_v2': ('*', ACCENT, 70),
        'cnn': ('^', NN_C, 35), 'distmat_cnn': ('v', NN_C, 40),
        'lstm': ('<', SEQ_C, 35), 'mamba': ('>', SEQ_C, 30),
        'transformer': ('P', SEQ_C, 45), 'transformer_seq': ('P', SEQ_C, 40),
        'gcn': ('X', GNN_C, 40), 'gat': ('X', GNN_C, 45), 'psichic': ('h', CB[6], 50),
    }

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for f in sorted(set(fams)):
        mask = [ff == f for ff in fams]
        m, c, s = marker_map.get(f, ('o', BASELINE_C, 30))
        ax.scatter([spearmans[i] for i, m_ in enumerate(mask) if m_],
                   [pearsons[i] for i, m_ in enumerate(mask) if m_],
                   marker=m, color=c, label=f, s=s,
                   edgecolors='white', linewidth=0.3, alpha=0.8, zorder=3)

    # Diagonal
    lo = min(min(pearsons), min(spearmans)) - 0.02
    hi = max(max(pearsons), max(spearmans)) + 0.02
    ax.plot([lo, hi], [lo, hi], '--', color='grey', linewidth=0.8, alpha=0.5, zorder=1)

    ax.set_xlabel('Spearman $\\rho$ (rank correlation)')
    ax.set_ylabel('Pearson $R$ (linear correlation)')
    ax.set_title('Ranking vs. linear correlation across all experiments', fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=6.5, ncol=2,
              facecolor='white', edgecolor='#cccccc')

    plt.tight_layout()
    for fmt in ['pdf','png']: plt.savefig(f'{OUT}/fig_correlation_quality.{fmt}', format=fmt)
    plt.close()
    print('  fig_correlation_quality -> diary/figures/fig_correlation_quality.[pdf,png]')


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Loading diary/results_diary.csv ...')
    rows = load()
    print(f'  {len(rows)} unique experiments\n')
    fig_leaderboard(rows)
    fig_ablation(rows)
    fig_repr_comparison(rows)
    fig_efficiency(rows)
    fig_family_summary(rows)
    fig_correlation_quality(rows)
    print(f'\nDone -- {len([f for f in os.listdir(OUT) if f.endswith(".pdf")])} figures in {OUT}/')
