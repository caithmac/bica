# Drug Discovery Benchmark — Complete Runbook

**Project goal**: Nature-level binding affinity prediction paper.
**Dataset**: BindingDB (scaffold split) + LeakyPDB (new_split column).
**Conda env**: activate it before running anything.

```bash
conda activate drug_discovery   # or whatever your env is named
cd "e:/Drug Discovery"
```

---

## Project Structure

```
e:/Drug Discovery/
├── run_experiment.py        # single experiment runner
├── run_all.py               # master runner (all phases, resumable)
├── analyze_results.py       # leaderboard + FINDINGS.md
├── bootstrap_ci.py          # 95% CIs on saved predictions
├── visualize_results.py     # plots
├── generate_targeted_experiments.py  # identifies missing experiments
│
├── harness/
│   ├── config.py            # hyperparams, paths
│   ├── data.py              # BindingDB + LeakyPDB loading
│   ├── featurizers.py       # ECFP, ChemBERTa, ESM-2, kmer, etc.
│   ├── tokenizers.py        # char/BPE/WordPiece tokenizers
│   ├── trainer.py           # train_torch (MLP/CNN/transformer)
│   ├── gnn_trainer.py       # train_gnn_model (GCN/GAT/GLI/Graphormer)
│   ├── seq_trainer.py       # train_seq_model (LSTM/Mamba/TransformerSeq)
│   ├── losses.py            # PairwiseRankingLoss
│   ├── metrics.py           # RMSE, Pearson, Spearman, CI
│   └── diary.py             # log_result → results_diary.csv
│
├── models/
│   ├── sklearn_models.py    # Ridge, RF, XGB, LGBM
│   ├── mlp.py               # MLP (shallow/medium/deep/wide) + encode/head split
│   ├── cnn.py               # 1D CNN on SMILES one-hot
│   ├── distmat_cnn.py       # 2D CNN on topological distance matrix
│   ├── transformer.py       # flat self-attn + cross-attn fusion
│   ├── sequence_models.py   # LSTMBindingModel, TransformerSeqModel, MambaBindingModel
│   ├── bica.py              # BiCA bidirectional cross-attention
│   ├── gnn.py               # GCNBindingModel, GATBindingModel + NodeDecoder
│   ├── graphormer.py        # Graph Transformer (Ying et al., NeurIPS 2021)
│   ├── gli.py               # GLI gated global-local interaction
│   └── dsm.py               # DualBind DSM auxiliary loss head
│
├── interpret/
│   ├── shap_analysis.py     # SHAP for XGB/RF/MLP
│   ├── attention_maps.py    # cross-attention heatmaps (BiCA, GLI)
│   ├── gnn_explain.py       # Grad-AAM + GNNExplainer
│   ├── captum_ig.py         # Integrated Gradients (LSTM/Mamba)
│   └── fidelity_eval.py     # AUC-Deletion/Insertion, Sufficiency, Comprehensiveness
│
├── diary/
│   ├── results_diary.csv    # ALL experiment results (append-only)
│   ├── bootstrap_ci.csv     # 95% confidence intervals
│   ├── FINDINGS.md          # auto-generated leaderboard + analysis
│   └── DIARY_SCHEMA.md      # Nature publication plan
│
└── cache/
    ├── features/            # cached .npy feature arrays
    └── predictions/         # cached .npz test predictions (y_true, y_pred)
```

---

## Phase Overview

| Phase | Command / Script | Experiments | Est. Time |
|-------|-----------------|-------------|-----------|
| 1 – Base | `run_all.py` | 44 original | hours |
| 1 – Phase1 | `run_all.py` | 5 (ranking + recon) | ~1h |
| 2 – Targeted | `run_all.py` | 65 repr combos | hours |
| 3 – Phase2 | `run_all.py` | 16 new architectures | hours |
| 4 – Multiseed | `run_all.py` | seeds 123+456 | hours |
| 5 – LeakyPDB | `run_all.py` | 15 cross-dataset | hours |
| 6 – Interpret | manual scripts | best models | hours |
| 7 – PSICHIC | TBD | external baseline | TBD |

---

## Step 1 — Run All Experiments

```bash
# Full run (resumable — already-done experiments are skipped)
python run_all.py

# Dry run first to see what will execute
python run_all.py --dry-run

# Run only one phase
python run_all.py --only bindingdb_base
python run_all.py --only phase1_new_objectives
python run_all.py --only targeted_repr_combos
python run_all.py --only phase2_new_architectures
python run_all.py --only bindingdb_multiseed
python run_all.py --only leakypdb_base

# Skip slow experiments (GNN, seq, distmat) for a quick pass
python run_all.py --skip_slow

# Run a single experiment
python run_experiment.py --exp graphormer_mol_esm2_8M
python run_experiment.py --exp gli_mol_esm2_8M
python run_experiment.py --exp bica_chemberta_esm2_8M_dsm
python run_experiment.py --exp mamba_smiles_char_protein_char

# Run on LeakyPDB dataset
python run_experiment.py --exp gat_ecfp_esm2_8M --dataset leakypdb

# Run with different seed
python run_experiment.py --exp mlp_chemberta_esm2_8M --seed 123
```

**Resumability**: `run_all.py` checks `diary/results_diary.csv` AND
`cache/predictions/<exp_id>.npz`. If both exist, the experiment is skipped.
Safe to Ctrl+C and restart at any time.

---

## Step 2 — Check What Has Run

```bash
# Print full leaderboard sorted by test RMSE
python run_experiment.py --leaderboard

# List all registered experiments (200+)
python run_experiment.py --list

# See what targeted experiments are still missing
python generate_targeted_experiments.py

# Count how many are done
python -c "
import pandas as pd
df = pd.read_csv('diary/results_diary.csv')
print(f'Total logged: {len(df)}')
print(df.groupby('model_family')['test_rmse'].min().sort_values())
"
```

---

## Step 3 — Analysis After Experiments Complete

```bash
# Regenerate FINDINGS.md leaderboard
python analyze_results.py

# Compute 95% bootstrap CIs (uses cache/predictions/*.npz)
python bootstrap_ci.py --n_bootstrap 1000

# Generate plots
python visualize_results.py
```

**Output files:**
- `diary/FINDINGS.md` — leaderboard, best models per family, key findings
- `diary/bootstrap_ci.csv` — CI bounds for all experiments
- `diary/figures/` — PNG plots

---

## Step 4 — Interpretability (Phase 3)

These are **library files** — you need to call them from a script or notebook.
Create `interpret_best_models.py` like this:

```python
"""Run all interpretability methods on the top models."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Load results diary ────────────────────────────────────────────────────
df = pd.read_csv("diary/results_diary.csv")
best = df.nsmallest(5, "test_rmse")[["experiment_id","model_family","test_rmse"]]
print(best)

# ── SHAP for XGB ──────────────────────────────────────────────────────────
# (re-train or load model first, then:)
from interpret.shap_analysis import shap_for_tree, plot_shap_summary
# shap_vals = shap_for_tree(xgb_model, X_test)
# plot_shap_summary(shap_vals, X_test, feature_names, save_path="diary/figures/shap_xgb.png")

# ── Attention heatmap for BiCA ────────────────────────────────────────────
from interpret.attention_maps import extract_bica_attention, plot_attention_heatmap
# attn = extract_bica_attention(bica_model, prot_feat, lig_feat)
# plot_attention_heatmap(attn["protein_to_ligand"], save_path="diary/figures/attn_bica.png")

# ── Grad-AAM for GNN ─────────────────────────────────────────────────────
from interpret.gnn_explain import grad_aam, visualise_atom_importance
# scores = grad_aam(gat_model, data, prot_vec)
# visualise_atom_importance(smiles, scores, save_path="diary/figures/gradam_gat.svg")

# ── Integrated Gradients for Mamba/LSTM ──────────────────────────────────
from interpret.captum_ig import ig_for_seq_model, plot_ig_scores
# result = ig_for_seq_model(mamba_model, lig_ids, prot_ids)
# plot_ig_scores(result["lig_ig"], token_labels=tokens, save_path="diary/figures/ig_mamba.png")

# ── Fidelity evaluation ───────────────────────────────────────────────────
from interpret.fidelity_eval import fidelity_report
# report = fidelity_report(xgb_model, X_test, shap_vals.mean(0),
#                          experiment_id="xgb_chemberta_esm2_8M",
#                          save_path="diary/figures/fidelity_xgb.json")
```

---

## Step 5 — PSICHIC Comparison (TODO — not yet implemented)

PSICHIC is the Nature Machine Intelligence 2024 gold standard.
It must be included as a baseline for the paper.

**Reference**: PSICHIC paper — sequence-only interpretable binding affinity.
**GitHub**: search "PSICHIC binding affinity" on GitHub.

To add it:
1. Clone/install PSICHIC
2. Create `models/psichic_wrapper.py` that wraps their inference
3. Register `register("psichic_baseline", group="psichic", ...)`
4. Write `run_psichic(exp_name)` in `run_experiment.py`
5. Add to `LEAKYPDB_EXPERIMENTS` in `run_all.py`

---

## Step 6 — Paper Writing Checklist

When experiments are done, check these before writing:

```python
# Minimum results needed for paper:
# 1. Graphormer or GLI beats all baselines on BindingDB test RMSE
# 2. DSM auxiliary loss improves BiCA by >0.02 RMSE
# 3. Phase1 ranking loss improves Spearman R by >0.01
# 4. LeakyPDB results confirm scaffold generalization
# 5. Multiseed std < 0.02 RMSE (stability)
# 6. Interpretability figures show chemically meaningful patterns

python -c "
import pandas as pd
df = pd.read_csv('diary/results_diary.csv')
# Check Phase 2 beats baseline
p2 = df[df['experiment_id'].str.startswith(('graphormer','gli','bica_chemberta_esm2_8M_dsm'))]
base = df[df['experiment_id'] == 'bica_chemberta_esm2_8M']
print('Phase2 best:', p2['test_rmse'].min())
print('BiCA baseline:', base['test_rmse'].values)
"
```

---

## Experiment Groups Reference

| Group | Model families | Key experiments |
|-------|---------------|-----------------|
| baselines | linear | ridge_ecfp4_aac |
| trees | tree | xgb_chemberta_esm2_8M |
| mlp | mlp | mlp_chemberta_esm2_8M |
| pretrained | mlp+tree | mlp_chemberta_esm2_35M |
| bica | bica | bica_chemberta_esm2_8M |
| graph_models | gcn, gat | gat_ecfp_esm2_8M |
| seq_models | lstm, transformer_seq | lstm_smiles_bpe1000_protein_bpe1000 |
| distmat | distmat_cnn | distmat_cnn_esm2_8M |
| phase1_ranking | mlp,bica,gat | *_ranked |
| phase1_recon | gcn,gat | *_recon |
| phase2_graphormer | graphormer | graphormer_mol_esm2_8M ← key |
| phase2_gli | gli | gli_mol_esm2_8M ← key |
| phase2_dsm | bica,mlp | *_dsm |
| phase2_mamba | mamba | mamba_smiles_bpe1000_protein_bpe1000 |
| targeted | all families | 65 best repr combos |

---

## Troubleshooting

**`KeyError: 'CL1'`** — stale LeakyPDB cache. Delete and retry:
```bash
rm cache/leakypdb_raw.pkl cache/splits/leakypdb_split.pkl
python run_experiment.py --exp ridge_ecfp4_aac --dataset leakypdb
```

**`KMP_DUPLICATE_LIB_OK` warning** — already suppressed in code, safe to ignore.

**CUDA out of memory** — reduce batch size in `harness/config.py`:
```python
BATCH_SIZE = 32   # default 64
```

**Graphormer slow** — it runs per-molecule BFS for SPD matrix on CPU.
This is expected for large datasets. Use smaller batch size for inference.

**`mamba-ssm` not installed** — Mamba falls back to BiLSTM with a warning.
To install: `pip install mamba-ssm` (requires CUDA + specific torch version).

**Experiment already in diary but re-runs** — check predictions exist:
```bash
ls cache/predictions/ | grep <exp_name>
```
If `.npz` missing but diary entry exists, delete the diary row or just re-run.

**Check what's registered**:
```bash
python run_experiment.py --list | grep phase2
python run_experiment.py --list | grep targeted
```

---

## Key Hyperparameters (harness/config.py)

```python
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS    = 200
PATIENCE      = 20       # early stopping
SPLIT_SEED    = 42
WEIGHT_DECAY  = 1e-4
```

---

## Adding a New Experiment

1. Register it in `run_experiment.py`:
```python
register("my_new_exp",
    group="my_group",
    model_family="mlp",          # or gat, bica, graphormer, gli, mamba, ...
    ligand_repr="chemberta_600",
    protein_repr="esm2_8M_320",
    fusion="concat",
    mlp_arch="deep",
    notes="My new experiment",
)
```

2. Run it:
```python
python run_experiment.py --exp my_new_exp
```

3. Results appear in `diary/results_diary.csv` automatically.

---

## Git / Version Control

```bash
# Clone
git clone <repo-url>
cd drug-discovery-benchmark

# Check status
git status
git log --oneline
```
