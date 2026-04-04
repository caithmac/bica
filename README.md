# Drug-Target Binding Affinity Benchmark

A systematic benchmark of machine learning models for drug-target binding affinity
prediction, evaluated on BindingDB with a rigorous scaffold-based split. The central
contribution is **BiCA v2** — a Bidirectional Cross-Attention model that performs
joint reasoning over protein residue sequences and ligand token sequences.

---

## Contents

```
harness/          Data loading, featurizers, training loops, metrics, logging
models/           All model implementations (see below)
interpret/        Interpretability library — attention maps, Integrated Gradients, SHAP
run_experiment.py Experiment registry and runners (333 experiments total)
run_all.py        Master pipeline — runs all phases, resumable
run_psichic_benchmark.py  PSICHIC baseline evaluation
analyze_results.py        Leaderboard and analysis
bootstrap_ci.py           Bootstrap confidence intervals
diary/            Results, figures, and findings
  results_diary.csv       All 333 experiment results
  bootstrap_ci.csv        95% CIs for top models
  FINDINGS.md             Full leaderboard and analysis
  figures/                Attention maps, interpretability figures
```

---

## Dataset and Split

**Dataset:** BindingDB_filtered from the [BALM benchmark](https://huggingface.co/datasets/BALM/BALM-benchmark)
— 24,700 protein-ligand pairs with experimentally measured pKd values.

**Split:** Bemis-Murcko scaffold split (seed 42), ensuring test compounds have
structurally distinct scaffolds from training data. This is a harder and more
realistic evaluation than random splits.

| Partition | Size |
|-----------|------|
| Train     | 17,312 |
| Val       | 2,673  |
| Test      | 4,715  |

**Metric:** RMSE on pKd (primary), Pearson R, Spearman R.

---

## Results

Best result per model family (seed 42, scaffold split):

| Model | RMSE ↓ | Pearson R ↑ | Spearman R ↑ |
|-------|--------|-------------|--------------|
| RF (ECFP4 + AAC) | **1.007** | 0.747 | 0.674 |
| XGBoost (ChemBERTa + ESM-2) | 1.047 | 0.722 | 0.652 |
| LightGBM (ECFP4 + AAC) | 1.053 | 0.720 | 0.640 |
| MLP (ChemBERTa + ESM-2) | 1.101 | 0.703 | 0.623 |
| **BiCA v2** (ChemBERTa-77M + ESMC) | **1.102** | **0.702** | **0.631** |
| PSICHIC fine-tuned | 1.176 | 0.631 | 0.548 |
| PSICHIC zero-shot | 1.787 | 0.456 | 0.360 |

Multi-seed stability (seeds 42 / 123 / 456):

| Model | Mean RMSE | Std |
|-------|-----------|-----|
| RF (ECFP4 + AAC) | 1.044 | ±0.035 |
| BiCA v2 | 1.108 | ±0.037 |

---

## BiCA v2 — Bidirectional Cross-Attention

### Motivation

Most binding affinity models use pretrained embeddings as fixed feature vectors —
a protein is reduced to a single mean-pooled vector before any ligand information
is seen. This discards the spatial structure of the sequence. BiCA v2 instead keeps
both sequences intact and lets them attend to each other, so the model can learn
*which residues matter for which ligand atoms* rather than averaging everything away.

### Architecture

```
Protein sequence          Ligand SMILES
(L_prot residues)         (L_lig tokens)
      │                        │
  ESM-2 encoder            ChemBERTa-2
  per-residue 480-dim      per-token  384-dim
      │                        │
  Linear proj              Linear proj
  → hidden_dim (256)       → hidden_dim (256)
      │                        │
      └──────────┬─────────────┘
                 │
     ┌───────────▼───────────┐
     │  CrossAttentionBlock  │  × 2 layers
     │                       │
     │  p→l: protein queries │  A_p2l ∈ (L_prot × L_lig)
     │        ligand keys/val│  "which ligand tokens does
     │                       │   each residue attend to?"
     │  l→p: ligand queries  │  A_l2p ∈ (L_lig × L_prot)
     │        protein keys/val  "which residues does each
     │                       │   ligand token attend to?"
     │  Pre-LayerNorm        │
     │  FFN (GELU, 4× dim)   │
     │  DropPath residuals   │
     └───────────┬───────────┘
                 │
        ┌────────┴────────┐
        │                 │
   AttentionPool      AttentionPool
   (protein)          (ligand)
   scalar α_i         scalar β_j
   per residue        per token
        │                 │
   prot_vec (256)    lig_vec (256)
        └────────┬────────┘
                 │  concat → 512-dim
                 │
           Predictor MLP
           512 → 256 → 1
                 │
               pKd̂
```

**Key design choices:**

- **True sequence attention** — not attention over a single CLS token, but over all
  `L_prot` residues and all `L_lig` tokens simultaneously.
- **Bidirectional** — protein-to-ligand *and* ligand-to-protein cross-attention in
  each block. Both sides update based on the other.
- **Pre-LayerNorm** — normalise before attention (more stable training than post-norm).
- **Value-weighted attention** — at inference, attention weights are scaled by the L2
  norm of the corresponding Value vectors, suppressing "sink tokens" that attract
  high attention but carry little information.
- **Gated AttentionPool** — replaces mean pooling with a learned scalar importance
  weight per position, giving interpretable residue/token importance scores (S3/S4).

### Interpretability

Three complementary attribution signals are extracted without retraining:

| Signal | What it captures |
|--------|-----------------|
| **S3 — AttentionPool (protein)** | Scalar importance α_i per residue; which residues the model relies on for the final prediction |
| **S4 — AttentionPool (ligand)** | Scalar importance β_j per token; which SMILES substrings matter |
| **Value-weighted S1** | Cross-attention A_p2l scaled by ‖V_j‖₂; residue × token interaction map with sink tokens suppressed |
| **Integrated Gradients** | Gradient-based attribution in embedding space via [Captum](https://captum.ai/) |
| **Consensus** | c_i = α_i^(IG) × w_i^(S3) — product of IG and AttentionPool, highlighting residues confirmed by both methods |

Running per-compound interpretability analysis:
```bash
python interpret_bica_per_compound.py
# outputs: diary/figures/per_compound/
```

This produces residue-level attribution profiles, residue×token cross-attention
heatmaps, value-weighted maps, and a consensus heatmap across all test compounds
for the protein with the most ligands.

### Ablations

All ablation variants are registered in `run_experiment.py` and share the same
forward signature:

| Variant | Change | RMSE |
|---------|--------|------|
| BiCA v2 (full) | — | 1.102 |
| BiCA v2 MeanPool | Replace AttentionPool with mean pool | +0.02 |
| BiCA v2 SingleLayer | 1 cross-attention layer instead of 2 | +0.10 |
| BiCA v2 NoFFN | Remove feed-forward block | +0.10 |
| BiCA v2 P2L only | Unidirectional protein→ligand only | — |
| SimpleConcatBaseline | Same projections/MLP, no attention | — |

### Code

The full model is in [`models/bica_v2.py`](models/bica_v2.py). Key classes:

- `AttentionPool` — learned pooling with optional masking
- `CrossAttentionBlock` — one bidirectional cross-attention layer with FFN
- `BiCA_v2` — full model stacking N blocks
- `build_bica_v2(protein_dim, ligand_dim, **kwargs)` — factory function

Minimal usage:
```python
from models.bica_v2 import build_bica_v2
import torch

model = build_bica_v2(
    protein_dim = 480,   # ESM-2 35M per-residue dim
    ligand_dim  = 384,   # ChemBERTa-77M per-token dim
    hidden_dim  = 256,
    num_heads   = 8,
    num_layers  = 2,
    dropout     = 0.3,
)

# protein_seq: (B, L_prot, 480)  — per-residue ESM-2 embeddings
# ligand_seq:  (B, L_lig,  384)  — per-token ChemBERTa embeddings
pred = model(protein_seq, ligand_seq)          # (B, 1)

# With attention weights for interpretability:
pred, attn = model(protein_seq, ligand_seq, return_attention=True)
# attn keys: p2l_weights, l2p_weights, prot_pool_weights, lig_pool_weights
```

---

## Other Models

All 12 model families evaluated:

| Family | Implementation | Best RMSE |
|--------|---------------|-----------|
| Linear (Ridge) | `models/sklearn_models.py` | 1.254 |
| Tree (RF / XGB / LGBM) | `models/sklearn_models.py` | 1.007 |
| MLP | `models/mlp.py` | 1.101 |
| CNN-1D (SMILES) | `models/cnn.py` | 1.324 |
| DistMat CNN | `models/distmat_cnn.py` | 1.109 |
| LSTM / Transformer-seq | `models/sequence_models.py` | 1.146 |
| GCN / GAT | `models/gnn.py` | 1.194 |
| Graphormer | `models/graphormer.py` | — |
| GLI (Gated Local-Global) | `models/gli.py` | — |
| BiCA (v1, flat vectors) | `models/bica.py` | 1.102 |
| BiCA v2 (sequence inputs) | `models/bica_v2.py` | 1.102 |
| PSICHIC | external + `PSICHIC/psichic_runner.py` | 1.176 |

---

## Reproducing Results

**Environment:**
```bash
conda create -n drug_discovery python=3.11
conda activate drug_discovery
pip install -r requirements.txt
# For GNN models:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

**Run everything** (resumes automatically if interrupted):
```bash
python run_all.py
```

**Run a single experiment:**
```bash
python run_experiment.py --exp rf_ecfp4_aac
python run_experiment.py --exp bica_v2_chemberta77M_tokens
```

**PSICHIC baseline:**
```bash
python run_psichic_benchmark.py --mode zero_shot
python run_psichic_benchmark.py --mode fine_tune --ft_iters 5000
```

**Analyse results:**
```bash
python analyze_results.py        # prints leaderboard
python bootstrap_ci.py           # 95% CIs → diary/bootstrap_ci.csv
```

All results are appended to `diary/results_diary.csv`.
Data is downloaded automatically from HuggingFace (`BALM/BALM-benchmark`) on first run.

---

## Dependencies

Core: `torch`, `torch_geometric`, `transformers`, `rdkit`, `scikit-learn`,
`xgboost`, `lightgbm`, `captum`, `fair-esm`, `datasets`, `pandas`, `scipy`

See `requirements.txt` for pinned versions.
