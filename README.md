# Drug-Target Binding Affinity Benchmark

A systematic benchmark of machine learning models for drug-target binding affinity
prediction, evaluated on BindingDB under a rigorous scaffold-based split. The central
finding: **tree ensembles with simple fingerprints outperform every deep learning
architecture tested**, including transformers, GNNs, and cross-attention models.

Two cross-attention models (BiCA v1 and BiCA v2/ChemCross) are included. BiCA v2
performs bidirectional cross-attention over protein residue sequences and ligand
token sequences, producing interpretable attention maps.

---

## Repository Structure

```
harness/          Data loading, featurizers, training loops, metrics, logging
models/           All model implementations
interpret/        Interpretability — attention maps, Integrated Gradients, SHAP
evaluation/       ASAP Polaris challenge evaluation protocol
experiments/      Standalone experiment scripts (ASAP, cross-dataset, learning curves)
scripts/          Analysis, figures, leaderboard generation, statistics
run_experiment.py Main experiment registry and runners
run_all.py        Master pipeline — runs all phases, resumable
diary/            Results diary, findings
  results_diary.csv   All experiment results
  diary_clean.csv     Cleaned/deduplicated results
  FINDINGS.md         Full leaderboard and analysis
figures/          Paper figures (PDF)
manuscript/       LaTeX source for the paper
docs/             Supplementary documentation
```

---

## Dataset and Split

**Dataset:** BindingDB_filtered from the [BALM benchmark](https://huggingface.co/datasets/BALM/BALM-benchmark)
— 24,700 protein-ligand pairs with experimentally measured pKd values.

**Split:** Bemis-Murcko scaffold split (seed 42), ensuring test compounds have
structurally distinct scaffolds from training. This is harder and more realistic
than random splits.

| Partition | Size |
|-----------|------|
| Train     | 17,312 |
| Val       | 2,673 |
| Test      | 4,715 |

**Metric:** RMSE on pKd (primary), Pearson R, Spearman R.

---

## Results

Best result per model family (seed 42, scaffold split, N=285 unique experiments):

| Model | RMSE ↓ | Pearson R ↑ |
|-------|--------|-------------|
| RF (ECFP4 + AAC) | **1.006** | 0.747 |
| XGBoost (ECFP4 + ESM-2 8M) | 1.048 | 0.721 |
| LightGBM (ECFP4 + AAC) | 1.053 | 0.720 |
| MLP (ChemBERTa-5M + ESM-2 8M) | 1.074 | 0.716 |
| BiCA v2 (ChemBERTa-77M + ESMC 300M) | 1.132 | 0.684 |
| BiCA v1 (ChemBERTa-600 + ESM-2 8M) | 1.146 | 0.676 |
| GP (Tanimoto kernel) | 1.179 | 0.642 |
| PSICHIC fine-tuned | 1.176 | 0.631 |
| LSTM (atom SMILES + char protein) | 1.146 | 0.654 |
| Mamba | 1.158 | 0.648 |
| CNN (distance matrix) | 1.109 | 0.690 |
| GAT | 1.194 | 0.598 |
| GCN | 1.202 | 0.640 |
| Transformer (seq) | 1.210 | 0.606 |
| Ridge (linear) | 1.254 | 0.522 |

Multi-seed stability (seeds 42/99/123/456):

| Model | Mean RMSE | Std |
|-------|-----------|-----|
| RF (ECFP4 + AAC) | 1.041 | ±0.028 |
| MLP (ChemBERTa-5M + ESM-2 8M) | 1.127 | ±0.038 |

---

## BiCA v2 — Bidirectional Cross-Attention

### Architecture

```
Protein sequence          Ligand SMILES
(L_prot residues)         (L_lig tokens)
      │                        │
  ESM-2 encoder            ChemBERTa-2
  per-residue              per-token
      │                        │
  Linear proj              Linear proj
      │                        │
      └──────────┬─────────────┘
                 │
     ┌───────────▼───────────┐
     │  CrossAttentionBlock  │  × 2 layers
     │  p→l + l→p attention  │
     │  Pre-LayerNorm        │
     │  FFN (GELU, 4× dim)   │
     │  DropPath residuals   │
     └───────────┬───────────┘
                 │
        ┌────────┴────────┐
   AttentionPool      AttentionPool
   (protein)          (ligand)
        │                 │
        └────────┬────────┘
                 │  concat
           Predictor MLP
                 │
               pKd̂
```

**Key design choices:**
- **Bidirectional cross-attention** — protein↔ligand attention in each block
- **Pre-LayerNorm** — normalise before attention (more stable)
- **Value-weighted attention** — attention weights scaled by ‖V‖₂ at inference
- **Gated AttentionPool** — learned scalar importance per position, producing interpretable residue/token scores

### Interpretability

Three complementary signals extracted without retraining:

| Signal | What it captures |
|--------|-----------------|
| S3 — AttentionPool (protein) | Residue-level importance α_i |
| S4 — AttentionPool (ligand) | Token-level importance β_j |
| Value-weighted S1 | Residue × token cross-attention with sink suppression |
| Integrated Gradients | Gradient-based attribution via Captum |
| Consensus | Product of IG and AttentionPool weights |

### Code

Full model in [`models/bica_v2.py`](models/bica_v2.py). Minimal usage:

```python
from models.bica_v2 import build_bica_v2

model = build_bica_v2(
    protein_dim=480,   # ESM-2 35M per-residue dim
    ligand_dim=384,    # ChemBERTa-77M per-token dim
    hidden_dim=256, num_heads=8, num_layers=2, dropout=0.3,
)

pred = model(protein_seq, ligand_seq)                     # (B, 1)
pred, attn = model(protein_seq, ligand_seq, return_attention=True)
# attn keys: p2l_weights, l2p_weights, prot_pool_weights, lig_pool_weights
```

---

## Other Models

| Family | Implementation | Best RMSE |
|--------|---------------|-----------|
| Linear (Ridge) | `models/sklearn_models.py` | 1.254 |
| Tree (RF / XGB / LGBM) | `models/sklearn_models.py` | 1.006 |
| MLP | `models/mlp.py` | 1.074 |
| CNN (SMILES 1D) | `models/cnn.py` | 1.324 |
| DistMat CNN | `models/distmat_cnn.py` | 1.109 |
| LSTM / Transformer-seq | `models/sequence_models.py` | 1.146 |
| GCN / GAT | `models/gnn.py` | 1.194 |
| Graphormer | `models/graphormer.py` | — |
| BiCA (flat vectors) | `models/bica.py` | 1.146 |
| BiCA v2 (sequence) | `models/bica_v2.py` | 1.132 |

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

**Run everything** (resumes if interrupted):
```bash
python run_all.py
```

**Single experiment:**
```bash
python run_experiment.py --exp rf_ecfp4_aac
python run_experiment.py --exp bica_v2_chemberta77M_tokens
```

**Standalone experiments:**
```bash
python experiments/run_asap_potency.py    # ASAP Polaris challenge
python experiments/run_learning_curves.py # RF vs MLP learning curves
python experiments/cross_dataset_bench.py # Cross-dataset evaluation
```

**Analysis:**
```bash
python scripts/gen_leaderboard.py     # Print leaderboard
python scripts/bootstrap_ci.py        # Bootstrap CIs
python scripts/advanced_stats.py      # ANOVA, Tukey HSD, permutation tests
python scripts/make_figures.py        # Generate paper figures
```

All results appended to `diary/results_diary.csv`. Data auto-downloaded from HuggingFace on first run.

---

## Dependencies

Core: `torch`, `torch_geometric`, `transformers`, `rdkit`, `scikit-learn`,
`xgboost`, `lightgbm`, `captum`, `fair-esm`, `datasets`, `pandas`, `scipy`

See `requirements.txt` for pinned versions.
