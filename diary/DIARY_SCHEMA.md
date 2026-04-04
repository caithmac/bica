# Experiment Diary Schema

All results are logged to `results_diary.csv`. This file is append-only — entries are never deleted.

## Column Reference

| Column | Description |
|---|---|
| timestamp | UTC datetime of when the experiment was logged |
| experiment_id | Unique experiment name (format: `model_ligandrepr_protrepr`) |
| model_name | Same as experiment_id |
| model_family | `linear` / `tree` / `mlp` / `cnn` / `transformer` |
| ligand_repr | Ligand featurization method |
| protein_repr | Protein featurization method |
| fusion_strategy | How ligand + protein features are combined |
| n_params | Trainable parameter count (N/A for sklearn) |
| epochs_trained | Epochs before early stopping (N/A for sklearn) |
| batch_size | Mini-batch size (N/A for sklearn) |
| learning_rate | Optimizer LR (N/A for sklearn) |
| split_type | Always `scaffold_bemis_murcko` |
| n_train / n_val / n_test | Dataset split sizes |
| val_rmse | Validation RMSE (pKd units) |
| val_pearson_r | Validation Pearson correlation |
| val_spearman_r | Validation Spearman rank correlation |
| test_rmse | **Primary metric** — Test RMSE (pKd units) |
| test_pearson_r | Test Pearson correlation |
| test_spearman_r | Test Spearman rank correlation |
| train_time_sec | Wall-clock training time in seconds |
| notes | Any free-text notes |

## Representation Vocabulary

**Ligand (fixed-size vectors — flat feature path):**
- `ecfp2_1024` — Morgan radius-1, 1024 bits
- `ecfp4_1024` — Morgan radius-2, 1024 bits (standard)
- `ecfp6_1024` — Morgan radius-3, 1024 bits
- `maccs_167` — MACCS 167-bit keys
- `rdkit_200` — 200 RDKit physicochemical descriptors
- `smiles_onehot` — Character-level one-hot, max 100 chars × 39 vocab (flattened 3900-dim)
- `chemberta_600` — Mean-pooled ChemBERTa-2 embeddings (dim=600)
- `distmat_100` — Topological distance matrix (RDKit GetDistanceMatrix), (100×100) padded, normalised

**Ligand (graph — GNN path):**
- `mol_graph` — Molecular graph (atoms=nodes, bonds=edges). Node: 78-dim, Edge: 10-dim. Used only with GCN/GAT runners.

**Protein (fixed-size vectors — flat feature path):**
- `aac_20` — Amino acid composition (20 features)
- `dipeptide_400` — Dipeptide composition (400 features)
- `kmer3_8000` — Tri-mer frequency hashing (8000 features)
- `esm2_8M_320` — ESM-2 8M mean-pooled embeddings (dim=320)
- `esm2_35M_480` — ESM-2 35M mean-pooled embeddings (dim=480)
- `prot_electra_256` — ProtElectra (RTD/ELECTRA pre-trained on BFD), mean-pooled (dim=256). Model: `Rostlab/prot_electra_generator_bfd`

**Ligand (token ID sequences — seq_models path):**
- `smiles_char` — Character-level: 1 ASCII char = 1 token (~40 vocab)
- `smiles_atom` — Atom-level: multi-char atoms (Br, Cl, [NH2+]) as 1 token (~60 vocab)
- `smiles_bpe_512` — BPE trained on dataset, vocab=512
- `smiles_bpe_1000` — BPE trained on dataset, vocab=1000

**Protein (token ID sequences — seq_models path):**
- `protein_char` — Character-level: 1 amino acid letter = 1 token (20+special vocab)
- `protein_bpe_512` — BPE trained on dataset, vocab=512
- `protein_bpe_1000` — BPE trained on dataset, vocab=1000
- `protein_wordpiece_512` — WordPiece trained on dataset, vocab=512

## Model Family Vocabulary

| `model_family` | Description |
|---|---|
| `linear` | Ridge / Lasso / ElasticNet |
| `tree` | Random Forest, XGBoost, LightGBM |
| `mlp` | Multi-layer perceptron (flat concat input) |
| `cnn` | 1D CNN on SMILES one-hot |
| `lstm` | Bidirectional LSTM dual-encoder |
| `transformer` | Transformer on flat features or cross-attention |
| `transformer_seq` | Transformer with learned token embeddings |
| `distmat_cnn` | 2D CNN on topological distance matrix (new) |
| `gcn` | Graph Convolutional Network via PyG (new) |
| `gat` | Graph Attention Network via PyG (new) |
| `bica` | Bidirectional Cross-Attention model (new) |

## Fusion Strategy Vocabulary

| `fusion_strategy` | Description |
|---|---|
| `concat` | Feature vectors concatenated before model |
| `dual_encoder` | Two separate encoders; representations concatenated |
| `cross_attention` | One-directional cross-attention (lig queries prot) |
| `bidirectional_cross_attn` | Bidirectional cross-attention (BiCA) |
| `distmat` | 2D CNN processes distance matrix; prot vector concatenated |
| `graph` | GNN processes molecular graph; prot vector concatenated after pooling |

## Interpreting Results

- Lower RMSE is better (units: pKd, so ~0.5–1.0 is reasonable, <0.5 is excellent)
- Higher Pearson / Spearman is better (closer to 1.0)
- All results use the same scaffold split (seed=42, 70/10/20 train/val/test)

---

## Critical Analysis & Known Limitations

This section documents methodological concerns identified via systematic critical review.
Use it to calibrate confidence in any conclusions drawn from the diary.

### What the results *can* support (proportionate claims)

| Claim | Confidence | Basis |
|---|---|---|
| XGB + pre-trained embeddings has the best point estimate on this benchmark | ✅ Strong | Consistent across multiple tree models |
| ESM-2 improves every architecture tested | ✅ Strong | Consistent improvement across 8 model families |
| BPE-512 > WordPiece for sequence models on this dataset | ✅ Strong | Consistent across both LSTM and TransformerSeq |
| Scaffold split is hard — best RMSE ~1.44, Pearson ~0.57 | ✅ Strong | Consistent across all 42 experiments |
| Distmat CNN is compute-inefficient | ✅ Strong | Worst RMSE/compute ratio, 690–850s training |
| GNNs underperform fingerprint models *on this benchmark* | ⚠️ Scoped | True for shallow, untrained GNNs on scaffold split; not a general statement |
| BiCA adds no value over MLP *with flat-vector inputs* | ⚠️ Scoped | Untested with true sequence inputs (atom tokens, residue tokens) |
| ProtElectra ≈ ESM-2 8M | ⚠️ Weak | Based on a single paired comparison; needs replication |

### Methodological concerns

**1. Single dataset, single split seed (high priority)**
All 42 experiments use one dataset (BindingDB_filtered) and one split seed (42).
Rankings are specific to this chemotype distribution and this particular scaffold assignment.
*To fix:* Re-run fast experiments (baselines, trees, MLP) across seeds 42/123/456 and report mean ± std RMSE.

**2. Confounded representation comparison (high priority)**
The top model (`xgb_chemberta_esm2_8M`) bundles a better ligand encoder *and* a better protein encoder.
The gain cannot be attributed to either factor independently.
*Missing ablations:* `xgb_ecfp4_esm2_8M` and `xgb_chemberta_aac` — needed to cleanly isolate protein vs. ligand representation gain from model family choice.

**3. No significance testing between models (medium priority)**
The top-10 models span only 0.06 RMSE units (1.443–1.503). On 4,938 test samples, many pairwise differences may not be statistically distinguishable. No bootstrap confidence intervals are reported.
*To fix:* Bootstrap test set 1000× on saved predictions (no re-training needed) to get 95% CIs per model.

**4. No hyperparameter optimisation (medium priority)**
All neural models use fixed lr=1e-3, batch=256, fixed architecture variants. Comparisons conflate architecture quality with hyperparameter luck. A well-tuned MLP may beat a poorly-initialised GNN regardless of their true capability.

**5. GNN evaluation is structurally disadvantaged (medium priority)**
GCN/GAT use 3-layer, hidden=128, randomly-initialised networks. Literature shows molecular GNNs need pre-training (GIN/GROVER), deeper architectures (5–6 layers), and virtual node augmentation to compete with ECFP. The finding is "shallow GNNs lose to ECFP on scaffold split" — not "GNNs are worse than fingerprints".

**6. BiCA comparison uses mismatched baseline (medium priority)**
`bica_ecfp4_aac` (1.36M params) is compared against `mlp_shallow_ecfp4_aac` (268K params).
The fair comparison is `bica_chemberta_esm2_8M` vs. `mlp_chemberta_esm2_8M` (both ~1.4–1.8M params) — BiCA loses there too (1.545 vs. 1.487), which is the valid finding.

**7. Multiple comparisons without correction (low priority)**
42 experiments × 3 metrics = 126 test statistics with no correction applied. Conclusions derived from subgroup averages (tokenization comparison, repr comparison) are exploratory, not confirmatory.

**8. Label noise floor (contextual)**
BindingDB pKd values originate from heterogeneous assays (Ki, Kd, IC50 converted). Estimated label noise: ~0.3–0.5 pKd units. This sets a practical lower bound on achievable RMSE — no model can score below ~0.3–0.5 regardless of architecture. The best models (RMSE ~1.44) are still ~1.0 units above the noise floor; the remaining gap is genuine generalisation difficulty, not noise.

**9. Dataset bias (contextual)**
BindingDB over-represents kinases (~40% of entries). Models encoding protein family information (ESM-2, ProtElectra) benefit from kinase-family similarity even across scaffold boundaries. Results may not generalise to GPCRs, proteases, or nuclear receptors.

### Recommended next experiments (priority order)

1. **Multi-seed evaluation** — rerun fast models (baselines, trees, MLP-medium) on seeds 42/123/456
2. **Missing ablations** — add `xgb_ecfp4_esm2_8M` and `xgb_chemberta_aac` to complete the 2×2 factorisation
3. **Bootstrap CIs** — post-hoc script on saved predictions, no re-training needed
4. **BiCA with sequence inputs** — feed atom-level SMILES tokens + residue sequences; this is the architecturally intended use case
5. **Pre-trained GNN** — replace randomly-initialised GCN/GAT with OGB-pretrained GIN for a fair graph vs. fingerprint comparison

---

## Nature-Level Publication Plan

*Recorded 2026-04-01. Target venue: Nature Communications or Nature Machine Intelligence.*

This section documents the agreed roadmap toward a high-impact publication. It covers: (A) new model architectures from literature, (B) interpretability strategy, (C) documentation and reproducibility standards.

---

### A. Model Architecture Additions (from literature review)

The following new model families will be added to the benchmark. Each is grounded in a specific limitation identified in the Critical Analysis section above.

#### A1. Pairwise Ranking Loss — from Multi-task Bioassay Pre-training (MBP)
**Motivation.** All current models train on MSE but are evaluated on Spearman R. A pairwise ranking loss directly optimises rank correlation, the scientifically more meaningful metric for virtual screening.
**Implementation.** New `harness/losses.py` with `PairwiseRankingLoss`. Augments `trainer.py`'s `train_torch` with an optional `aux_loss` argument. Applied to top-3 deep models: `mlp_chemberta_esm2_8M`, `bica_chemberta_esm2_8M`, `gat_ecfp_esm2_8M`.
**New experiment IDs:** `*_ranked` suffix.
**Effort:** Low. **Expected gain:** +0.02–0.04 Spearman.

#### A2. Graphormer — graph-structured transformer for molecules
**Motivation.** GNNs (RMSE ~1.56) trail ECFP (RMSE ~1.42) because message-passing is local. Graphormer gives every atom global attention with three additive structural biases: degree centrality, shortest-path distance (already in distmat_cnn.py), and edge-type-on-path. All computable from 2D SMILES.
**Implementation.** New `models/graphormer.py` with `GraphormerLayer` (custom attention + bias injection) and `GraphormerBindingModel`. Protein vector concatenated after CLS-pooled graph transformer output. Two experiments: `graphormer_aac`, `graphormer_esm2_8M`.
**Effort:** Medium-high. **Expected impact:** Best graph-based model in the benchmark.

#### A3. GLI-style Gated Graph + Cross-Attention Fusion — from Joint Global-Local Interaction Modeling
**Motivation.** BiCA degenerates with seq_len=1 flat vectors (Concern §6 above). GNNs pool molecules but ignore protein residue-level interaction. GLI fuses both via a learned sigmoid gate: GNN per-atom hidden states (local) + BiCA cross-attention (global).
**Implementation.** New `models/gli.py` with `GLIBindingModel`. Reuses existing `GNNDataset` (graph + protein vector) and `BiCA_VariableHeads`. Gate: `g = sigmoid(W * [mol_repr; bica_repr]); fused = g ⊙ mol_repr + (1-g) ⊙ bica_repr`.
**Effort:** Medium. **Expected impact:** Best interaction model; directly addresses §6.

#### A4. DualBind DSM Auxiliary Loss
**Motivation.** MSE training gives equal weight to all examples. DSM forces the encoder to learn a smooth binding energy surface by denoising corrupted embeddings, regularising the latent space without additional data.
**Implementation.** New `models/dsm.py` with `DSMAuxHead`. Models (`bica.py`, `mlp.py`) gain `encode()` / `head()` split. Total loss: `L = L_MSE + λ * L_DSM`. Five noise levels σ from 0.01–1.0, geometric schedule.
**Effort:** Medium. **Expected gain:** +0.02–0.04 RMSE on pretrained-repr models.

#### A5. GraphMAE Node Reconstruction Auxiliary Loss
**Motivation.** GNN node representations trained solely on binding affinity receive sparse gradients. Masking 15% of atom features and adding reconstruction loss improves intermediate representations (GraphMAE, NeurIPS 2022).
**Implementation.** Modify `models/gnn.py` (masking + NodeDecoder head). Modify `gnn_trainer.py` (dual-forward: masked graph for recon, original for affinity). New experiments: `gcn_esm2_8M_recon`, `gat_esm2_8M_recon`.
**Effort:** Low-medium. **Expected gain:** ~0.02–0.03 RMSE on GNN family.

#### A6. Mamba SSM Encoder — from Mamba (selective state space models)
**Motivation.** Protein sequences are truncated to 512 tokens to manage O(N²) attention cost. Mamba provides O(N) selective SSM for full-length sequences. Also adds SSM as an architecture category to the benchmark.
**Implementation.** New `MambaEncoder` and `MambaBindingModel` in `models/sequence_models.py`, same interface as `LSTMEncoder`. Registered as `model_family="mamba"`.
**Effort:** Medium. **Expected gain:** Marginal RMSE improvement; benchmark completeness.

**Papers skipped (require 3D coordinates not available in our setup):** FABind, IPBind, DAGML, ImageBind. DPO skipped (MBP ranking loss is the correct regression analogue). Data2vec skipped (poor effort-to-impact ratio; frozen pretrained embeddings already dominate).

#### A7. PSICHIC as a Baseline Comparison (not re-implemented — run their public model)
**What it is.** PSICHIC (Hao et al., *Nature Machine Intelligence* 2024) is the current best sequence-only binding affinity model with built-in interpretability. It uses physicochemical graph embeddings and works from sequence alone.
**Why we need it.** Any Nature-tier paper in this space must benchmark against PSICHIC. Reviewers will ask "how does this compare to PSICHIC?"
**Action.** Run the official PSICHIC checkpoint on our BindingDB and LeakyPDB test sets and log the results to `results_diary.csv` as `psichic_official`. Record their published metrics (RMSE, Pearson, Spearman) if the model does not run on our exact split. Note any distributional differences.
**New experiment ID:** `psichic_official`

---

### B. Interpretability Strategy (Nature-level standard)

Based on review of published Nature-tier papers in this space (see key papers below), interpretability must satisfy two criteria for high-impact publication:
1. **Chemical validity**: highlighted atoms/residues must correspond to known pharmacophoric features or binding-site residues, validated against PDB crystal structures or mutagenesis data.
2. **Quantitative evaluation**: fidelity (masking top-attributed features should degrade accuracy), stability across similar compounds, and sparsity.

Three complementary interpretability methods will be implemented, covering all model families:

#### B1. Cross-attention Maps (for BiCA and GLI models)
**What it shows:** A (protein_residue × ligand_atom) 2D heatmap from the bidirectional cross-attention layer. This is the most information-rich interpretability output and dominates recent high-impact DTA papers (CAPLA, CoaDTI, AttentionMGT-DTA, DeepDTAGen Nature Comms 2025).
**Implementation.** Extract attention weight tensors during inference from `BiCA_VariableHeads.protein2ligand` and `ligand2protein` attention layers. Average across heads. Output: per-(residue, atom) score matrix. Visualize: residue scores on protein sequence (bar plot), atom scores rendered on 2D RDKit structure depiction.
**Validation.** Compare top-10 residue attributions against known binding-site residues from PDBBind (LeakyPDB dataset comes from PDBBind v2020 — use the `header` column for PDB IDs, then query PDB for binding site annotation).
**Tool:** `nn.MultiheadAttention` already returns `attn_weights` as second output; we just need to capture it during inference.

#### B2. SHAP Substructure Attribution (for tree and MLP models)
**What it shows:** Which bits of ECFP4 (or dimensions of ChemBERTa) contribute most to the predicted pKd. ECFP bit → substructure via RDKit's `GetMorganGenerator` `GetBitInfo()`. Renders as a 2D molecular depiction with atom-level colour coding.
**Implementation.** `shap.TreeExplainer` for XGBoost/RF (exact Shapley values). `shap.DeepExplainer` or `shap.GradientExplainer` for MLP. Aggregate per-atom SHAP across all ECFP bits containing that atom.
**Validation.** Top-10 SHAP substructures compared against known pharmacophores (e.g., hinge-binding motifs for kinase inhibitors). Sanity check: high-affinity compound should show more pharmacophoric features highlighted than low-affinity compound.
**New file:** `interpret/shap_analysis.py`

#### B3. Substructure Mask Explanation (SME) + Grad-AAM for GNN and Graphormer
**What it shows:** Which chemically meaningful substructures (functional groups, BRICS fragments, Murcko scaffolds) drive the GNN's prediction. SME is preferred over raw GNNExplainer because it produces explanations aligned with how medicinal chemists think — not arbitrary atom subgraphs. Grad-AAM (Gradient-weighted Affinity Activation Mapping, the GNN analogue of GradCAM) provides a complementary atom-level heatmap from gradient flow into the final graph conv layer.
**Source papers:**
- Wang et al., *Nature Communications* 2023: "Chemistry-intuitive explanation of GNNs for molecular property prediction with substructure masking" (SME)
- Yang et al., *Chemical Science* 2022: "MGraphDTA: deep multiscale graph neural network for explainable drug–target binding affinity prediction" (Grad-AAM)
**Implementation.**
- SME: Segment each molecule via RDKit BRICS fragmentation + functional-group SMARTS. Mask each fragment's node features to zero; importance = |ΔpKd|. Renders as fragment-coloured 2D depiction.
- Grad-AAM: Backpropagate gradients to final GNN convolutional layer activations; weight by gradient magnitude to produce per-atom importance scores.
- Also run PyG `GNNExplainer` as baseline comparison (required to benchmark against).
- Run on 100 test compounds stratified by pKd quintile.
**Validation.** For 20 well-characterised kinase inhibitors, confirm highest-importance fragment corresponds to hinge-binding pharmacophore. Quantitatively compare SME vs. GNNExplainer vs. Grad-AAM fidelity scores.
**New file:** `interpret/gnn_explain.py`

#### B4. Integrated Gradients via Captum (for sequence models and ESM-2 embeddings)
**What it shows:** Residue-level attribution for protein sequences — which amino acid positions drive the prediction. For ligand SMILES, which character tokens or BPE segments matter.
**Implementation.** `captum.attr.IntegratedGradients` applied to `LSTMBindingModel` and `MambaBindingModel`. Baseline = zero embedding. Attribute to input embeddings; aggregate per-token.
**Validation.** Top-5 attributed protein positions compared against UniProt annotated active site residues for a set of 20 well-studied targets.
**New file:** `interpret/captum_ig.py`

#### B5. Quantitative Fidelity Evaluation (required for Nature reviewers)
All four methods above will be evaluated for fidelity: mask the top-K attributed features (atoms, residues, ECFP bits) and measure the drop in prediction accuracy. Expected: ΔAUCfidelity > 0 (masking the most important features hurts more than masking random features).
**New file:** `interpret/fidelity_eval.py`

**Critical reference papers (must cite / must compare against):**
- **PSICHIC** (Hao et al.) — *Nature Machine Intelligence* 2024 vol.6 pp.673-687. PhySIcoCHemICal GNN; sequence-only, matches structure-based methods; decodes interaction fingerprints identifying binding-site residues AND ligand atoms simultaneously. Experimentally validated in adenosine A1R screen. **This is the current gold standard and closest competitor to our approach.**
- "Chemistry-intuitive explanation of GNNs with substructure masking" (Wang et al.) — *Nature Communications* 2023. Defines SME method.
- "MGraphDTA: deep multiscale GNN for explainable DTA" (Yang et al.) — *Chemical Science* 2022. Defines Grad-AAM method.
- "AI-Bind: improving generalizability of protein-ligand binding predictions" — *Nature Communications* 2023. Network sampling + unsupervised pretraining for binding site attribution.
- "CAPLA: improved prediction via cross-attention mechanism" — *Bioinformatics* 2023. Cross-attention interpretability, critical residues.
- "Drug discovery with explainable artificial intelligence" — *Nature Machine Intelligence* 2020. Foundational survey.
- "Explainable Artificial Intelligence in Drug Discovery" (Lavecchia) — *WIREs Computational Molecular Science* 2025. Current methodology review.
- "Interpretation of compound activity predictions with Shapley values" (Danel et al.) — *J. Med. Chem.* 2020. SHAP mapped to molecular graphs.
- "Geometric deep learning of protein-DNA binding specificity" — *Nature Methods* 2024. Template for attribution validated by mutagenesis.

---

### C. Documentation and Reproducibility Standards (Nature requirements)

All the following must be in place before submission:

#### C1. Experiment Diary (already implemented)
- `diary/results_diary.csv` — append-only, all experiments logged
- `diary/bootstrap_ci.csv` — 95% CIs for all test metrics (1000 bootstrap samples)
- `diary/FINDINGS.md` — auto-updated analysis narrative

#### C2. Reproducibility Checklist
- All random seeds fixed and logged per experiment
- All cached splits stored (`cache/splits/`) and never regenerated mid-run
- All model checkpoints saved to `cache/checkpoints/{experiment_id}.pt`
- Full `environment.yml` with pinned package versions (generate via `conda env export`)
- `run_all.py` is the single command to reproduce all experiments from scratch

#### C3. Interpretability Output Artefacts
- `interpret/figures/attention_maps/` — per-compound (residue × atom) heatmaps for BiCA/GLI
- `interpret/figures/shap/` — atom-level SHAP overlays on 2D molecular depictions
- `interpret/figures/gnn_explain/` — GNNExplainer subgraph visualizations
- `interpret/figures/captum_ig/` — per-residue attribution bar plots for sequence models
- `interpret/fidelity_results.csv` — quantitative fidelity scores for all methods

#### C4. Statistical Rigour
- Bootstrap 95% CIs on all test metrics (already implemented via `bootstrap_ci.py`)
- Multi-seed (42/123/456) mean ± std for all key models (already in `run_all.py`)
- Cross-dataset replication (BindingDB → LeakyPDB) for top-5 models
- Pairwise Wilcoxon signed-rank tests between top model families, Bonferroni-corrected

#### C5. Paper-Specific Sections Already Covered
| Paper section | Source in this repo |
|---|---|
| Dataset & split | `harness/data.py`, `harness/config.py` |
| Baselines | `run_experiment.py` baselines/trees/mlp groups |
| Architecture comparisons | All model families in `models/` |
| Ablation study | Multi-seed + 2×2 repr ablation (xgb 2×2) |
| Cross-dataset validation | LeakyPDB phase in `run_all.py` |
| Interpretability | `interpret/` (to be created) |
| Statistical significance | `bootstrap_ci.py`, multi-seed std |

---

### D. Implementation Sequence

```
Phase 1 — Training objectives (low-risk, high-reward):
  harness/losses.py          PairwiseRankingLoss (from MBP)
  harness/trainer.py         aux_loss hook (optional kwarg, backward-compatible)
  models/gnn.py              GraphMAE node masking + NodeDecoder recon head
  harness/gnn_trainer.py     dual-forward training step

Phase 2 — New architectures:
  models/graphormer.py       Graphormer (SPD/degree/edge biases in attention)
  models/gli.py              GLI gated fusion (GNN local + BiCA global)
  models/dsm.py              DualBind DSM head (encode/head split in bica.py + mlp.py)
  models/sequence_models.py  MambaEncoder + MambaBindingModel (append)
  run_experiment.py          Register all new experiments + psichic_official

Phase 3 — Interpretability:
  interpret/shap_analysis.py   SHAP (TreeExplainer for XGB/RF, DeepExplainer for MLP)
                                Maps ECFP bits → atoms via RDKit GetBitInfo()
  interpret/attention_maps.py  Cross-attention (residue × atom) heatmaps for BiCA/GLI
                                Validate against PDB binding sites via LeakyPDB headers
  interpret/gnn_explain.py     SME (BRICS fragment masking) + Grad-AAM + GNNExplainer baseline
                                Benchmark all three methods' fidelity scores
  interpret/captum_ig.py       Integrated Gradients (Captum) for LSTM/Mamba + ESM-2
                                Validate against UniProt annotated active-site residues
  interpret/fidelity_eval.py   Mask top-K features → measure ΔpKd for all methods
                                Output: interpret/fidelity_results.csv

Phase 4 — Reproducibility packaging:
  environment.yml              conda env export --no-builds > environment.yml
  REPRODUCE.md                 Step-by-step guide: data → run_all.py → interpret/ → figures
  interpret/figures/           All visualizations: attention heatmaps, SHAP overlays,
                               fragment importance, residue attribution bar plots
  diary/                       Final bootstrap_ci.csv, multi-seed mean±std table
  supplementary/               Full leaderboard table, fidelity results, all hyperparams
```

### E. What "Interpretability" Means at Each Granularity

For reviewers of *Nature Machine Intelligence* or *Nature Communications*, interpretability must be **mechanistically plausible, quantitatively evaluated, and ideally experimentally validated**. Pure attention visualization without ground-truth validation is increasingly criticized (2025 WIREs review).

| Granularity | Method in this paper | Ground-truth validation |
|---|---|---|
| Residue-level | Cross-attention maps (B1), Integrated Gradients (B4) | PDB binding-site residues via LeakyPDB `header` → PDB API |
| Atom/substructure-level | SHAP (B2), SME + Grad-AAM (B3) | Known pharmacophores for kinase inhibitors; matched molecular pairs |
| Cross-modal (residue × atom) | Cross-attention maps (B1), GLI gate weights | PSICHIC's published interaction fingerprints as reference |
| Quantitative fidelity | Fidelity evaluation (B5) | Mask top-K → ΔpKd must exceed random-mask baseline |

**The "Clever Hans" risk.** If the model learns dataset shortcuts (e.g., kinase-family protein similarity leaks across scaffold splits), attributions will highlight protein family features, not true binding determinants. This will be flagged by running cross-family ablations: separate interpretability analysis for kinase vs. non-kinase test compounds.
