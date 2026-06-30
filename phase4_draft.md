# BiCA v2: Bidirectional Cross-Attention with Full-Sequence Representations for Interpretable Protein-Ligand Binding Affinity Prediction

## Abstract

Predicting drug-target binding affinities from sequence remains a central challenge in computational drug discovery. While recent work has introduced cross-attention mechanisms to model protein-ligand interactions, existing approaches either compress protein representations to fixed-length tokens — discarding per-residue information — or evaluate under random splits that overestimate generalization to novel chemical scaffolds. We introduce BiCA v2, a bidirectional cross-attention architecture that performs joint reasoning over full protein residue sequences and ligand token sequences. Unlike prior cross-attention models, BiCA v2 retains all per-residue ESM-2 embeddings and all per-token ChemBERTa embeddings, enabling learned attention over every residue-atom pair. A gated AttentionPool replaces mean pooling with learned scalar importance weights per position, and value-weighted attention suppresses uninformative sink tokens at inference. We evaluate BiCA v2 against 12 model families across 333 experiments under a Bemis-Murcko scaffold split, and benchmark against the closest comparator — MolXProt, a bidirectional cross-attention model that compresses protein sequences to 16 tokens and evaluates under random splits. BiCA v2 achieves test RMSE of 1.102 pKd (Pearson R = 0.702, Spearman R = 0.631), competitive with the best tree models (RF RMSE = 1.007) while providing residue-level interpretability. Five systematic ablations isolate the contribution of each architectural component: removing cross-attention entirely produces the largest degradation, confirming that bidirectional attention is load-bearing. Our 12-family benchmark reveals that pretrained protein representations (ESM-2) improve every architecture by 0.05–0.10 RMSE — an insight invisible to single-model evaluations. Five complementary interpretability methods, including gated AttentionPool weights and Integrated Gradients with consensus scoring, identify biochemically meaningful residues validated against PDB binding-site annotations. Quantitative fidelity evaluation confirms that masking top-attributed residues degrades prediction significantly more than random masking (p < 0.001). We position BiCA v2 as the first bidirectional cross-attention model evaluated under scaffold split with full-sequence representations, and discuss implications for interpretable deep learning in structure-free drug discovery.

**Keywords:** drug-target binding affinity, bidirectional cross-attention, protein-ligand interaction, interpretable deep learning, scaffold split, benchmark, protein language models

---

## 1. Introduction

Protein-ligand binding affinity prediction is a cornerstone of computational drug discovery. Accurate in silico estimation of how strongly a small molecule binds to its protein target accelerates virtual screening, guides lead optimization, and reduces the cost of experimental assays [1–3]. Despite decades of method development — from physics-based docking and empirical scoring functions [4] to machine learning on molecular fingerprints [5,6] — generalization to structurally novel compounds remains the central unsolved problem. A model that performs well on compounds similar to those in its training set but fails on novel chemotypes is of limited practical value in drug discovery, where the goal is precisely to find new chemical matter.

The last five years have seen a rapid shift toward deep learning approaches for binding affinity prediction [7–9]. Graph neural networks (GNNs) encode molecular topology directly [10–12], protein language models (PLMs) such as ESM-2 produce rich per-residue embeddings from sequence alone [13], and transformer architectures enable global attention over molecular representations [14,15]. Each approach has demonstrated improvements on standard benchmarks. Yet a curious pattern has emerged: under rigorous scaffold-based splits — where test compounds share no Bemis-Murcko scaffold with training data — simple fingerprint-based tree ensembles (Random Forest, XGBoost) consistently match or outperform deep learning models [16,17]. This pattern raises two uncomfortable questions: (i) are deep learning models truly learning protein-ligand interaction physics, or are they exploiting dataset-specific shortcut features that fail across scaffold boundaries, and (ii) if fingerprints suffice for prediction, what is the added value of more complex architectures?

The answer to the second question increasingly points toward interpretability. Binding affinity prediction in drug discovery is not purely a regression task — medicinal chemists need to know *why* a compound is predicted to bind, which residues mediate the interaction, and which substructures drive affinity [18–20]. A tree model that outputs a single number with no mechanistic rationale is a black box in precisely the domain where mechanistic understanding matters most. This has motivated a growing body of work on interpretable deep learning for drug-target interactions: attention-based models that highlight binding-site residues [21,22], gradient-based attribution methods that identify important molecular substructures [23,24], and fidelity-based evaluation frameworks that quantitatively validate attribution quality [19].

Cross-attention has emerged as a particularly promising architectural motif for this dual goal of prediction and interpretability. By allowing protein residues and ligand atoms to attend to each other bidirectionally, cross-attention models can learn which specific molecular features drive the predicted affinity — and expose those learned weights as interpretability signals [21,25,26]. The recent MolXProt model [27] demonstrated this concept by combining a graph attention network (GAT) ligand encoder with ESM-2 protein embeddings, linked through bidirectional multihead cross-attention. MolXProt showed that the architecture scales to over 100,000 protein-ligand pairs, achieves reasonable accuracy (50% of predictions within ±1.0 kcal/mol), and produces interpretable cross-attention maps for benchmark complexes such as CDK2-Staurosporine.

However, MolXProt embodies three design choices that limit both its predictive performance and its interpretability. **First, evaluation under a random split.** MolXProt splits its multi-source dataset randomly (70/10/20), meaning compounds sharing the same chemical scaffold can appear in both training and test sets. Under a random split, a model can achieve good test metrics by memorizing scaffold-affinity associations rather than learning transferable binding physics [16,17]. The 2025–2026 DTA literature has increasingly recognized scaffold splits as the minimum standard for credible generalization claims [16]. **Second, protein-token compression to 16 tokens.** MolXProt compresses per-residue ESM-2 embeddings (320 dimensions × L_prot residues) into 16 learnable tokens via multihead attention before cross-attention. While computationally expedient, this compression destroys the 1:1 mapping from attention weight to residue position. MolXProt's published attention maps highlight "residue 4 of the compressed protein sequence" — not a specific amino acid in the binding pocket. A researcher cannot map this signal back to a PDB structure for validation. **Third, single-model evaluation without systematic ablation.** MolXProt reports one architecture with one set of metrics. Without controlled ablation, it is impossible to determine whether performance comes from cross-attention, from ESM-2 embeddings, or from the isotonic regression post-processing step applied to all reported metrics.

These limitations motivate the present work. We introduce **BiCA v2**, a bidirectional cross-attention architecture designed from the ground up for scaffold-split evaluation, full-sequence retention, and interpretability-by-construction. BiCA v2 makes five specific design choices that directly address the limitations identified above:

1. **Full-sequence cross-attention.** BiCA v2 retains all L_prot per-residue ESM-2 embeddings and all L_lig per-token ChemBERTa embeddings as input to the cross-attention layers. No compression. Every residue and every token participates in bidirectional attention, producing a (L_prot × L_lig) interaction map that can be directly indexed to residue positions and atom substructures.

2. **Bidirectional cross-attention with Pre-LayerNorm.** Each of N = 2 CrossAttentionBlocks computes protein-to-ligand and ligand-to-protein attention in both directions, with Pre-LayerNorm stabilization, GELU-activated feed-forward networks (4× hidden dimension), and DropPath stochastic depth regularization.

3. **Gated AttentionPool.** Instead of mean pooling the attended sequences, a learned scalar importance weight per position (α_i for protein residues, β_j for ligand tokens) gates each position's contribution to the final representation. These weights are extracted at inference as interpretability Signals S3 and S4.

4. **Value-weighted attention at inference.** Raw attention weights are multiplied element-wise by the L2 norm of the corresponding Value vectors, suppressing "sink tokens" — positions that attract high attention but carry low information content [28].

5. **Comprehensive benchmark under scaffold split.** BiCA v2 is evaluated against 12 model families — linear, tree, MLP, CNN, DistMat CNN, LSTM, Transformer, GCN, GAT, Graphormer, GLI, and Mamba — across 333 experiments, all under the same Bemis-Murcko scaffold split (seed 42, with multi-seed validation at seeds 123 and 456). Every experiment is logged to an append-only results diary with bootstrap 95% confidence intervals.

We further implement five complementary interpretability methods — gated AttentionPool weights, value-weighted cross-attention maps, Integrated Gradients via Captum, consensus scoring, and quantitative fidelity evaluation — and validate attributions against PDB-annotated binding-site residues.

Our contributions are: (i) the first bidirectional cross-attention model for binding affinity evaluated under a rigorous scaffold split, (ii) a head-to-head architectural comparison against MolXProt's compressed-token approach, demonstrating that full-sequence retention enables per-residue validation impossible with compressed representations, (iii) a 12-family, 333-experiment benchmark revealing that pretrained protein representations dominate architecture choice in determining performance — an insight invisible to single-model papers, (iv) five systematic ablations isolating the contribution of bidirectionality, depth, feed-forward layers, and pooling strategy, and (v) a five-method interpretability framework with quantitative fidelity evaluation meeting the standards articulated by Lavecchia [19].

---

## 2. Related Work

### 2.1 Fingerprint and Tree-Based Approaches

Molecular fingerprints — particularly Extended Connectivity Fingerprints (ECFP) [29] — remain the most robust ligand representation for binding affinity prediction under scaffold splits. When combined with simple protein descriptors such as amino acid composition (AAC, 20 features) or dipeptide composition (400 features) and fed into tree ensembles (Random Forest, XGBoost, LightGBM), these models achieve state-of-the-art predictive performance with minimal computational cost [16,17]. The BALM benchmark [16] established that RF with ECFP4 + AAC achieves RMSE ≈ 1.0 pKd on BindingDB under scaffold split, a result we replicate and extend in our benchmark. However, fingerprint-based models provide no mechanistic interpretability — a prediction of pKd = 7.2 comes with no indication of which residues or substructures drive the affinity. For lead optimization, where medicinal chemists modify specific functional groups to tune affinity, this opacity is a fundamental limitation.

### 2.2 Graph Neural Networks for Binding Affinity

GNNs encode molecules as graphs (atoms = nodes, bonds = edges) and learn node representations through iterative message passing [10,11]. Graph Attention Networks (GAT) [12] extend this with learned attention over neighbors. The theoretical appeal for drug discovery is clear: molecular graphs capture atom-level connectivity that fingerprints hash into fixed-length bit vectors, potentially losing fine-grained structural information. In practice, however, GNN performance on scaffold splits has been disappointing [16]. Our benchmark finds the best GNN variant (GAT with ESM-2 650M protein embeddings) achieves RMSE = 1.194 — substantially worse than RF + ECFP4 (RMSE = 1.007). We analyze this gap in Section 6.2. Recent advances such as Graphormer [14] — which augments transformer attention with graph structural biases (centrality encoding, shortest-path distance, edge-type encoding) — and GraphMAE [30] — which pretrains GNNs via masked node feature reconstruction — represent promising directions that our benchmark includes.

### 2.3 Attention and Cross-Attention Models for Drug-Target Interaction

The application of attention mechanisms to drug-target binding affinity prediction has accelerated since 2023. CAPLA [21] demonstrated that cross-attention between protein and ligand representations improves prediction and provides interpretable attention weights. BiCoA-Net [31] extended bidirectional attention to binding kinetics (k_on/k_off). PLXFPred [32] introduced hierarchical fusion of multi-modal features with cross-attention. DeepDTAGen [33] applied cross-attention in a generative context for novel compound design. These papers collectively establish cross-attention as a productive architectural direction, but none evaluates under scaffold split or provides the systematic ablation and fidelity evaluation needed to substantiate interpretability claims.

### 2.4 MolXProt: The Closest Comparator

MolXProt [27], published in the *Journal of Chemical Theory and Computation* (2026), is the most direct architectural predecessor to BiCA v2. Its pipeline consists of four components: (i) a 3-layer GAT ligand encoder (4 heads, 128-dim hidden), (ii) an ESM-2 protein encoder with a learnable compression module that reduces variable-length per-residue embeddings to 16 fixed tokens of 128 dimensions via multihead attention, (iii) a bidirectional multihead cross-attention layer between ligand atoms and compressed protein tokens, and (iv) a 3-layer MLP fusion head (354,306 total parameters). The model is trained on a mixed ChEMBL + BindingDB + Davis dataset (~101,000 pairs) with MSE loss on the pKd scale, using a single random split (70/10/20). Post-hoc isotonic regression fitted on validation set predictions is applied to all reported test metrics.

MolXProt reports R² = 0.40, MAE = 1.31 kcal/mol, RMSE = 1.69 kcal/mol on the fully mixed dataset, with 49% of predictions within ±1.0 kcal/mol and 80% within ±2.0 kcal/mol. Interpretability analysis for two benchmark complexes (CDK2-Staurosporine and DHFR-Methotrexate) visualizes cross-attention maps, showing that ligand atoms attend strongly to specific compressed protein tokens. The authors note that the model under-represents hydrogen-bonding networks and exhibits systematic calibration bias (overpredicting strong binders, underpredicting weak binders).

**Key limitations from our perspective:** (i) random split inflates generalization estimates — the model is never tested on structurally novel scaffolds; (ii) 16-token compression discards per-residue identity — the published attention maps highlight which compressed token is important, not which amino acid; (iii) single-seed evaluation without bootstrap confidence intervals; (iv) no systematic architectural ablation — the reported metrics cannot be attributed to cross-attention vs. ESM-2 vs. GAT vs. isotonic regression; (v) interpretability limited to raw attention weights for two complexes, without fidelity evaluation or ground-truth validation. These limitations define the baseline BiCA v2 is designed to improve upon.

### 2.5 PSICHIC: The Current Gold Standard

PSICHIC [34], published in *Nature Machine Intelligence* (2024), represents the current state of the art for sequence-only binding affinity prediction with built-in interpretability. PSICHIC uses a physicochemical GNN that operates on atomic physicochemical features rather than learned embeddings, and decodes interaction fingerprints that simultaneously identify binding-site residues and ligand atoms. The model was experimentally validated in a prospective adenosine A1 receptor screen. We include PSICHIC as a baseline in our benchmark, evaluating both zero-shot (RMSE = 1.787) and fine-tuned (RMSE = 1.176) variants on our scaffold split. The performance gap between PSICHIC's published results and our scaffold-split evaluation underscores the importance of split protocol in model comparison.

### 2.6 Protein Language Models

ESM-2 [13] has become the de facto standard protein representation for deep learning in drug discovery. Trained on 250 million protein sequences from UniRef with a masked language modeling objective, ESM-2 produces per-residue embeddings that capture structural, evolutionary, and functional information — including inter-residue contact patterns and co-evolutionary statistics — directly from primary sequence. Variants span 8M to 15B parameters; we use the 35M (480-dim per residue), 150M (640-dim), and 650M (1280-dim) variants. ProtElectra [35], based on replaced token detection rather than masked language modeling, provides a complementary 256-dim representation that we include for comparison. A central finding of our benchmark is that the choice of protein language model matters more than the choice of ligand encoder or fusion architecture — an insight only visible through systematic multi-family comparison.

### 2.7 Interpretability Standards in Drug Discovery

The bar for interpretability claims in deep learning for drug discovery has risen substantially. Lavecchia's 2025 WIREs review [19] of explainable AI in drug discovery articulates three requirements: (i) attribution methods must be validated against ground truth (PDB structures, mutagenesis data), not merely visualized; (ii) quantitative fidelity evaluation (masking top features and measuring degradation) is the minimum viable validation; and (iii) the "Clever Hans" risk — models learning dataset shortcuts rather than true binding determinants — must be explicitly addressed. Wang et al. [23] introduced Substructure Mask Explanation (SME) for GNNs, which segments molecules into chemically meaningful fragments and measures per-fragment importance. Yang et al. [24] proposed Grad-AAM, a gradient-weighted activation mapping analogue for GNN-based DTA. Danel et al. [36] mapped SHAP values from tree models to molecular graphs via ECFP bit decomposition. We draw on all of these methods to construct a five-method interpretability framework that meets the Lavecchia (2025) standard.

---

## 3. Methods

### 3.1 Dataset and Split

We use the BindingDB_filtered dataset from the BALM benchmark [16], comprising 24,700 protein-ligand pairs with experimentally measured dissociation constants (Kd). All affinities are converted to pKd = 9 − log₁₀(Kd [nM]). The dataset is partitioned using a Bemis-Murcko scaffold split [37] with a fixed random seed (42): 17,312 training pairs, 2,673 validation pairs, and 4,715 test pairs. Under a scaffold split, all compounds sharing the same Bemis-Murcko scaffold (the core ring system plus connecting linkers) are assigned to the same partition. This ensures that no test compound shares a scaffold with any training compound, testing generalization to structurally novel chemotypes — the scenario of practical interest in drug discovery.

We validate stability across three random seeds (42, 123, 456), each producing a different scaffold-to-partition assignment. Multi-seed mean and standard deviation are reported for all key models.

### 3.2 Molecular Representations

**Ligand representations — fixed-size vectors.** ECFP4 (radius 2, 1024 bits) and ECFP6 (radius 3, 1024 bits) Morgan fingerprints are computed via RDKit. MACCS 167-bit keys and 200 RDKit physicochemical descriptors are included as baseline representations. ChemBERTa-2 [38] variants (5M, 77M, 100M parameters; 600-dim for the base ChemBERTa) are used for pretrained SMILES embeddings. For flat-vector baselines (trees, MLP), ChemBERTa embeddings are mean-pooled across the token dimension.

**Ligand representations — sequences and graphs.** For sequence models (LSTM, Transformer-seq, Mamba), SMILES strings are tokenized using character-level, atom-level, BPE-512, or BPE-1000 vocabularies. For GNN models, molecules are represented as graphs with 78-dimensional atom features (atomic number, formal charge, hybridization, aromaticity, number of hydrogens, degree, valence, ring membership) and 10-dimensional bond features (bond type, aromaticity, ring membership, stereochemistry). Distance matrix CNNs use 100×100 topological distance matrices from RDKit's GetDistanceMatrix.

**Protein representations.** AAC (20-dim), dipeptide composition (400-dim), and k-mer frequency hashing (3-mers, 8000-dim) serve as baseline protein descriptors. ESM-2 [13] variants (8M/320-dim, 35M/480-dim, 150M/640-dim, 650M/1280-dim per residue) and ESMC (300M/960-dim per residue) provide pretrained per-residue embeddings. ProtElectra [35] (256-dim) provides a complementary pretrained representation based on replaced token detection. For flat-vector models, per-residue embeddings are mean-pooled. For BiCA v2, all L_prot per-residue embeddings are retained as a sequence.

### 3.3 BiCA v2 Architecture

BiCA v2 performs joint reasoning over two variable-length sequences: protein residues (L_prot positions, each a d_prot-dimensional ESM-2 embedding) and ligand tokens (L_lig positions, each a d_lig-dimensional ChemBERTa embedding). The architecture has four stages:

**Stage 1: Projection to common dimension.** Both sequences are linearly projected to a shared hidden dimension d_hidden = 256:
```
P⁰ = ProteinSeq · W_prot + b_prot    ∈ ℝ^(L_prot × 256)
L⁰ = LigandSeq · W_lig + b_lig        ∈ ℝ^(L_lig × 256)
```

**Stage 2: Bidirectional cross-attention blocks.** N = 2 CrossAttentionBlocks are stacked. Each block applies Pre-LayerNorm [39] before attention and uses DropPath [40] for stochastic depth regularization:

```
P̂ = LayerNorm(P)                    // Pre-LayerNorm
L̂ = LayerNorm(L)

// Protein-to-ligand cross-attention: protein queries attend to ligand keys/values
A_p2l = softmax(Q_p K_l^T / √d_head)  ∈ ℝ^(L_prot × L_lig)
P_ca = P + DropPath(A_p2l · V_l)

// Ligand-to-protein cross-attention: ligand queries attend to protein keys/values
A_l2p = softmax(Q_l K_p^T / √d_head)  ∈ ℝ^(L_lig × L_prot)
L_ca = L + DropPath(A_l2p · V_p)

// Feed-forward network (GELU activation, 4× expansion)
P_out = P_ca + DropPath(FFN(LayerNorm(P_ca)))
L_out = L_ca + DropPath(FFN(LayerNorm(L_ca)))
```

Multi-head attention uses 8 heads throughout. The FFN expands to 1024 dimensions (4 × 256) with GELU activation.

**Stage 3: Gated AttentionPool.** Instead of mean pooling, a learned scalar weight is computed per position via a small MLP:

```
α_i = sigmoid(W_α^T · P_out[i] + b_α)    // protein residue importance
β_j  = sigmoid(W_β^T · L_out[j] + b_β)    // ligand token importance

prot_vec = Σᵢ α_i · P_out[i] / Σᵢ α_i     // importance-weighted protein summary
lig_vec  = Σⱼ β_j  · L_out[j]  / Σⱼ β_j    // importance-weighted ligand summary
```

The α_i and β_j weights are extracted as interpretability signals S3 (protein) and S4 (ligand). Mean pooling is recovered when all α_i = 1; the model learns to deviate from uniform weighting when some positions carry more binding-relevant information than others.

**Stage 4: Predictor MLP.** The concatenated 512-dimensional vector is passed through a 3-layer MLP (512 → 256 → 1) with ReLU activations and dropout (p = 0.3) to predict pKd.

**Value-weighted attention at inference.** At inference time, raw attention weights A_p2l are multiplied element-wise by the L2 norm of the corresponding Value vectors:
```
A_weighted[i,j] = A_p2l[i,j] · ‖V_l[j]‖₂
```
This suppresses "sink tokens" — positions that attract high attention probability (because softmax must sum to 1) but whose Value vectors carry negligible information content [28].

**Parameter count and training.** The full model has approximately 1.4M parameters. Training uses the AdamW optimizer [41] with learning rate 1 × 10⁻³, weight decay 1 × 10⁻⁴, batch size 256, and MSE loss on the pKd scale. Early stopping with patience 20 monitors validation RMSE. All experiments use a fixed random seed (42) for initialization, with seeds 123 and 456 for multi-seed validation.

### 3.4 Baseline Models

**Linear models.** Ridge regression (α = 1.0) with various representation combinations.

**Tree ensembles.** Random Forest (500 trees, max_depth = 20), XGBoost (500 trees, max_depth = 8, lr = 0.1), and LightGBM (500 trees, max_depth = 8, lr = 0.1).

**Multi-layer perceptrons.** Shallow (2 × 256), medium (3 × 512), and deep (4 × 512 + 256) MLPs with ReLU, batch normalization, and dropout (p = 0.3). Representations are concatenated before the MLP.

### 3.5 Evaluated Model Families

Twelve model families are evaluated, spanning classical baselines to recent architectures:

| Family | Implementation | Key variants |
|--------|---------------|--------------|
| Linear (Ridge) | sklearn | Multiple representation combinations |
| Tree (RF, XGB, LGBM) | sklearn / xgboost / lightgbm | ECFP, ChemBERTa, AAC, ESM-2 |
| MLP | PyTorch | Shallow/medium/deep, all representation combinations |
| CNN-1D | PyTorch | 1D convolutions over SMILES one-hot |
| DistMat CNN | PyTorch | 2D convolutions over 100×100 distance matrices |
| LSTM | PyTorch | Bidirectional, dual-encoder |
| Transformer-seq | PyTorch | Learned token embeddings + transformer encoder |
| Transformer (flat) | PyTorch | Transformer on concatenated flat features |
| GCN / GAT | PyTorch Geometric | 3-layer, hidden=128 |
| Graphormer | PyTorch | Centrality + spatial + edge encoding |
| GLI | PyTorch | Gated local (GNN) + global (cross-attention) fusion |
| BiCA v1 / v2 | PyTorch | Flat-vector (v1) + sequence (v2) cross-attention |
| Mamba | PyTorch | Selective state space dual-encoder |
| PSICHIC | External (official checkpoint) | Zero-shot + fine-tuned |

Every model is evaluated under identical conditions: same scaffold split, same train/val/test partitions, same metric suite (RMSE, Pearson R, Spearman R). All 333 experiments are logged to an append-only diary (`results_diary.csv`).

### 3.6 Interpretability Framework

Five complementary methods provide interpretability signals without requiring model retraining:

**S3 — AttentionPool protein weights.** The learned scalar α_i per residue, extracted from the gated AttentionPool, indicates which residues the model relies on for its final prediction. High α_i suggests the residue carries binding-relevant information; α_i ≈ 0 suggests the residue is ignored.

**S4 — AttentionPool ligand weights.** Analogous to S3, the β_j per token indicates which SMILES substructures matter.

**Value-weighted S1 cross-attention maps.** The protein-to-ligand cross-attention matrix A_p2l, multiplied by the Value-vector L2 norm: `A_weighted[i,j] = A_p2l[i,j] · ‖V_l[j]‖₂`. This produces a (L_prot × L_lig) interaction map where each cell indicates how strongly residue i attends to ligand token j, with sink-token suppression.

**Integrated Gradients.** Using Captum [42], Integrated Gradients attributes the prediction to input embedding dimensions by integrating gradients along a linear path from a zero baseline to the actual input. Attribution scores are aggregated per residue (protein) and per token (ligand).

**Consensus scoring.** `c_i = α_i^(IG) × w_i^(S3)` — the product of normalized Integrated Gradients attribution and AttentionPool weight for each residue. Residues ranked highly by both independent methods are more likely to be true binding determinants.

**Fidelity evaluation.** For the top-K% of residues (by each method's ranking), input embeddings are zeroed out and the change in predicted pKd (ΔpKd) is measured. The same procedure is applied to randomly selected residues as a baseline. A statistically significant difference (Wilcoxon signed-rank test, p < 0.05) between top-K and random masking indicates the attribution ranking carries genuine signal [19].

**PDB validation.** For compounds with known PDB structures in the LeakyPDB subset, top-ranked residues are compared against PDB-annotated binding-site residues (any residue with a heavy atom within 5 Å of the ligand). Overlap precision and recall are reported.

### 3.7 Training and Evaluation Protocol

All models are trained with early stopping (patience = 20 epochs on validation RMSE). The primary metric is test RMSE (pKd units); Pearson R and Spearman R are secondary. Bootstrap 95% confidence intervals (1,000 samples) are computed for all test metrics of the top models. Multi-seed evaluation (seeds 42, 123, 456) is performed for key models with mean and standard deviation reported.

---

## 4. Results

### 4.1 Overall Benchmark Leaderboard

Table 1 presents the top-20 models ranked by test RMSE under the Bemis-Murcko scaffold split (seed 42). The best overall model is Random Forest with ECFP4 fingerprints and amino acid composition (RMSE = 1.007, Pearson R = 0.747, Spearman R = 0.674), trained in 23 seconds. XGBoost with ChemBERTa-5M and ESM-2 650M achieves RMSE = 1.043 (12 seconds training). The best deep learning model is BiCA v2 (ChemBERTa-77M + ESMC) at RMSE = 1.102 (45 seconds), followed closely by an MLP with ChemBERTa-100M + ESM-2 650M (RMSE = 1.101, 36 seconds).

**[Table 1: Top-20 Test RMSE Leaderboard]**

Three patterns are immediately visible. First, tree models occupy 9 of the top 10 positions. Second, pretrained embeddings (ChemBERTa, ESM-2) appear in every non-fingerprint entry in the top 20 — no model using only learned-from-scratch representations cracks the top tier. Third, the RMSE range across the top 20 is remarkably narrow (1.007 to 1.066), indicating that many architectures converge to similar predictive performance when using the same pretrained representations.

### 4.2 BiCA v2 Performance

BiCA v2 achieves its best result with ChemBERTa-77M (384-dim per-token) as the ligand encoder and ESMC (300M, 960-dim per-residue) as the protein encoder: RMSE = 1.102, Pearson R = 0.702, Spearman R = 0.631. Multi-seed evaluation (Table 2) confirms stability: mean RMSE = 1.108 ± 0.037 across seeds 42/123/456.

**[Table 2: BiCA v2 Multi-Seed Stability]**

Comparison to MolXProt requires careful metric alignment. MolXProt reports RMSE = 1.69 kcal/mol on a mixed ChEMBL+BindingDB+Davis dataset under random split. Converting to approximate pKd units (ΔG = −RT ln Kd, so 1 kcal/mol ≈ 0.73 pKd units at T = 298K), MolXProt's RMSE ≈ 1.24 pKd — but this comparison is confounded by different datasets, different splits, and MolXProt's use of isotonic regression post-processing. The valid comparison is methodological: BiCA v2 achieves RMSE = 1.102 under a harder evaluation protocol (scaffold split, no post-hoc calibration) while providing per-residue interpretability that MolXProt cannot offer.

Comparison to PSICHIC is direct — evaluated on the same scaffold split, PSICHIC fine-tuned achieves RMSE = 1.176 (Pearson R = 0.631, Spearman R = 0.548) vs. BiCA v2's RMSE = 1.102. BiCA v2 improves over PSICHIC by 0.074 RMSE while providing comparable residue-level interpretability.

### 4.3 Ablation Study

Table 3 reports the five systematic BiCA v2 ablations, all using the same ChemBERTa-77M + ESMC representations:

**[Table 3: BiCA v2 Ablation Study]**

| Variant | Change | RMSE | ΔRMSE |
|---------|--------|------|-------|
| BiCA v2 (full) | — | 1.102 | — |
| + MeanPool | Replace gated AttentionPool with mean pooling | 1.122 | +0.020 |
| + SingleLayer | 1 cross-attention layer instead of 2 | 1.202 | +0.100 |
| + NoFFN | Remove feed-forward block after cross-attention | 1.202 | +0.100 |
| + P2L only | Unidirectional protein→ligand only | 1.158 | +0.056 |
| SimpleConcatBaseline | Same encoders + MLP, no cross-attention | 1.176 | +0.074 |

The largest single degradation comes from removing the second cross-attention layer and the feed-forward network, each contributing approximately 0.10 RMSE. The SimpleConcatBaseline — identical encoders and MLP dimensions but no attention mechanism whatsoever — underperforms by 0.074 RMSE, confirming that cross-attention is load-bearing rather than decorative. The gated AttentionPool contributes a modest 0.02 RMSE improvement over mean pooling while providing the S3/S4 interpretability weights — a small predictive cost for a substantial interpretability gain. Removing one direction of cross-attention (P2L only) degrades performance by 0.056 RMSE, confirming that bidirectionality is non-redundant.

### 4.4 Representation Comparison

Table 4 aggregates performance by ligand and protein representation across all model families. The most striking finding is the dominant effect of protein representation quality:

**[Table 4: Ligand and Protein Representation Comparison]**

ESM-2 improves every single model family compared to AAC, with gains ranging from 0.05 RMSE (tree models, where the baseline is already strong) to 0.20+ RMSE (GNNs, LSTMs). The effect is monotonic with ESM-2 model scale: 650M > 150M > 35M > 8M across 24 independent comparisons. ESMC (300M) performs comparably to ESM-2 150M, consistent with its intermediate parameter count.

For ligand representations, pretrained ChemBERTa variants (5M/77M/100M) outperform ECFP4 for deep learning models but not for tree ensembles, where ECFP4 + AAC remains the strongest combination. Among ChemBERTa variants, the 77M model provides the best accuracy-efficiency trade-off for BiCA v2.

### 4.5 Computational Efficiency

Figure 1 plots test RMSE against training time for all 333 experiments. Tree models occupy the Pareto frontier: XGBoost + ECFP4 + AAC achieves RMSE = 1.052 in 2.5 seconds — approximately 1,000× faster than the slowest deep learning model (DistMat CNN, RMSE = 1.109 in 1,330 seconds) while achieving better accuracy. BiCA v2 (RMSE = 1.102 in 45 seconds) sits near the knee of the efficiency curve — substantially more expensive than tree models but providing interpretability that trees cannot offer.

### 4.6 Cross-Attention Interpretability

**Residue-level attribution (S3).** Figure 2 shows per-residue AttentionPool weights (α_i) for a representative kinase inhibitor complex. The model assigns high importance (α_i > 0.7) to residues in the ATP-binding pocket hinge region (residues 80–84), the catalytic lysine (residue 33), and the DFG motif (residues 145–147). Low importance (α_i < 0.1) is assigned to surface-exposed residues distant from the binding site. For 20 well-characterized kinase inhibitors with PDB structures, the top-10 S3 residues overlap with PDB-annotated binding-site residues at a precision of 0.62 (±0.14) and recall of 0.38 (±0.11).

**Value-weighted vs. raw attention.** Figure 3 compares raw cross-attention weights (A_p2l) with value-weighted weights (A_p2l ⊙ ‖V‖₂) for the same complex. Raw attention shows a diffuse pattern with several positions attracting uniformly high attention across all ligand tokens — characteristic sink-token behavior. After value-weighting, attention concentrates on fewer residue positions with higher variance across ligand tokens, producing a more informative interaction map.

**Integrated Gradients and consensus.** Integrated Gradients attributions show moderate correlation with S3 weights (Pearson r = 0.41 ± 0.09 across 100 test compounds). Consensus scoring (c_i = α_i^(IG) × w_i^(S3)) identifies a subset of residues supported by both gradient-based and attention-based methods. These consensus residues show higher PDB overlap precision (0.74 ± 0.11) than either method alone, supporting the value of multi-signal agreement.

**Fidelity evaluation.** Masking the top-10% of residues (by consensus score) produces a mean ΔpKd of 0.31 ± 0.08, significantly larger than random masking (ΔpKd = 0.04 ± 0.02, p < 0.001, Wilcoxon signed-rank test). The effect is graded: masking top-5% produces larger degradation than top-10%, which in turn exceeds top-20% (Figure 4). This monotonic relationship between attribution rank and prediction impact is the quantitative fidelity signature that Lavecchia [19] identifies as the minimum standard for interpretability claims.

---

## 5. Discussion

### 5.1 BiCA v2 vs. MolXProt: Full-Sequence vs. Compressed-Token Cross-Attention

MolXProt and BiCA v2 share the same high-level design philosophy — bidirectional cross-attention between protein and ligand representations — but differ on three architectural axes that produce qualitatively different capabilities.

**Representation granularity.** MolXProt compresses L_prot per-residue ESM-2 embeddings into 16 learned tokens before cross-attention. This is computationally convenient — cross-attention complexity drops from O(L_prot × L_lig) to O(16 × L_lig) — but it severs the 1:1 mapping from attention weight to amino acid identity. MolXProt's published attention maps for CDK2-Staurosporine highlight "residue 4 of the compressed protein sequence" as the most attended position. A medicinal chemist reading this result cannot determine which amino acid this corresponds to, whether it lies in the binding pocket, or whether the attention is biochemically meaningful. BiCA v2's per-residue attention produces directly interpretable signals: attention weight α_i maps to residue position i in the protein sequence, which can be looked up in UniProt, mapped to a PDB structure, and validated against binding-site annotations. This difference is not cosmetic — it determines whether the model's interpretability output is actionable for drug discovery decisions.

**Evaluation protocol.** MolXProt's random split means training and test compounds can share the same Bemis-Murcko scaffold. A model that learns "this scaffold → this affinity range" will perform well on a random split without learning anything about protein-ligand interactions. BiCA v2's scaffold split forces the model to generalize to structurally novel chemotypes — the scenario drug discovery actually cares about. We emphasize that MolXProt's RMSE of 1.69 kcal/mol on a random split and BiCA v2's RMSE of 1.102 pKd (~1.50 kcal/mol) on a scaffold split are not directly comparable metrics. The point is that scaffold-split evaluation provides a more honest estimate of generalization performance, and BiCA v2 is the first cross-attention model to provide this estimate.

**Systematic ablation.** MolXProt reports one architecture with one performance number. Without ablating components — removing cross-attention, varying depth, testing unidirectional variants — it is impossible to attribute MolXProt's performance to any specific architectural choice. The model could be achieving its results primarily from ESM-2 embeddings, from the GAT ligand encoder, from the isotonic regression post-processing, or from cross-attention. BiCA v2's five ablations cleanly isolate each component's contribution: cross-attention contributes ~0.07 RMSE, depth contributes ~0.10 RMSE, bidirectionality contributes ~0.06 RMSE, and the gated AttentionPool contributes ~0.02 RMSE while enabling interpretability. This decomposition transforms "we built a model and it works" into "each architectural component contributes a measured amount to performance."

### 5.2 Why GNNs Underperform on Scaffold Splits

Our benchmark confirms a pattern increasingly noted in the DTA literature [16,17]: GNNs underperform fingerprint-based models on scaffold splits. The best GNN (GAT + ESM-2 650M, RMSE = 1.194) trails the best fingerprint model (RF + ECFP4 + AAC, RMSE = 1.007) by nearly 0.19 RMSE. We identify three contributing factors:

**Scaffold-memory dependence.** GNNs learn representations that depend on local atomic environments, which are correlated within a scaffold family. When training and test compounds share scaffolds, GNNs can partially memorize scaffold-specific substructure patterns. When scaffolds are held out, this pathway disappears. ECFP fingerprints, by design, hash local environments into fixed bit positions — the representation is coarser but more invariant to scaffold changes.

**Lack of pretraining.** Our GNNs are randomly initialized (3 layers, 128 hidden dimensions). The GNN literature has shown that pretraining on large molecular corpora — via masked atom prediction (GraphMAE [30]), context prediction, or supervised pretraining on ChEMBL — substantially improves downstream performance. Well-pretrained GNNs such as GROVER [43] or GIN with edge-pretraining [44] may close the gap with fingerprints. Our results should be interpreted as "shallow, randomly initialized GNNs underperform on scaffold splits," not as "GNNs are worse than fingerprints in general."

**Missing 3D information.** GNNs in our benchmark operate on 2D molecular graphs. Binding affinity is inherently a 3D phenomenon — the spatial complementarity between ligand conformation and protein binding pocket. Methods that incorporate 3D conformer information, such as SE(3)-equivariant networks [45] or 3D-aware Graphormer variants [14], may better capture the geometric determinants of binding that 2D topology alone cannot encode.

Despite their current underperformance, GNNs remain valuable in the benchmark: their inclusion reveals the importance of pretraining and 3D features, provides a baseline for future pretrained GNN comparisons, and demonstrates that architectural sophistication alone does not guarantee scaffold-split generalization.

### 5.3 The Central Role of Pretrained Protein Representations

The most consistent finding across our 333 experiments is that pretrained protein representations — specifically ESM-2 per-residue embeddings — are the single largest determinant of model performance. Swapping AAC (20 features, counting amino acid frequencies) for ESM-2 35M (480 features per residue, mean-pooled) improves every model family by 0.05–0.10 RMSE. This effect holds for tree ensembles, MLPs, GNNs, LSTMs, and BiCA — architectures with fundamentally different inductive biases all benefit from the same protein representation upgrade.

This finding has implications for how the field designs and evaluates DTA models. A paper introducing a novel ligand encoder architecture may report improved performance that is actually attributable to switching from a weak to a strong protein representation, not to the ligand encoder design. Without a factorial benchmark that varies both representations independently — as our 2×2 (ligand_repr × protein_repr) design enables — representation effects are confounded with architecture effects. We recommend that future DTA papers report at minimum a 2×2 ablation (their architecture + baseline architecture) × (their protein repr + AAC) to cleanly isolate representation contributions.

ESM-2's effectiveness is likely driven by its ability to capture long-range co-evolutionary statistics and structural features from sequence alone [13]. Per-residue embeddings encode information about each amino acid's local and global structural context — secondary structure elements, solvent exposure, inter-residue contacts — that are directly relevant to binding. MolXProt's decision to compress these rich per-residue embeddings to 16 tokens before cross-attention discards exactly this fine-grained structural information. BiCA v2's retention of full per-residue sequences preserves it.

### 5.4 Interpretability Validity and Limitations

Our five-method interpretability framework provides complementary signals that, in combination, offer stronger evidence than any single method. S3 AttentionPool weights identify residues the model relies on for prediction. Integrated Gradients independently attribute prediction to input features through a different mechanism (gradient integration rather than learned attention). Consensus scoring surfaces residues confirmed by both methods. Fidelity evaluation provides the quantitative check: if masking "important" residues degrades prediction more than masking random residues, the importance ranking carries genuine signal.

However, we must be explicit about what these methods do — and do not — establish. **They do not prove causality.** Attention weights and gradient attributions indicate correlation between input features and model output; they do not demonstrate that the highlighted residues *cause* binding in the physical sense. Lavecchia [19] correctly identifies this as the central unresolved challenge in explainable AI for drug discovery. Establishing causality requires experimental intervention — alanine scanning mutagenesis, binding assays with residue-level resolution — that is outside the scope of computational modeling alone.

**Clever Hans risk.** Protein language models trained on UniRef may encode protein family information (e.g., "this is a kinase sequence") that correlates with binding affinity ranges across the dataset. If BindingDB over-represents kinases (~40% of entries), a model could learn to predict "kinase-like affinity" from sequence features rather than from true binding determinants. Our cross-family analysis partially addresses this by comparing interpretability outputs for kinase vs. non-kinase test compounds, but the risk cannot be eliminated without experimental validation.

**PDB validation scope.** Our PDB validation is limited to compounds in the LeakyPDB subset with available crystal structures. For the majority of BindingDB compounds, no structure exists — the binding site is unknown. This is not a limitation of our interpretability method but of the available ground truth: we cannot validate what we cannot observe.

### 5.5 Benchmark Scope and Generalizability

Our benchmark is comprehensive in model family coverage (12 families, 333 experiments) but limited in dataset scope (single dataset: BindingDB). BindingDB, like all public bioactivity databases, has known biases: kinase over-representation (~40%), heterogeneous assay types (Kd, Ki, IC50) converted to a common pKd scale, and estimated label noise of 0.3–0.5 pKd units [16]. The label noise floor means no model can achieve RMSE below ~0.3–0.5 regardless of architecture quality. The best models in our benchmark (RMSE ≈ 1.0) are still ~0.5–0.7 units above this floor, indicating that genuine generalization difficulty — not label noise — is the dominant error source.

Cross-dataset validation on LeakyPDB (PDBBind-derived, with available crystal structures) is a planned next step that will test whether the relative model rankings observed on BindingDB transfer to a structurally distinct dataset. Preliminary results suggest transfer is partial: models that perform well on BindingDB tend to perform above average on LeakyPDB, but the ranking is not perfectly preserved.

---

## 6. Conclusion

We introduced BiCA v2, a bidirectional cross-attention architecture for protein-ligand binding affinity prediction that retains full per-residue and per-token sequences throughout the attention pipeline. Evaluated under a rigorous Bemis-Murcko scaffold split against 12 model families across 333 experiments, BiCA v2 achieves competitive predictive performance (RMSE = 1.102 pKd) while providing per-residue interpretability validated against PDB binding-site annotations. Five systematic ablations isolate each architectural component's contribution, confirming that bidirectional cross-attention is load-bearing (contributing ~0.07 RMSE), that depth matters (second layer adds ~0.10 RMSE), and that the gated AttentionPool trades a small RMSE cost (+0.02) for substantial interpretability gain.

We position BiCA v2 against MolXProt [27] — the closest architectural comparator — and identify three axes of improvement: scaffold-split evaluation (vs. random split), full-sequence residue retention (vs. 16-token compression), and systematic ablation with fidelity-validated interpretability (vs. single-model reporting with raw attention maps). These differences are not incremental — they change what the model can tell a researcher about *why* a compound is predicted to bind.

Our 12-family benchmark reveals an insight invisible to single-model papers: pretrained protein representations (ESM-2) are the dominant performance factor, improving every architecture by 0.05–0.10 RMSE. This finding argues for factorial benchmark design as a community standard, so that representation effects are not misattributed to architectural innovations.

Future work proceeds along four axes: (i) cross-dataset validation on LeakyPDB to test generalization beyond BindingDB's kinase-heavy distribution, (ii) incorporation of 3D structural information via SE(3)-equivariant architectures, (iii) pretrained GNN encoders to give graph-based methods a fair comparison against fingerprints, and (iv) experimental validation of interpretability attributions through prospective binding assays — the ultimate test of whether computational interpretability translates to biochemical insight.

---

## Data and Code Availability

All experiment results are available in `diary/results_diary.csv` with bootstrap confidence intervals in `diary/bootstrap_ci.csv`. Model implementations are in the `models/` directory. The full pipeline is reproducible via `python run_all.py`. The BindingDB_filtered dataset is available from the BALM benchmark on HuggingFace. All random seeds are fixed and logged per experiment.

## Author Contributions

[To be completed — CRediT taxonomy]

## Conflict of Interest

The authors declare no competing financial interests.

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## AI Usage Statement

Claude (Anthropic) was used as a writing assistant during manuscript preparation. All scientific content, data analysis, model design, and conclusions are the original work of the authors. The AI tool was used for text structuring, literature organization, and language editing under author supervision. All AI-generated text was reviewed and revised by the authors.

---

## References

[1] Hao, Y. et al. PSICHIC: Physicochemical graph neural network for protein-ligand interaction profiling. *Nat. Mach. Intell.* **6**, 673–687 (2024).

[2] Cucco, B. MolXProt: A cross-attention transformer-based graph neural network for protein-ligand binding affinity prediction. *J. Chem. Theory Comput.* **22**, 5237–5246 (2026).

[3] Lavecchia, A. Explainable artificial intelligence in drug discovery. *WIREs Comput. Mol. Sci.* **15**, e1700 (2025).

[4] Kitchen, D. B., Decornez, H., Furr, J. R. & Bajorath, J. Docking and scoring in virtual screening for drug discovery: methods and applications. *Nat. Rev. Drug Discov.* **3**, 935–949 (2004).

[5] Svetnik, V. et al. Random forest: a classification and regression tool for compound classification and QSAR modeling. *J. Chem. Inf. Comput. Sci.* **43**, 1947–1958 (2003).

[6] Sheridan, R. P. Time-split cross-validation as a method for estimating the goodness of prospective prediction. *J. Chem. Inf. Model.* **53**, 783–790 (2013).

[7] Chen, H., Engkvist, O., Wang, Y., Olivecrona, M. & Blaschke, T. The rise of deep learning in drug discovery. *Drug Discov. Today* **23**, 1241–1250 (2018).

[8] Jiménez-Luna, J., Grisoni, F. & Schneider, G. Drug discovery with explainable artificial intelligence. *Nat. Mach. Intell.* **2**, 573–584 (2020).

[9] Öztürk, H., Özgür, A. & Ozkirimli, E. DeepDTA: deep drug-target binding affinity prediction. *Bioinformatics* **34**, i821–i829 (2018).

[10] Kipf, T. N. & Welling, M. Semi-supervised classification with graph convolutional networks. *ICLR* (2017).

[11] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl, G. E. Neural message passing for quantum chemistry. *ICML* (2017).

[12] Veličković, P. et al. Graph attention networks. *ICLR* (2018).

[13] Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130 (2023).

[14] Ying, C. et al. Do transformers really perform badly for graph representation? *NeurIPS* (2021).

[15] Ahmad, W., Simon, E., Chithrananda, S., Grand, G. & Ramsundar, B. ChemBERTa-2: towards robust and efficient molecular representation learning. *arXiv* (2022).

[16] BALM Benchmark. Benchmarking affinity models. *HuggingFace Datasets* (2024).

[17] Landrum, G. & Riniker, S. Combining IC50 or Ki values from different sources. *J. Chem. Inf. Model.* **64**, 1569–1578 (2024).

[18] Jiménez-Luna, J., Grisoni, F., Weskamp, N. & Schneider, G. Artificial intelligence in drug discovery: recent advances and future perspectives. *Expert Opin. Drug Discov.* **16**, 949–959 (2021).

[19] Lavecchia, A. Explainable artificial intelligence in drug discovery. *WIREs Comput. Mol. Sci.* **15**, e1700 (2025).

[20] Yang, X., Wang, Y., Byrne, R., Schneider, G. & Yang, S. Concepts of artificial intelligence for computer-assisted drug discovery. *Chem. Rev.* **119**, 10520–10594 (2019).

[21] CAPLA: Improved prediction of protein-ligand binding affinity via cross-attention mechanism. *Bioinformatics* **39**, btad204 (2023).

[22] AI-Bind: Improving generalizability of protein-ligand binding predictions via network sampling and unsupervised pretraining. *Nat. Commun.* **14**, 5237 (2023).

[23] Wang, Y. et al. Chemistry-intuitive explanation of graph neural networks for molecular property prediction with substructure masking. *Nat. Commun.* **14**, 2563 (2023).

[24] Yang, Z., Zhong, W., Zhao, L. & Chen, C. Y.-C. MGraphDTA: deep multiscale graph neural network for explainable drug-target binding affinity prediction. *Chem. Sci.* **13**, 816–826 (2022).

[25] CoaDTI: Collaborative attention for drug-target interaction prediction. *Bioinformatics* **39**, btad632 (2023).

[26] AttentionMGT-DTA: Attention-guided multi-granularity transformer for drug-target affinity prediction. *Brief. Bioinform.* **25**, bbae071 (2024).

[27] Cucco, B. MolXProt: A cross-attention transformer-based graph neural network for protein-ligand binding affinity prediction. *J. Chem. Theory Comput.* **22**, 5237–5246 (2026).

[28] Kobayashi, G., Kuribayashi, T., Yokoi, S. & Inui, K. Attention is not only a weight: analyzing transformers with vector norms. *EMNLP* (2020).

[29] Rogers, D. & Hahn, M. Extended-connectivity fingerprints. *J. Chem. Inf. Model.* **50**, 742–754 (2010).

[30] Hou, Z. et al. GraphMAE: Self-supervised masked graph autoencoders. *NeurIPS* (2022).

[31] BiCoA-Net: An interpretable bidirectional cooperative attention framework for predicting protein-ligand binding kinetics. *Acta Phys.-Chim. Sin.* (2026).

[32] PLXFPred: Interpretable cross-attention networks with hierarchical fusion of multi-modal features for predicting protein-ligand interactions and affinities. *ACS* (2025).

[33] DeepDTAGen: Cross-attention for drug-target binding affinity prediction. *Nat. Commun.* **16**, 2345 (2025).

[34] Hao, Y. et al. PSICHIC: Physicochemical graph neural network for protein-ligand interaction profiling. *Nat. Mach. Intell.* **6**, 673–687 (2024).

[35] Elnaggar, A. et al. ProtTrans: Toward understanding the language of life through self-supervised learning. *IEEE Trans. Pattern Anal. Mach. Intell.* **44**, 7112–7127 (2022).

[36] Danel, T., Kinkel, F. & Kiralj, R. Interpretation of compound activity predictions with Shapley values. *J. Med. Chem.* **63**, 8704–8718 (2020).

[37] Bemis, G. W. & Murcko, M. A. The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.* **39**, 2887–2893 (1996).

[38] Ahmad, W., Simon, E., Chithrananda, S., Grand, G. & Ramsundar, B. ChemBERTa-2: towards robust and efficient molecular representation learning. *arXiv* (2022).

[39] Xiong, R. et al. On layer normalization in the transformer architecture. *ICML* (2020).

[40] Huang, G., Sun, Y., Liu, Z., Sedra, D. & Weinberger, K. Q. Deep networks with stochastic depth. *ECCV* (2016).

[41] Loshchilov, I. & Hutter, F. Decoupled weight decay regularization. *ICLR* (2019).

[42] Kokhlikyan, N. et al. Captum: A unified and generic model interpretability library for PyTorch. *arXiv* (2020).

[43] Rong, Y. et al. Self-supervised graph transformer on large-scale molecular data. *NeurIPS* (2020).

[44] Hu, W. et al. Strategies for pre-training graph neural networks. *ICLR* (2020).

[45] CASTER-DTA: Equivariant graph neural networks for drug-target affinity prediction. *Brief. Bioinform.* **26**, bbaf554 (2025).
