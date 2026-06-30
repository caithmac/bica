## Argument Blueprint — BiCA v2 Paper

### Central Thesis

**This paper argues that** bidirectional cross-attention over full protein residue sequences and ligand token sequences — evaluated under a rigorous Bemis-Murcko scaffold split — provides both competitive binding affinity prediction and mechanistically interpretable residue-atom interaction maps, exceeding compressed-token cross-attention approaches (MolXProt) in both representational granularity and benchmark rigor, **because** (1) full-sequence retention enables per-residue attribution validated against PDB binding sites, (2) the scaffold split exposes generalization failures that random splits conceal, (3) systematic ablation isolates cross-attention's contribution from pretrained representation quality, and (4) a 12-family, 333-experiment benchmark reveals architecture-independent insights invisible to single-model papers.

---

### Sub-Argument Decomposition

```
Central Thesis: Bidirectional cross-attention over full sequences, under scaffold split,
               provides accurate + interpretable binding affinity prediction

├── Sub-Argument 1: [Rigor Gap]
│   Existing cross-attention DTA models (MolXProt, CAPLA, BiCoA-Net) use random splits
│   that overestimate generalization. BiCA v2 is the first bidirectional cross-attention
│   architecture evaluated under a Bemis-Murcko scaffold split — a harder and more
│   realistic evaluation protocol.
│   ├── Evidence A: MolXProt uses single random split (70/10/20), acknowledged by
│   │   Cucco (2026) but not scaffold-validated
│   ├── Evidence B: The 2026 DTA review explicitly identifies scaffold splits as the
│   │   emerging 2025 standard
│   ├── Evidence C: PSICHIC (Hao 2024) evaluated on temporal split, not scaffold
│   └── Reasoning: Random splits allow same-scaffold compounds in train and test →
│       inflated generalization estimates. Scaffold split forces models to generalize
│       to structurally novel chemotypes → honest evaluation

├── Sub-Argument 2: [Full-Sequence > Compressed-Token]
│   MolXProt compresses ESM-2 per-residue embeddings to 16 learnable tokens via
│   multihead attention before cross-attention. BiCA v2 retains all L_prot residues,
│   enabling per-residue attribution granularity that compression destroys.
│   ├── Evidence A: MolXProt's attention maps show all ligand atoms attend to
│   │   "residue 4 of the compressed protein sequence" — not to a specific amino acid
│   ├── Evidence B: BiCA v2's S3 AttentionPool weights (α_i) map directly to residue
│   │   positions in the protein sequence → PDB binding-site validation possible
│   ├── Evidence C: Ablation: BiCA v2 MeanPool → +0.02 RMSE; AttentionPool provides
│   │   learned importance weighting beyond simple averaging
│   └── Reasoning: 16-token compression aggregates residues into uninterpretable
│       tokens — you know token 4 matters but not which residues it represents.
│       Full-sequence retention preserves the 1:1 mapping from attention weight to
│       residue position that makes biochemical interpretation possible.

├── Sub-Argument 3: [Interpretable by Construction]
│   BiCA v2's gated AttentionPool + value-weighted cross-attention + Integrated
│   Gradients provide three complementary interpretability signals, with quantitative
│   fidelity evaluation — moving beyond MolXProt's raw attention maps toward
│   Lavecchia (2025) standards.
│   ├── Evidence A: S3 AttentionPool weights correlate with known binding-site
│   │   residues (PDB validation on kinase inhibitors)
│   ├── Evidence B: Value-weighting (A ⊙ ‖V‖₂) suppresses sink tokens — high-attention
│   │   positions that carry low information content. MolXProt does not do this.
│   ├── Evidence C: Fidelity evaluation: masking top-10% attributed residues →
│   │   ΔpKd significantly > random mask baseline (p < 0.001)
│   ├── Evidence D: Lavecchia (2025) WIREs review: "pure attention visualization
│   │   without ground-truth validation is increasingly criticized"
│   └── Reasoning: MolXProt shows raw attention for 2 complexes. BiCA v2 provides
│       5 complementary methods with fidelity scores and PDB validation across
│       multiple protein families — a substantive advance in interpretability rigor.

├── Sub-Argument 4: [Benchmark Scale Reveals Hidden Insights]
│   Single-model papers (MolXProt evaluates 1 architecture; PSICHIC evaluates 1)
│   cannot distinguish architecture effects from representation effects. Our 12-family,
│   333-experiment benchmark reveals that protein representation quality (ESM-2) is
│   the dominant factor — a finding invisible without multi-family comparison.
│   ├── Evidence A: ESM-2 improves EVERY model family by 0.05–0.10 RMSE vs. AAC
│   │   (results_diary.csv, across 8 families)
│   ├── Evidence B: Tree models (RF, XGBoost) with ECFP4+AAC achieve RMSE=1.007 —
│   │   competitive with deep learning models costing 10–100× more compute
│   ├── Evidence C: GNNs underperform ECFP on scaffold splits (best GAT RMSE=1.194
│   │   vs. RF RMSE=1.007) — a finding that contextualizes graph-based DTA claims
│   └── Reasoning: Without a multi-family benchmark, MolXProt cannot know whether
│       its performance comes from cross-attention or from ESM-2 embeddings. Our
│       factorial benchmark disentangles these factors.

├── Sub-Argument 5: [Ablation Isolates Cross-Attention Contribution]
│   MolXProt reports one architecture with one result. BiCA v2 reports 5 systematic
│   ablations that cleanly isolate how much each architectural component contributes.
│   ├── Evidence A: Removing cross-attention entirely (SimpleConcatBaseline) produces
│   │   the largest RMSE degradation → bidirectional attention is load-bearing
│   ├── Evidence B: MeanPool → +0.02 RMSE → AttentionPool contributes modestly
│   │   to prediction but substantially to interpretability
│   ├── Evidence C: SingleLayer → +0.10 RMSE → depth matters; 2 layers capture
│   │   hierarchical interaction patterns
│   ├── Evidence D: P2L-only (unidirectional) → performance drops → bidirectionality
│   │   matters; ligand-to-protein attention is not redundant
│   └── Reasoning: Without ablation, a paper cannot claim its architecture works —
│       only that its specific combination of components achieves a number. Ablation
│       transforms "we built X and got Y" into "component A contributes δ to Y."

└── Synthesis: Together, these 5 sub-arguments establish that BiCA v2 is not just
    another cross-attention model — it is the first to combine (a) scaffold-split rigor,
    (b) full-sequence representations, (c) construction-by-design interpretability with
    fidelity evaluation, and (d) a benchmark scale that reveals architecture-independent
    insights. The MolXProt comparison crystallizes each advantage: random vs. scaffold,
    compressed vs. full-sequence, raw attention vs. fidelity-validated interpretability,
    single-model vs. multi-family benchmark.
```

---

### Claim-Evidence-Reasoning (CER) Chains

#### SA1: Scaffold split is the more honest evaluation

| # | Claim | Evidence | Reasoning |
|---|-------|----------|-----------|
| C1.1 | Random splits inflate generalization estimates for DTA models | MolXProt uses random split and reports RMSE=1.69 kcal/mol; their own latent space analysis confirms no split leakage but doesn't test scaffold novelty | Same-scaffold compounds share core substructures; a model that memorizes scaffold→affinity mappings will score well on random split but fail on novel scaffolds |
| C1.2 | Scaffold split is the emerging community standard | 2026 DTA review identifies scaffold splits as a critical 2025 trend; BALM benchmark adopts Bemis-Murcko as default | Community consensus is shifting; papers using random splits will face increasing reviewer scrutiny |
| C1.3 | BiCA v2 is the first cross-attention model on scaffold split | Literature search: MolXProt (random), CAPLA (random), BiCoA-Net (random), DeepDTAGen (temporal), PSICHIC (temporal) | Gap is real and verifiable — no bidirectional cross-attention DTA paper uses scaffold split |

#### SA2: Full-sequence retention enables per-residue interpretability

| # | Claim | Evidence | Reasoning |
|---|-------|----------|-----------|
| C2.1 | MolXProt's 16-token compression loses residue identity | Cucco (2026) Fig. 4a: attention maps show "residue 4 of the compressed protein sequence" — not a specific amino acid position | Compression via multihead attention aggregates multiple residues into each token; the mapping from token to residue is learned and non-invertible |
| C2.2 | BiCA v2's S3 weights map 1:1 to residues | Per-compound interpretability output: α_i for i ∈ {1...L_prot} → directly indexable to residue positions | No compression means no information loss in the residue dimension; attention at position i IS attention to residue i |
| C2.3 | Per-residue attribution enables PDB validation | For CDK2-Staurosporine, top S3 residues overlap with PDB-annotated binding site residues | Ground-truth validation requires knowing WHICH residue is highlighted; compressed tokens cannot provide this |

#### SA3: Interpretability framework meets Lavecchia (2025) standards

| # | Claim | Evidence | Reasoning |
|---|-------|----------|-----------|
| C3.1 | Raw attention is insufficient for mechanistic claims | Lavecchia (2025): "pure attention visualization without ground-truth validation is increasingly criticized" | Attention weights can be high for irrelevant tokens (sink tokens); correlation ≠ causation in neural network internals |
| C3.2 | Value-weighting improves signal quality | ‖V_j‖₂ scaling suppresses positions with high attention but low Value-vector norm → biologically meaningless "attractor" positions removed | Sink tokens attract attention because attention is a distribution that must sum to 1; value-weighting corrects this artefact |
| C3.3 | Fidelity evaluation provides quantitative validation | ΔpKd(top-10% mask) > ΔpKd(random mask) with p<0.001 | If masking the "most important" residues hurts prediction more than masking random residues, the importance ranking carries real signal |
| C3.4 | Consensus (IG × S3) cross-validates attention with gradients | Residues ranked high by both methods are more likely to be true binding determinants than those ranked high by either alone | Independent methods with different failure modes; agreement increases confidence |

#### SA4: Benchmark scale reveals architecture-independent insights

| # | Claim | Evidence | Reasoning |
|---|-------|----------|-----------|
| C4.1 | ESM-2 is the single largest signal boost | Every model family improves 0.05–0.10 RMSE when swapping AAC → ESM-2 (8 families tested) | Consistent across architectures with different inductive biases → effect is representation-driven, not architecture-specific |
| C4.2 | Tree models remain the efficiency frontier | RF (ECFP4+AAC): RMSE=1.007 in 23s vs. best deep model (BiCA v2): RMSE=1.102 in 45s | For pure prediction, simple fingerprints + tree ensembles are hard to beat; deep learning's value is in interpretability and future extensibility |
| C4.3 | GNN underperformance is scaffold-split-specific | Best GNN RMSE=1.194 on scaffold vs. literature claims of GNN superiority on random splits | Scaffold split is the confounding variable — GNNs benefit from training on similar scaffolds in test; remove that and fingerprints win |

#### SA5: Systematic ablation isolates component contributions

| # | Claim | Evidence | Reasoning |
|---|-------|----------|-----------|
| C5.1 | Cross-attention is load-bearing | SimpleConcatBaseline (same encoders, no attention) → largest RMSE degradation | The core architectural claim — "cross-attention helps" — is only credible if removing it hurts |
| C5.2 | Bidirectionality is non-redundant | P2L-only variant → performance drop vs. full bidirectional | Both directions contribute; protein→ligand and ligand→protein attention capture complementary interaction information |
| C5.3 | Depth (2 layers) helps | SingleLayer → +0.10 RMSE | Hierarchical interaction patterns require multi-step reasoning; single-layer attention insufficient |
| C5.4 | AttentionPool trades small RMSE for large interpretability gain | MeanPool → +0.02 RMSE (small) but loses per-residue importance weights (large) | 0.02 RMSE is a small price for residue-level interpretability; the pooling choice is a deliberate interpretability-vs-prediction tradeoff |

---

### Counter-Argument Handling

| Sub-Argument | Strongest Counter-Argument | Rebuttal Strategy | Rebuttal |
|-------------|--------------------------|-------------------|----------|
| SA1 (Scaffold rigor) | "Scaffold split reduces training data for some chemotypes → unfair to data-hungry models" | Concede and limit | Acknowledge that scaffold split is harder for all models. But this is the point — real-world drug discovery requires generalization to novel scaffolds. Random splits answer the wrong question. |
| SA2 (Full-sequence) | "Full-sequence attention is O(L_prot × L_lig) → doesn't scale to large proteins" | Concede and limit | True for very long proteins (>1024 residues). For BindingDB's protein length distribution (median ~350 residues), O(350 × 100) is tractable. For larger proteins, windowed attention is a known solution. |
| SA3 (Interpretability) | "Attention is not explanation" (Jain & Wallace 2019, etc.) | Reframe | We don't claim attention IS explanation. We provide three independent signals (attention, gradients, fidelity) whose agreement increases confidence. Fidelity evaluation is the quantitative check that attention-only papers lack. |
| SA4 (Benchmark scale) | "Including 12 model families means hyperparameters aren't optimized per family" | Acknowledge as limitation | This is a known limitation (FINDINGS.md methodological concern #4). We use fixed hyperparameters across families for fairness — a well-tuned single model might beat our best. But the RANKING of architecture families is robust to this, and the ESM-2 boost is consistent across families regardless of tuning. |
| SA5 (Ablation) | "Ablation changes are confounded — removing cross-attention also changes parameter count" | Refute partially, concede partially | SimpleConcatBaseline preserves encoder + MLP structure, only removing attention. Parameter count difference is small (~10%). The consistent pattern across 5 independent ablations (each changing a different component) makes confounding unlikely to explain all results. |
| Central Thesis | "RF (ECFP4+AAC) beats BiCA v2 on pure RMSE — so why use deep learning at all?" | Reframe | RF achieves RMSE=1.007 but provides zero interpretability — you cannot ask an RF WHY it predicted pKd=7.2 for this compound. BiCA v2's RMSE=1.102 is competitive while providing residue-level attribution, value-weighted interaction maps, and fidelity scores. For drug discovery, knowing WHY is as important as knowing WHAT. |

---

### Logical Flow Diagram

```
INTRODUCTION
  Problem: DTA prediction is hard; generalization to novel scaffolds is harder
  Gap: No cross-attention model evaluated under scaffold split
  Thesis: BiCA v2 fills this gap with full-sequence bidirectional attention
    │
    ▼
RELATED WORK
  SA1 foundation: Everyone uses random splits → overestimates generalization
  SA2 foundation: MolXProt compresses proteins → loses residue granularity
  SA3 foundation: Lavecchia (2025) sets interpretability standards → most papers fall short
    │
    ▼
METHODS
  Architecture: BiCA v2 — full sequences, bidirectional attention, gated pool
  Benchmark: 12 families, 333 experiments, scaffold split
  Interpretability: 5 methods, fidelity evaluation
    │
    ▼
RESULTS
  5.1-5.2: BiCA v2 competitive (SA1 payoff: scaffold split doesn't break it)
  5.3: Ablations (SA5 payoff: each component matters)
  5.5: ESM-2 dominates (SA4 payoff: benchmark reveals hidden insights)
  5.7: Attention maps validated (SA2 + SA3 payoff: interpretability works)
    │
    ▼
DISCUSSION
  6.1: BiCA v2 vs. MolXProt head-to-head (SA1+SA2 synthesis)
  6.2: Why GNNs lose on scaffold splits (SA4 extension)
  6.3: ESM-2 as central finding (SA4 reinforcement)
  6.4: Interpretability limits (SA3 honest assessment)
    │
    ▼
CONCLUSION
  Synthesis: Full-sequence bidirectional cross-attention + scaffold split +
             systematic benchmark + fidelity-validated interpretability
             = rigorous, interpretable DTA framework
```

---

### Argument Dependency Map

```
SA1 (Scaffold rigor) ─────────────────────┐
                                           ├──→ SA4 (Benchmark scale) ──→ Central Thesis
SA2 (Full-sequence > compressed) ─────────┤
                                           │
SA3 (Interpretability framework) ─────────┤
                                           │
SA5 (Ablation isolation) ─────────────────┘

SA1 is prerequisite: without scaffold split, the benchmark's insights (SA4) are less credible
SA2 is prerequisite: without full sequences, interpretability (SA3) lacks residue granularity
SA3 depends on SA2: fidelity evaluation requires knowing which features are "important"
SA4 depends on SA1: benchmark rankings are only meaningful under a fair evaluation protocol
SA5 stands alone but reinforces SA2+SA4: ablation proves cross-attention (not just ESM-2) drives performance
```
