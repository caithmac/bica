## Paper Outline — BiCA v2: Bidirectional Cross-Attention for Drug-Target Binding Affinity Prediction

**Structure:** IMRaD (adapted for benchmark + novel architecture)
**Word count:** TBD (section proportions given; absolute counts to be scaled when target set)
**Date:** 2026-06-20

---

### Top-Level Structure

```
1. Abstract
2. Introduction
3. Related Work
4. Methods
   4.1 Dataset and Split
   4.2 Molecular Representations
   4.3 BiCA v2 Architecture
   4.4 Baseline Models
   4.5 Evaluated Model Families
   4.6 Interpretability Framework
   4.7 Training and Evaluation Protocol
5. Results
   5.1 Overall Benchmark Leaderboard
   5.2 BiCA v2 Performance Analysis
   5.3 Ablation Study
   5.4 Multi-Seed Stability
   5.5 Representation Comparison
   5.6 Computational Efficiency
   5.7 Cross-Attention Interpretability
6. Discussion
   6.1 BiCA v2 vs. MolXProt: Full-Sequence vs. Compressed-Token Cross-Attention
   6.2 Why GNNs Underperform on Scaffold Splits
   6.3 The Central Role of Pretrained Protein Representations
   6.4 Interpretability Validity and Limitations
   6.5 Benchmark Scope and Generalizability
7. Conclusion
References
```

---

### Section-by-Section Outline

#### 1. Abstract
- **Purpose:** Standalone summary of the full paper
- **Content:** 1–2 sentences each on: problem/motivation, gap (no scaffold-split cross-attention benchmark), BiCA v2 architecture highlight, key results (RMSE, Pearson, Spearman; comparison to MolXProt and PSICHIC), interpretability finding, conclusion
- **Key sources:** MolXProt, PSICHIC, BALM (cited in background context)
- **Word count:** 200–300 words

---

#### 2. Introduction — ~15%
- **Purpose:** Establish the problem, motivate the approach, state contributions
- **Content:**
  - ¶1: Drug-target binding affinity prediction is fundamental to drug discovery. Current approaches (docking, empirical scoring, ML) face generalization challenges.
  - ¶2: Two dominant paradigms — fingerprint-based models (RF, XGBoost) achieve strong results but lack mechanistic interpretability; deep learning models (GNNs, transformers) promise richer representations but underperform on rigorous scaffold splits.
  - ¶3: **The representation bottleneck.** Most models collapse protein and ligand representations to single fixed-size vectors before any interaction is modeled. This discards sequence structure — which residues matter for which ligand substructures is lost.
  - ¶4: **MolXProt** (Cucco, JCTC 2026) introduced bidirectional cross-attention for this problem but with critical limitations: random split (not scaffold), compressed 16-token protein representation, no systematic ablation. **Gap:** No bidirectional cross-attention model has been evaluated under a rigorous scaffold split with full sequence representations and a comprehensive benchmark.
  - ¶5: **Contributions:** (1) BiCA v2 — bidirectional cross-attention over full protein residue sequences and ligand token sequences with gated AttentionPool, (2) comprehensive 12-family, 333-experiment benchmark under Bemis-Murcko scaffold split, (3) five-method interpretability framework with quantitative fidelity evaluation, (4) head-to-head comparison against MolXProt and PSICHIC, (5) five systematic ablations isolating cross-attention contribution.
- **Key sources:** MolXProt (Cucco 2026), PSICHIC (Hao 2024), Lavecchia (2025), BALM (2024), DeepDTAGen (2025)
- **Transition to Related Work:** "The following section situates BiCA v2 within the landscape of binding affinity prediction methods."

---

#### 3. Related Work — ~15%
- **Purpose:** Situate the work in the literature, establish the gap precisely
- **Content:**
  - **3.1 Fingerprint and Tree-Based Approaches** — RF/XGBoost with ECFP + AAC/ESM-2; strong baselines but no mechanistic interpretability. Sources: BALM, BindingDB.
  - **3.2 Graph Neural Networks for Binding Affinity** — GCN, GAT, Graphormer; structural encoding advantage but underperformance on scaffold splits without pretraining. Sources: Graphormer (Ying 2021), GraphMAE (Hou 2022), MGraphDTA (Yang 2022), CASTER-DTA (2025).
  - **3.3 Attention and Cross-Attention Models** — CAPLA, BiCoA-Net, PLXFPred, DeepDTAGen; cross-attention as emerging paradigm. Sources: CAPLA (2023), BiCoA-Net (2026), PLXFPred (2025), DeepDTAGen (2025).
  - **3.4 MolXProt: The Closest Comparator** — Detailed description of architecture (GAT + ESM-2 → 16-token compression → bidirectional cross-attention → MLP). Strengths: lightweight, scalable, bidirectional attention concept. Weaknesses: random split, compressed protein tokens, single seed, no ablations, basic attention maps only. Sources: MolXProt (Cucco 2026).
  - **3.5 PSICHIC: The Current Gold Standard** — Physicochemical GNN, interaction fingerprints, experimental validation. Sources: PSICHIC (Hao 2024).
  - **3.6 Protein Language Models** — ESM-2, ProtElectra; per-residue embeddings encode structural and evolutionary information. Sources: ESM-2 (Lin 2023), ProtElectra (Elnaggar 2022).
  - **3.7 Interpretability Standards** — From attention visualization to quantitative fidelity. Lavecchia (2025) critique of attention-only interpretability. SME, Grad-AAM, SHAP, Integrated Gradients as complementary methods. Sources: Lavecchia (2025), SME (Wang 2023), MGraphDTA (Yang 2022), Danel (2020), Nature Methods GeoDL (2024).
- **Key sources:** All 28 literature sources distributed across subsections
- **Transition to Methods:** "To address the identified gaps — scaffold-split evaluation, full-sequence cross-attention, and rigorous interpretability — we designed BiCA v2 and a comprehensive benchmarking framework."

---

#### 4. Methods — ~20%
- **Purpose:** Complete, reproducible description of all methods
- **Content:**
  - **4.1 Dataset and Split:** BindingDB_filtered from BALM benchmark (24,700 pairs). Bemis-Murcko scaffold split (seed 42, 70/10/20). Train 17,312 / Val 2,673 / Test 4,715. pKd regression target. Justification: scaffold split ensures test compounds are structurally novel — harder, more realistic than random split.
  - **4.2 Molecular Representations:**
    - *Ligand:* ECFP4/ECFP6 (1024-bit), ChemBERTa-2 (5M/77M/100M/600, per-token 384-dim), molecular graphs (78-dim nodes, 10-dim edges), distance matrices (100×100), SMILES tokenizations (char/BPE-512/BPE-1000/atom-level)
    - *Protein:* AAC (20-dim), dipeptide composition (400-dim), k-mer (8000-dim), ESM-2 (8M/35M/150M/650M, per-residue), ESMC (300M, per-residue), ProtElectra (256-dim)
  - **4.3 BiCA v2 Architecture:**
    - ESM-2 per-residue embeddings (L_prot × 480) → Linear proj → hidden_dim (256)
    - ChemBERTa-2 per-token embeddings (L_lig × 384) → Linear proj → hidden_dim (256)
    - N=2 bidirectional CrossAttentionBlocks (Pre-LayerNorm, 8 heads, GELU FFN 4× dim, DropPath residuals)
    - Gated AttentionPool: learned scalar weight per position (α_i for protein, β_j for ligand) — replaces mean pooling
    - Predictor MLP: 512 → 256 → 1 (pKd)
    - Value-weighted attention at inference: A_scaled = A ⊙ ‖V‖₂ (suppresses sink tokens)
    - ~1.4M parameters. Trained with AdamW (lr=1e-3), MSE loss, early stopping on val RMSE.
  - **4.4 Baseline Models:** Ridge, RF, XGBoost, LightGBM — ECFP4/ECFP6/MACCS + AAC/dipeptide/ESM-2 (flat concat). MLP shallow/medium/deep with varying representation combinations.
  - **4.5 Evaluated Model Families (12 total):** Linear, Tree, MLP, CNN-1D, DistMat CNN, LSTM, Transformer-seq, Transformer (flat), GCN, GAT, Graphormer, GLI, Mamba, BiCA v1 (flat), BiCA v2 (sequence).
  - **4.6 Interpretability Framework:**
    - S3: AttentionPool protein weights (α_i per residue)
    - S4: AttentionPool ligand weights (β_j per token)
    - Value-weighted S1: A_p2l ⊙ ‖V_j‖₂ (residue × token interaction map)
    - Integrated Gradients (Captum) on input embeddings
    - Consensus: c_i = α_i^(IG) × w_i^(S3)
    - Quantitative fidelity: mask top-K features → measure ΔpKd
  - **4.7 Training and Evaluation Protocol:** Fixed seeds (42/123/456), early stopping (patience=20), metrics (RMSE primary, Pearson R, Spearman R), bootstrap 95% CIs (1000 samples). All experiments logged to append-only results_diary.csv.
- **Key sources:** BALM, ESM-2, ChemBERTa-2, ProtElectra, SME, MGraphDTA, Danel, Lavecchia, MolXProt architecture description
- **Transition to Results:** "We evaluate BiCA v2 against 12 model families across 333 experiments, with particular attention to scaffold-split generalization, cross-attention contribution, and interpretability quality."

---

#### 5. Results — ~25%
- **Purpose:** Present findings systematically, with figures and tables
- **Content:**
  - **5.1 Overall Benchmark Leaderboard:** Top-20 table (RMSE, Pearson, Spearman, train time, params). Figure: bar chart of top-10 RMSE. Key finding: RF (ECFP4+AAC) RMSE=1.007 best overall; BiCA v2 RMSE=1.102 best deep learning model.
  - **5.2 BiCA v2 Performance Analysis:** Best config: ChemBERTa-77M + ESMC, RMSE=1.102. Comparison to MolXProt: our scaffold-split RMSE of 1.102 pKd is achieved under a harder evaluation (MolXProt's RMSE=1.69 kcal/mol ≈ 1.24 pKd on random split — not directly comparable, but scaffold split is the more rigorous standard). Comparison to PSICHIC: BiCA v2 RMSE=1.102 vs. PSICHIC fine-tuned RMSE=1.176.
  - **5.3 Ablation Study:** Table with 5 variants. MeanPool → +0.02 RMSE. SingleLayer → +0.10 RMSE. NoFFN → +0.10 RMSE. P2L only (unidirectional) → performance drop. SimpleConcatBaseline (no attention) → largest drop. **Key finding:** Bidirectional cross-attention contributes ~0.08 RMSE over simple concatenation.
  - **5.4 Multi-Seed Stability:** BiCA v2 mean RMSE=1.108 ±0.037 across seeds 42/123/456. Bootstrap 95% CIs confirm stability.
  - **5.5 Representation Comparison:** ESM-2 improves every model family by 0.05–0.10 RMSE vs. AAC. ChemBERTa pretrained embeddings outperform ECFP for deep models. Tokenization comparison: BPE-512 > atom-level > char > wordpiece.
  - **5.6 Computational Efficiency:** RMSE vs. training time scatter. Tree models dominate efficiency frontier. BiCA v2 (45s training) is competitive with MLP (36s) while providing interpretability.
  - **5.7 Cross-Attention Interpretability:**
    - Per-compound residue × token attention heatmaps (Figure: representative kinase inhibitor)
    - Value-weighting vs. raw attention comparison — demonstrates sink-token suppression
    - S3 AttentionPool weights correlate with known binding-site residues (PDB validation)
    - Consensus (IG × S3) highlights residues confirmed by both gradient and attention methods
    - Fidelity: masking top-10% attributed residues → ΔpKd significantly > random mask baseline (p < 0.001)
- **Key sources:** All benchmark results from results_diary.csv. MolXProt for comparison. PSICHIC for baseline. SME, MGraphDTA for interpretability comparison.
- **Transition to Discussion:** "These results demonstrate that bidirectional cross-attention over full sequences provides both competitive predictive performance and interpretable interaction maps. We now discuss the implications, position BiCA v2 against MolXProt, and address limitations."

---

#### 6. Discussion — ~20%
- **Purpose:** Interpret results, compare to literature, acknowledge limitations
- **Content:**
  - **6.1 BiCA v2 vs. MolXProt — Full-Sequence vs. Compressed-Token Cross-Attention:** MolXProt compresses ESM-2 to 16 tokens → loses per-residue granularity → attention maps identify protein tokens, not specific residues. BiCA v2 retains all L_prot residues → S3 weights directly map to residue positions → PDB validation possible. MolXProt's random split overestimates generalization; our scaffold split is the more honest evaluation. Both papers show bidirectional cross-attention learns meaningful interaction patterns — but BiCA v2's full-sequence approach provides finer granularity and value-weighting improves signal quality.
  - **6.2 Why GNNs Underperform on Scaffold Splits:** GCN/GAT RMSE > 1.19 vs. ECFP RMSE ~1.01. Scaffold split penalizes topology-based methods heavily because scaffold-hopping removes training substructures GNNs rely on. ECFP's hashed fingerprints are more robust to scaffold changes. GNNs likely need pretraining (GraphMAE, GROVER) to compete — our randomly-initialized GNNs are disadvantaged. This is not "GNNs are worse" but "shallow, untrained GNNs are worse on scaffold splits."
  - **6.3 The Central Role of Pretrained Protein Representations:** ESM-2 embeddings are the single largest signal boost across all model families (+0.05–0.10 RMSE). Protein encoding quality matters more than ligand encoding architecture choice. ESM-2's per-residue embeddings capture evolutionary and structural information that simple AAC cannot. This finding aligns with MolXProt's use of ESM-2 but our per-residue retention extracts more value from the same embeddings.
  - **6.4 Interpretability Validity and Limitations:** Our five methods provide complementary signals, but Lavecchia (2025) correctly notes that pure attention visualization is not mechanistic proof. We provide fidelity scores (quantitative) and PDB validation (structural ground truth) but lack mutagenesis validation (experimental ground truth). The "Clever Hans" risk — models may learn dataset shortcuts (kinase family features) rather than true binding determinants. Cross-family ablation partially addresses this.
  - **6.5 Benchmark Scope and Generalizability:** Single dataset (BindingDB) limits generalizability. Kinase over-representation (~40%) may inflate ESM-2 benefit. Future: LeakyPDB cross-dataset validation, more diverse protein families (GPCRs, proteases, nuclear receptors), 3D structure incorporation.
- **Key sources:** MolXProt, PSICHIC, Lavecchia, Nature Methods GeoDL, KEPLA, CASTER-DTA
- **Transition to Conclusion:** "In summary, BiCA v2 demonstrates that bidirectional cross-attention over full molecular sequences provides a viable path toward both accurate and interpretable binding affinity prediction, with clear advantages over compressed-token approaches."

---

#### 7. Conclusion — ~5%
- **Purpose:** Summarize contributions and future directions
- **Content:**
  - Summary of 5 contributions
  - BiCA v2 as the first bidirectional cross-attention model evaluated under scaffold split with full sequence representations
  - Interpretability framework as a step toward Lavecchia (2025) standards
  - Future: 3D structure integration, cross-dataset validation, experimental validation of attributions
- **Key sources:** (none new — synthesis of paper's own findings)
- **Word count:** ~250–400 words

---

### Evidence-to-Section Mapping

| Section | Assigned Sources | Evidence Type |
|---------|-----------------|---------------|
| 2. Introduction | MolXProt, PSICHIC, Lavecchia, DeepDTAGen, BALM | Gap identification, motivation |
| 3.1 Fingerprint/Tree | BALM, BindingDB | Baseline context |
| 3.2 GNNs | Graphormer, GraphMAE, MGraphDTA, CASTER-DTA | Architecture context |
| 3.3 Cross-Attention | CAPLA, BiCoA-Net, PLXFPred, DeepDTAGen | Paradigm context |
| 3.4 MolXProt | MolXProt (Cucco 2026) | Direct comparator analysis |
| 3.5 PSICHIC | PSICHIC (Hao 2024) | Gold standard |
| 3.6 Protein LMs | ESM-2 (Lin 2023), ProtElectra (Elnaggar 2022) | Representation context |
| 3.7 Interpretability | Lavecchia, SME, MGraphDTA, Danel, Nature Methods GeoDL | Standards context |
| 4.1 Dataset | BALM, BindingDB | Dataset description |
| 4.2 Representations | ESM-2, ChemBERTa-2, ProtElectra | Featurization |
| 4.3 BiCA v2 | (novel — self-citation) | Architecture |
| 4.6 Interpretability | SME, MGraphDTA, Danel, Captum | Method justification |
| 5.1–5.6 Results | results_diary.csv, bootstrap_ci.csv | Primary data |
| 5.7 Interpretability | SME, MGraphDTA, PDB validation | Comparison data |
| 6.1 MolXProt comparison | MolXProt (Cucco 2026) | Comparator analysis |
| 6.2 GNN discussion | Graphormer, GraphMAE, MGraphDTA | Contextualization |
| 6.3 Protein reps | ESM-2, ProtElectra | Interpretation |
| 6.4 Interpretability limits | Lavecchia, Nature Methods GeoDL | Limitation framing |
| 6.5 Generalizability | KEPLA, CASTER-DTA, LeakyPDB | Future work context |

---

### Transition Logic

| Boundary | Logic |
|----------|-------|
| Intro → Related Work | "The following section situates BiCA v2 within the landscape..." |
| Related Work → Methods | "To address the identified gaps, we designed BiCA v2 and a comprehensive benchmarking framework." |
| Methods → Results | "We evaluate BiCA v2 against 12 model families across 333 experiments..." |
| Results → Discussion | "These results demonstrate... We now discuss implications and limitations." |
| Discussion → Conclusion | "In summary, BiCA v2 demonstrates that bidirectional cross-attention over full molecular sequences..." |
