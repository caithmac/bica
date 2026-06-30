## Peer Review Report — BiCA v2 Paper

**Review type:** Simulated double-blind internal review
**Date:** 2026-06-20

---

### Five-Dimension Scoring

| Dimension | Weight | Score (1-10) | Weighted |
|-----------|--------|-------------|----------|
| Originality | 20% | 7 | 1.4 |
| Methodological Rigor | 25% | 8 | 2.0 |
| Evidence Sufficiency | 25% | 7 | 1.75 |
| Argument Coherence | 15% | 8 | 1.2 |
| Writing Quality | 15% | 7 | 1.05 |
| **Total** | **100%** | | **7.4** |

**Decision:** Accept with minor revisions

---

### Dimension Details

#### 1. Originality (7/10)

**Strengths:**
- First bidirectional cross-attention model evaluated under scaffold split — this is a genuine gap
- Full-sequence retention vs. compressed-token approach (MolXProt) is a clear conceptual advance
- Five-method interpretability framework with fidelity evaluation exceeds typical DTA paper standards
- 12-family benchmark scale is unprecedented in DTA literature

**Weaknesses:**
- Bidirectional cross-attention itself is not novel — MolXProt, CAPLA, BiCoA-Net all do it
- The individual architectural components (Pre-LN, gated pooling, value-weighted attention) are borrowed from other domains
- No prospective experimental validation — pure computational benchmark
- The core finding ("ESM-2 is the dominant factor") is confirmatory of existing knowledge

**Recommendation:** Strengthen the novelty narrative around the *combination* — scaffold split + full sequence + fidelity evaluation — rather than any single component. The contribution is the rigorous synthesis, not the individual pieces.

#### 2. Methodological Rigor (8/10)

**Strengths:**
- Scaffold split with multi-seed validation (3 seeds) — community best practice
- Bootstrap 95% CIs — statistical rigor beyond typical DTA benchmarks
- Five systematic ablations cleanly isolate component contributions
- Append-only results diary — reproducible and auditable
- Fixed seeds, logged hyperparameters — meets reproducibility standards

**Weaknesses:**
- Single dataset (BindingDB) — generalizability not demonstrated
- No hyperparameter optimization per model family — fixed architecture variants may disadvantage some families
- Label noise floor discussed but not explicitly modeled
- Kinase over-representation bias acknowledged but not corrected for
- Missing the 2×2 factorial ablation (ligand_repr × protein_repr) recommended in Discussion §5.3 for all models

**Recommendation:** Add the missing 2×2 ablation as a supplementary table if the data exists. Acknowledge the hyperparameter optimization limitation more prominently.

#### 3. Evidence Sufficiency (7/10)

**Strengths:**
- 333 experiments across 12 families — comprehensive
- All results from single append-only CSV — auditable
- Multi-seed mean ± std for key models
- PDB validation of attention attributions (precision 0.62, recall 0.38)
- Fidelity evaluation with statistical significance test

**Weaknesses:**
- PDB validation limited to LeakyPDB subset — n not reported (how many complexes?)
- No cross-dataset validation (LeakyPDB results mentioned as "preliminary")
- Fidelity evaluation details sparse — effect sizes per interpretability method not compared
- The "MolXProt comparison" is qualitative/conceptual, not quantitative (different datasets prevent direct comparison)
- Missing negative control for interpretability: what do attributions look like for a randomly-initialized BiCA v2?

**Recommendation:** Report n for PDB validation. Add random-initialization baseline for interpretability. Run LeakyPDB cross-validation if feasible, or clearly mark as future work.

#### 4. Argument Coherence (8/10)

**Strengths:**
- Clear thesis: BiCA v2 fills a specific gap (scaffold-split cross-attention with full sequences)
- MolXProt as consistent comparator throughout — provides narrative thread
- Each section's argument builds on the previous
- Counter-arguments anticipated and addressed in Discussion
- Logical structure follows claim-evidence chain

**Weaknesses:**
- Introduction could more sharply state "this is NOT the first cross-attention DTA model" to avoid overclaiming
- The transition from "RF beats BiCA v2 on RMSE" (Results) to "BiCA v2 is valuable because interpretability" (Discussion) needs a bridging paragraph
- The five sub-arguments are clear to the authors but a reader might lose track — a summary paragraph at the end of Introduction enumerating contributions would help navigation

#### 5. Writing Quality (7/10)

**Strengths:**
- Appropriate academic register for ACS/RSC journal
- Technical descriptions precise (architecture details, equation notation)
- Section structure is predictable and scannable
- Good use of quantitative hedging ("~0.07 RMSE" not "exactly 0.07")
- Honest about limitations — no overclaiming detected

**Weaknesses:**
- Some paragraphs are uniform in length (4-5 sentences) — vary rhythm
- A few throat-clearing constructions remain ("It is well-known that...", "The last five years have seen...")
- Em dashes used 12 times across ~8,000 words — within acceptable range but monitor
- No AI-typical overused terms ("delve into", "crucial", "paramount") — clean
- References section has placeholder-like entries that undermine credibility if not fixed

---

### Revision Action Items

**High priority (blocking submission):**
1. Fill 8 incomplete references with full author names and venue details
2. Report n for PDB validation subset
3. Add random-initialization baseline for interpretability fidelity comparison

**Medium priority (strengthen before submission):**
4. Add bridging paragraph between Results and Discussion on "why interpretability justifies the RMSE gap"
5. Run 2×2 factorial ablation (ligand_repr × protein_repr) if data exists
6. Vary paragraph length in Introduction and Methods sections
7. Replace 2-3 throat-clearing openers with direct claims

**Low priority (nice to have):**
8. Cross-dataset validation on LeakyPDB (or clearly mark as future work)
9. Add contribution enumeration paragraph at end of Introduction
10. Per-method fidelity comparison (S3 vs. IG vs. consensus)

---

### Writing Quality Check Results

| Check | Status |
|-------|--------|
| AI-typical terms (delve, crucial, paramount, etc.) | ✅ Clean — 0 instances |
| Em dash count | ⚠️ 12 in ~8000 words (~1.5/1000) — borderline acceptable |
| Throat-clearing openers | ⚠️ 3 instances: "It is well-known...", "The last five years...", "It is increasingly recognized..." |
| Uniform paragraph lengths | ⚠️ Methods §3.3-3.5 — 4 consecutive paragraphs of similar length |
| Monotonous sentence rhythm | ✅ Acceptable — mix of short declarative and longer explanatory sentences |

---

### Verdict

This is a solid paper that fills a genuine gap. The architecture is not conceptually novel (bidirectional cross-attention exists), but the **rigorous evaluation protocol** — scaffold split + multi-seed + bootstrap CIs + systematic ablation — is what makes the contribution publishable. The MolXProt comparison crystallizes the advantages. The interpretability framework meets Lavecchia (2025) standards better than most DTA papers.

**Recommendation:** Fix the 8 incomplete references (blocking), add the 3 high-priority items, and submit. Target: ACS JCIM, RSC Digital Discovery, or JCTC (MolXProt's venue — positioning as improvement over prior JCTC work is a strong submission narrative).
