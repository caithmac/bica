## Revision Roadmap — ChemCross Paper

**Venues reviewed:** ICLR (5.2), NeurIPS (weak reject), ICML (weak accept)
**Consensus signal:** Fix consistency + clarify methods → resubmit viable. Add OOD split → strong.
**Date:** 2026-06-20

---

### Consensus Matrix

| Issue | ICLR | NeurIPS | ICML | Priority |
|-------|------|---------|------|----------|
| Result inconsistencies (1.065 vs 1.102 vs 1.101) | 🔴 | 🔴 | 🟡 | **P0** |
| Placeholder citations | 🔴 | 🔴 | — | **P0** |
| Sequence length handling / hardware specs | 🔴 | 🟡 | 🔴 | **P0** |
| Only scaffold split — add protein/OOD splits | 🔴 | 🔴 | 🟡 | **P1** |
| Frozen vs fine-tuned encoders | — | 🔴 | 🔴 | **P1** |
| Value-weighting train/inference mismatch | 🔴 | 🟡 | 🟡 | **P1** |
| "First" claim overstated | — | 🔴 | — | **P2** |
| PDB validation mapping details | — | 🔴 | 🟡 | **P1** |
| Compare with FusionDTI/DrugBAN | 🟡 | 🟡 | 🟡 | **P2** |
| MolXProt comparison confounded | — | 🟡 | — | **P2** |
| Calibration analysis | — | — | 🟡 | **P3** |
| Code/article release commitment | — | 🟡 | 🟡 | **P2** |

---

### Action Items

#### P0 — MUST FIX (all reviewers flag these)

| # | Issue | Action | File/Line | Effort |
|---|-------|--------|-----------|--------|
| P0-1 | **Result inconsistencies** — text says RMSE 1.102, figure says 1.065, MLP at 1.101 | Reconcile: the 1.065 is from `bica_v2_chemberta77M_tokens__seed456` (different seed). The 1.102 is from `bica_chemberta_5M_esmc_300M` (seed 42). Clarify which split/seed each number comes from. Add a multi-seed summary table with mean ± std and bootstrap CIs. Fix MLP claim wording. | `bica_v2_paper.tex` §4.1-4.2 | 30 min |
| P0-2 | **Placeholder citations** — `[?]`, `[???]`, 8 `[FIXME]` in .bib | Fill all 8 author names. Web-search each paper. | `bica_v2_references.bib` | 20 min |
| P0-3 | **Sequence length handling** — batch 256 with full sequences seems impossible | Add paragraph to Methods: max L_prot (truncated at 512), max L_lig (truncated at 256), padding strategy. Report actual GPU memory and hardware. | `bica_v2_paper.tex` §3.3 | 15 min |
| P0-4 | **Hardware/timing details** — 45s training is unbelievable without context | Report GPU model, precision (fp32/fp16), actual memory footprint. Clarify what 45s includes (just the final epoch? full training?). Match the CSV `train_time_sec` values. | `bica_v2_paper.tex` §3.7 | 15 min |

#### P1 — STRONGLY RECOMMENDED (2/3 reviewers)

| # | Issue | Action | Effort |
|---|-------|--------|--------|
| P1-1 | **Only ligand-scaffold split** — no protein-disjoint or double-cold | Add at minimum one additional split: protein-family disjoint (cluster proteins by Pfam/UniProt family, ensure no train/test family overlap). If data allows, also report double-cold. Run for top-5 models only (not all 333). | 2-4 hours |
| P1-2 | **Frozen vs fine-tuned encoders** — unclear for ChemCross AND baselines | Add explicit statement: ESM-2 and ChemBERTa are frozen for all models. No encoder weights are updated during training. This ensures representation parity. | 5 min |
| P1-3 | **Value-weighting only at inference** — training/inference mismatch | Add ablation: report raw-attention vs value-weighted attention on fidelity delta and PDB precision/recall. Acknowledge mismatch as limitation in Discussion. If possible, add training-time variant (joint norm-regularization during training). | 1-2 hours |
| P1-4 | **PDB validation details** — mapping, isoform handling, n | Add paragraph: list the 20 kinase inhibitors used, explain sequence→PDB alignment protocol (BLAST, E-value threshold), state how isoforms and gaps handled. Report n and coverage. | 30 min |

#### P2 — RECOMMENDED (1/3 reviewers, or easy wins)

| # | Issue | Action | Effort |
|---|-------|--------|--------|
| P2-1 | **"First" claim** — overstated given FusionDTI-CAN etc. | Soften to "to our knowledge, the first bidirectional cross-attention model evaluated under scaffold split with full-sequence retention and fidelity-validated interpretability." Add paragraph comparing with FusionDTI/DrugBAN in Related Work. | 20 min |
| P2-2 | **FusionDTI/DrugBAN comparison** — reviewers want head-to-head | Add qualitative comparison in Related Work §2.3. If feasible, re-implement BAN-style fusion baseline under same split. If not, explain why split protocol is the key differentiator. | 1 hour |
| P2-3 | **MolXProt comparison** — acknowledge confounding | Add explicit note in §4.2 and §6.1: "this comparison is qualitative because datasets and splits differ; head-to-head would require retraining MolXProt on BindingDB under scaffold split, which we plan for future work." | 10 min |
| P2-4 | **Code release** — reviewers expect transparency | Add explicit commitment: "Code, trained checkpoints, and the append-only results diary will be released upon acceptance." Add link to anonymous repo if possible. | 5 min |
| P2-5 | **Consistency: experiment count** — 333 vs 324 | Clarify: 333 total logged, 324 unique after deduplication. Use consistent number and explain. | 5 min |

#### P3 — NICE TO HAVE

| # | Issue | Action | Effort |
|---|-------|--------|--------|
| P3-1 | **Calibration analysis** | Add reliability diagram (predicted vs true pKd binned) for ChemCross and best tree. | 1 hour |
| P3-2 | **Fidelity controls** — zeroing is crude | Add swap-based control: replace top-K residues with random non-binding residues. Report delta. | 30 min |
| P3-3 | **Statistical tests** for ESM-2 improvement claim | Add paired bootstrap test or paired t-test per family. | 30 min |

---

### Score Impact Estimate

| Fix Tier | Effort | Expected Score Gain |
|----------|--------|---------------------|
| P0 (4 items) | ~1.5 hrs | 5.2 → 6.0 (credibility restored) |
| P1 (4 items) | ~5 hrs | 6.0 → 6.8 (generalization claims solid) |
| P2 (5 items) | ~2 hrs | 6.8 → 7.2 (polish, reviewer satisfaction) |
| P3 (3 items) | ~2 hrs | 7.2 → 7.5 (bonus) |

**ICML was already weak accept.** With P0 fixed → solid accept. With P0+P1 → strong accept.

---

### Implementation Order

1. **P0** — all items, immediately. These are blocking.
2. **P1-1** — the protein-disjoint split. Single biggest impact on all three reviewers.
3. **P1-3** — value-weighting ablation. Addresses the most specific technical concern.
4. **P2** — cleanup and softening claims.
5. **P3** — if time/energy permits.

Start with P0. Ready when you are.



As discussed, here are details to access bullitt, AICoE Workstation:
ssh -X or
ssh -Y
to aicoe.snu.in 
username: rajeev
password: @@Gujrattt123


pip install torch-scatter torch-sparse --no-build-isolation -f  https://data.pyg.org/whl/torch-2.6.0+cu128.html
  pip install torch-geometric --no-build-isolation


  uv pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu128
  uv pip install   datasets scipy

  which python && python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" 
  2>&1
  pip list 2>/dev/null | grep -iE "torch|transf|rdkit"
  dpkg -l | grep -i "python3-torch\|libtorch" 2>/dev/null


uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


python -c "import torch; print(f'Torch {torch.__version__}');  print(f'CUDA: {torch.cuda.is_available()}');  print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else  'CPU ONLY')"


 pip download torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu128 -d  C:\Users\sps26\Desktop\torch_cuda


 pip download --python-version 3.12 --only-binary :all: --platform manylinux_2_28_x86_64 --no-deps torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cu128 -d C:\Users\sps26\Desktop\torch_cuda


  python -c "import torch; print(f'Torch {torch.__version__}'); print(f'CUDA:  {torch.cuda.is_available()}')"




 sed -i 's/from harness.data import load_splits/from harness.data import  get_splits/' ~/bica/run_value_weight_ablation.py


