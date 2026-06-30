## Revision Plan — Benchmark Paper v2 (NMI/NeurIPS/IJCAI reviews)

### P0 — Text fixes (now)
- [x] Fix experiment count: 324 unique, 335 logged with reruns. Use consistent number.
- [ ] Add Hyperparameter appendix (grid per family)
- [ ] Clarify protein split: UniProt prefix → note limitation, add MMseqs2 plan
- [ ] Add controlled within-architecture ESM-2 ablation (XGBoost AAC vs ESM-2, matched ligand)
- [ ] Soften cross-attention claims: "on flat embeddings" qualifier
- [ ] Report bootstrap CIs for all family representatives
- [ ] Clarify variance decomposition methodology limitations

### P1 — Experiments (bullitt)
- [ ] Run XGBoost + ECFP4 + ESM-2 8M (controlled ablation)
- [ ] Run MLP + ECFP4 + ESM-2 35M (fill 2x2 table)
- [ ] Run ChemCross seq on protein split (if checkpoints available)
- [ ] Fine-tuned ESM-2 last-n layers baseline (1 experiment)

### P2 — Nice to have
- [ ] Cross-dataset: BindingDB → LeakyPDB transfer
- [ ] Learning curves (subsampled training sizes)
- [ ] MMseqs2 protein-family split (replace UniProt prefix)
- [ ] Carbon/energy estimates per family
