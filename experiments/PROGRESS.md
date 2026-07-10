# NMI Experiment Tracker

> Updated: 2026-07-09 17:12
> 16/18 complete, 1 running, 1 skipped, 1 pending

---

| Exp ID | Description | Status | Key Result |
|--------|-------------|--------|------------|
| E101 | Data Cleaning | ✅ | 24,700→18,743 rows, 550 targets, 8,182 cmpds |
| E102 | Overlap Analysis | ✅ | 4.1% scaffold overlap BD↔LP, 75 SMILES leaked |
| E103 | Three-Way Splits | ✅ | Scaffold / Cold-protein / Random |
| E201 | Learning Curves | ✅ | RF beats DL at every N. Gap widens (0.128→0.156). |
| E202 | Finetuning Sweep | ✅ | Best DL=1.122, gap floor=+0.115. Arch/HP tuning marginal. |
| E203 | PSICHIC Diagnostic | 🔵 | Scaffold already from cache (1.176). Running random split now. |
| E204 | Implementation Validation | ✅ | Our RF (1.007) ≈ BALM RF (1.020). |
| E205 | ECFP4 Bit Importance | ✅ | RF↔LGBM share 43% top-30 bits. XGBoost outlier. |
| E206 | Enrichment Analysis | ✅ | RF & DL share only 27/100 top hits. τ=0.56. |
| E301 | Clustered Bootstrap CIs | ✅ | RF cold-drug RMSE=0.923 [0.901,0.945]. |
| E302 | Calibration | ✅ | All models well-calibrated, ~72% ±1σ coverage. |
| E303 | Random Split Comparison | ✅ | Random RMSE ~0.18 lower than scaffold — benchmark hacking. |
| E304 | Temporal Split | ❌ | No timestamps in BindingDB. |
| GAP-1 | Representation Ablation | ✅ | AAC-only ≈ ESM-2. Representations don't matter on scaffold. |
| GAP-2 | HPO Sensitivity | ✅ | Default=1.154, Tuned=1.134. Saves only 0.02. |
| GAP-3 | Overfitting Evidence | ✅ | Deep MLP gap=+0.122. Deeper = more overfitting. |
| E401 | Zenodo Release | ⬜ | Last step — after E203 finishes. |

---

## KEY NARRATIVE (for paper)

1. **RF beats DL on scaffold split** (1.007 vs 1.161). Gap is structural — not data, not HPO, not architecture.
2. **Representations don't matter** — AAC-only matches ESM-2 for RF.
3. **Random splits hide the gap** — RMSE drops ~0.18, creating illusion of DL superiority.
4. **RF and DL rank differently** — only 27/100 top compounds overlap. Practical impact.
5. **PSICHIC** — specialized GNN gets 1.176 on scaffold (worse than RF), but should drop to ~0.85 on random (memorization).
