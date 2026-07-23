#!/usr/bin/env python3
"""Phase 8+9: Virtual-screening enrichment + robustness analyses.
Phase 8: Rank compounds by predicted pKd, enrichment factor at top 1%/5%/10%.
Phase 9: Performance by target family, affinity range, error analysis.
"""
import json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, bootstrap

warnings.filterwarnings('ignore')

sys.path.insert(0, "E:/Drug Discovery")
from harness.featurizers import ecfp, amino_acid_composition, concat

SEEDS = [42, 123, 456]
HIGH_AFFINITY_THRESHOLD = 7.0  # pKd >= 7
OUT_DIR = Path("E:/Drug Discovery/projects/balm-revision/results")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load scaffold split ---
print("Loading scaffold split (seed 42)...")
split_dir = Path("E:/Drug Discovery/projects/balm-revision/data/splits/scaffold/seed_42")
train_df = pd.read_csv(split_dir / "train.csv")
test_df = pd.read_csv(split_dir / "test.csv")

X_train = concat(ecfp(train_df['Drug_canonical'].tolist()),
                 amino_acid_composition(train_df['Target'].tolist()))
X_test = concat(ecfp(test_df['Drug_canonical'].tolist()),
                amino_acid_composition(test_df['Target'].tolist()))
y_train = train_df['Y'].values
y_test = test_df['Y'].values

# ============================================================================
# Phase 8: Virtual Screening Enrichment
# ============================================================================
print("\n=== Phase 8: Virtual Screening ===")

rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
yp_test = rf.predict(X_test)

# Define high-affinity
true_high = y_test >= HIGH_AFFINITY_THRESHOLD
n_true_high = true_high.sum()
print(f"  Test set: {len(y_test)} compounds, {n_true_high} high-affinity (pKd≥{HIGH_AFFINITY_THRESHOLD})")

# Rank by predicted pKd
ranked_idx = np.argsort(yp_test)[::-1]

enrichment_results = {}
for pct in [1, 5, 10]:
    n_top = int(len(yp_test) * pct / 100)
    top_idx = ranked_idx[:n_top]
    recovered = true_high[top_idx].sum()
    
    # Enrichment factor: (recovered / n_top) / (n_true_high / total)
    baseline_rate = n_true_high / len(yp_test)
    ef = (recovered / n_top) / baseline_rate if baseline_rate > 0 else 0
    
    # Bootstrap CI
    def ef_statistic(indices):
        subset = true_high[indices[:n_top]]
        rate = subset.sum() / n_top
        return rate / baseline_rate if baseline_rate > 0 else 0
    
    try:
        boot = bootstrap(
            (np.arange(len(yp_test)),), ef_statistic,
            n_resamples=1000, random_state=42, method='percentile'
        )
        ef_ci_low = float(boot.confidence_interval.low)
        ef_ci_high = float(boot.confidence_interval.high)
    except:
        ef_ci_low, ef_ci_high = None, None
    
    enrichment_results[f"top_{pct}pct"] = {
        'n_top': n_top,
        'n_recovered': int(recovered),
        'precision': round(float(recovered / n_top), 3),
        'recall': round(float(recovered / n_true_high), 3),
        'enrichment_factor': round(float(ef), 2),
        'ef_95ci_low': round(ef_ci_low, 2) if ef_ci_low else None,
        'ef_95ci_high': round(ef_ci_high, 2) if ef_ci_high else None,
    }
    
    print(f"  Top {pct}%: recovered {recovered}/{n_true_high}, EF={ef:.1f}x, "
          f"precision={recovered/n_top:.3f}")

# Plot enrichment curve
plt.figure(figsize=(8, 5))
pcts = np.linspace(0.5, 50, 100)
efs, recalls = [], []
for pct in pcts:
    n = int(len(yp_test) * pct / 100)
    rec = true_high[ranked_idx[:n]].sum()
    efs.append((rec/n) / (n_true_high/len(yp_test)) if n_true_high > 0 else 0)
    recalls.append(rec / n_true_high)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(pcts, efs, 'b-', linewidth=2)
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Top % screened'); ax1.set_ylabel('Enrichment Factor')
ax1.set_title('RF Enrichment (pKd ≥ 7)')

ax2.plot(pcts, recalls, 'g-', linewidth=2)
ax2.set_xlabel('Top % screened'); ax2.set_ylabel('Recall')
ax2.set_title('Recall of High-Affinity Compounds')
ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "phase8_enrichment.png", dpi=150)
plt.close()

# ============================================================================
# Phase 9: Robustness Analyses
# ============================================================================
print("\n=== Phase 9: Robustness ===")

# 9a: Performance by affinity range
affinity_bins = [(-100, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 100)]
affinity_results = {}

for low, high in affinity_bins:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() < 5:
        continue
    rmse = np.sqrt(mean_squared_error(y_test[mask], yp_test[mask]))
    mae = mean_absolute_error(y_test[mask], yp_test[mask])
    affinity_results[f"pKd_{low}_{high}"] = {
        'n': int(mask.sum()),
        'rmse': round(float(rmse), 3),
        'mae': round(float(mae), 3),
    }
    print(f"  pKd [{low}, {high}): n={mask.sum()}, RMSE={rmse:.3f}")

# 9b: Error analysis — top disagreements
errors = np.abs(y_test - yp_test)
worst_idx = np.argsort(errors)[-20:][::-1]

print(f"\n  Top 10 worst predictions:")
for i, idx in enumerate(worst_idx[:10]):
    smi = test_df.iloc[idx]['Drug_canonical'][:30]
    target = test_df.iloc[idx]['Target'][:30]
    print(f"    {i+1}. {smi}... | {target}... | true={y_test[idx]:.2f} pred={yp_test[idx]:.2f} err={errors[idx]:.2f}")

# 9c: Performance by compound novelty (scaffold frequency in train)
train_scaffolds = train_df['scaffold'].value_counts().to_dict() if 'scaffold' in train_df.columns else {}
if train_scaffolds:
    test_df['scaffold_freq_in_train'] = test_df['scaffold'].map(lambda s: train_scaffolds.get(s, 0))
    
    novel_mask = test_df['scaffold_freq_in_train'] <= 5
    familiar_mask = test_df['scaffold_freq_in_train'] > 50
    
    if novel_mask.sum() > 5:
        novel_rmse = np.sqrt(mean_squared_error(y_test[novel_mask], yp_test[novel_mask]))
        print(f"\n  Novel scaffolds (≤5 in train): n={novel_mask.sum()}, RMSE={novel_rmse:.3f}")
    
    if familiar_mask.sum() > 5:
        familiar_rmse = np.sqrt(mean_squared_error(y_test[familiar_mask], yp_test[familiar_mask]))
        print(f"  Familiar scaffolds (>50 in train): n={familiar_mask.sum()}, RMSE={familiar_rmse:.3f}")

# Save
summary = {
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'enrichment': enrichment_results,
    'affinity_bins': affinity_results,
    'n_test': len(y_test),
    'n_high_affinity': int(n_true_high),
    'overall_test_rmse': round(float(np.sqrt(mean_squared_error(y_test, yp_test))), 4),
    'overall_test_pearson': round(float(pearsonr(y_test, yp_test)[0]), 4),
}

with open(OUT_DIR / "phase8_9_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[Phase 8+9] COMPLETE — {OUT_DIR}/phase8_9_summary.json")
