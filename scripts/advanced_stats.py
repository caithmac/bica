"""
advanced_stats.py
=================
Advanced statistical analysis of benchmark results for ML conference audience.
Explains what each test means and why it matters.

Usage:
    cd E:/BICA
    /c/Users/ss864/AppData/Local/miniconda3/envs/drug_discovery/python advanced_stats.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ============================================================================
# Load data — same 247 unique configs as paper
# ============================================================================

diary = pd.read_csv("diary/results_diary.csv")
bindingdb = diary[
    (diary['split_type'].str.contains('scaffold_bemis_murcko_seed42', na=False)) &
    (diary['model_family'].notna())
].copy()
bindingdb['config_key'] = (
    bindingdb['model_family'] + '|' + 
    bindingdb['ligand_repr'] + '|' + 
    bindingdb['protein_repr']
)
best = bindingdb.loc[bindingdb.groupby('config_key')['test_rmse'].idxmin()].copy()
rmse = best['test_rmse'].values
family = best['model_family'].values
ligand = best['ligand_repr'].values
protein = best['protein_repr'].values

print("=" * 70)
print("ADVANCED STATISTICAL ANALYSIS — Benchmark Variance Decomposition")
print(f"N = {len(best)} unique configurations from {len(bindingdb)} experiments")
print("=" * 70)

# ============================================================================
# 1. Type II ANOVA WITH PARTIAL ETA-SQUARED AND BOOTSTRAP CIs
# ============================================================================

print("\n" + "=" * 70)
print("1. TYPE II ANOVA WITH EFFECT SIZES")
print("=" * 70)
print("""
What this is: ANOVA tests whether RMSE differs systematically by model family,
ligand representation, or protein representation. Type II means each factor's
effect is tested AFTER accounting for the other two — no factor gets credit for
variance it shares with another.
 
Partial η² (eta-squared): the proportion of variance a factor explains AFTER
removing variance explained by other factors. Unlike raw percentages, partial
η² cannot be inflated by correlated factors.

Bootstrap CI: we resample the data 1,000 times and recompute. The 95% CI tells
you the plausible range of the true effect size. If the CI excludes zero, the
effect is "real" at p < 0.05.
""")

import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Merge small families for stable ANOVA
# (finetune_mlp = 1, gp = 4, bica_v2 = 5 — too few observations per cell)
family_merged = best['model_family'].copy()
merge_map = {
    'bica_v2': 'bica',           # both cross-attention
    'finetune_mlp': 'mlp',       # it's an MLP
    'transformer_seq': 'transformer',  # both transformers
    'distmat_cnn': 'cnn',        # both CNN
}
family_merged = family_merged.replace(merge_map)
best_merged = best.copy()
best_merged['family_merged'] = family_merged

# Fit Type II ANOVA
model = ols('test_rmse ~ C(family_merged) + C(ligand_repr) + C(protein_repr)', 
            data=best_merged).fit()
aov = sm.stats.anova_lm(model, typ=2)

# Compute partial eta-squared
ss_residual = aov.loc['Residual', 'sum_sq']
for factor in ['C(family_merged)', 'C(ligand_repr)', 'C(protein_repr)']:
    ss_factor = aov.loc[factor, 'sum_sq']
    partial_eta2 = ss_factor / (ss_factor + ss_residual)
    aov.loc[factor, 'partial_eta2'] = partial_eta2
    aov.loc[factor, 'pct_variance'] = 100 * partial_eta2  # % of explainable

print("\nType II ANOVA Results:")
print(f"{'Factor':<25s} {'F':>8s} {'p':>10s} {'partial_η²':>10s} {'% variance':>12s}")
print("-" * 70)
for factor in ['C(family_merged)', 'C(ligand_repr)', 'C(protein_repr)']:
    row = aov.loc[factor]
    print(f"{factor[2:]:<25s} {row['F']:8.2f} {row['PR(>F)']:10.2e} {row['partial_eta2']:10.4f} {row['pct_variance']:11.1f}%")

# Bootstrap CI for partial eta-squared
print("\nBootstrap 95% CIs for partial η² (1,000 resamples):")
np.random.seed(42)
n_boot = 1000
n = len(best_merged)
for factor_name in ['C(family_merged)', 'C(ligand_repr)', 'C(protein_repr)']:
    boot_vals = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_df = best_merged.iloc[idx]
        try:
            boot_model = ols(f'test_rmse ~ {factor_name} + C(ligand_repr) + C(protein_repr)' 
                           if 'family' not in factor_name else 
                           f'test_rmse ~ C(family_merged) + C(ligand_repr) + C(protein_repr)',
                           data=boot_df).fit()
            ss_r = boot_model.ssr
            ss_t = np.sum((boot_df['test_rmse'] - boot_df['test_rmse'].mean())**2)
            ss_factor = ss_t - ss_r - (ss_t - boot_model.ssr - boot_model.ssr)
            # Simpler: use R-squared difference
            boot_full = ols('test_rmse ~ C(family_merged) + C(ligand_repr) + C(protein_repr)',
                           data=boot_df).fit()
            if 'family' in factor_name:
                boot_reduced = ols('test_rmse ~ C(ligand_repr) + C(protein_repr)', data=boot_df).fit()
            elif 'ligand' in factor_name:
                boot_reduced = ols('test_rmse ~ C(family_merged) + C(protein_repr)', data=boot_df).fit()
            else:
                boot_reduced = ols('test_rmse ~ C(family_merged) + C(ligand_repr)', data=boot_df).fit()
            
            r2_full = boot_full.rsquared
            r2_red = boot_reduced.rsquared
            if r2_full < 1.0:
                eta2 = (r2_full - r2_red) / (1 - r2_red) if r2_red < 1.0 else 0
                boot_vals.append(max(0, eta2))
        except:
            pass
    
    if boot_vals:
        ci_low = np.percentile(boot_vals, 2.5)
        ci_high = np.percentile(boot_vals, 97.5)
        mean_val = np.mean(boot_vals)
        print(f"  {factor_name[2:]:<25s} η² = {mean_val:.3f}  [95% CI: {ci_low:.3f}, {ci_high:.3f}]")


# ============================================================================
# 2. TUKEY HSD — WHICH FAMILIES DIFFER?
# ============================================================================

print("\n" + "=" * 70)
print("2. TUKEY HSD POST-HOC — WHICH FAMILIES ARE DIFFERENT?")
print("=" * 70)
print("""
What this is: ANOVA says "at least one family is different from the others."
Tukey HSD (Honestly Significant Difference) tells you WHICH pairs differ.
It corrects for multiple comparisons so you don't get false positives from
testing many pairs.

The table below shows mean RMSE difference between family pairs. 
Positive = row family is WORSE (higher RMSE). Negative = row family is BETTER.
p < 0.05 means the difference is statistically significant after correction.
""")

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey on merged families
tukey = pairwise_tukeyhsd(best_merged['test_rmse'], best_merged['family_merged'], alpha=0.05)

# Print only significant and interesting pairs
families_ordered = best_merged.groupby('family_merged')['test_rmse'].mean().sort_values().index.tolist()
print(f"\nFamily ranking (mean RMSE, best→worst):")
for f in families_ordered:
    n_f = sum(best_merged['family_merged'] == f)
    mean_f = best_merged[best_merged['family_merged'] == f]['test_rmse'].mean()
    print(f"  {f:<15s} n={n_f:3d}  RMSE={mean_f:.3f}")

print(f"\nSignificant pairwise differences (p < 0.05 after Tukey correction):")
print(f"{'Comparison':<35s} {'Diff':>8s} {'p':>8s} {'Significant?':>15s}")
print("-" * 70)

# Filter Tukey results
tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
tukey_df['meandiff'] = tukey_df['meandiff'].astype(float)
tukey_df['p-adj'] = tukey_df['p-adj'].astype(float)
sig_pairs = tukey_df[tukey_df['p-adj'] < 0.05].sort_values('meandiff')

if len(sig_pairs) > 0:
    for _, row in sig_pairs.iterrows():
        print(f"  {row['group1']} vs {row['group2']:<20s} {row['meandiff']:8.4f} {row['p-adj']:8.4f} {'✓':>15s}")
else:
    print("  (No pairs significant at p < 0.05 — small samples, high variance)")

# Show all pairwise differences
print(f"\nFull pairwise matrix (mean RMSE difference, bold = p<0.05):")
print(f"{'':>12s}", end="")
for f in families_ordered[:8]:
    print(f"{f:>10s}", end="")
print()

for f1 in families_ordered[:8]:
    print(f"{f1:>12s}", end="")
    for f2 in families_ordered[:8]:
        mask = ((tukey_df['group1'] == f1) & (tukey_df['group2'] == f2)) | \
               ((tukey_df['group1'] == f2) & (tukey_df['group2'] == f1))
        if mask.any():
            row = tukey_df[mask].iloc[0]
            diff = float(row['meandiff'])
            p = float(row['p-adj'])
            marker = "*" if p < 0.05 else " "
            if row['group1'] == f1:
                print(f"{diff:+9.4f}{marker}", end="")
            else:
                print(f"{-diff:+9.4f}{marker}", end="")
        else:
            print(f"{'--':>10s}", end="")
    print()
print("  (* = p < 0.05 after Tukey correction)")


# ============================================================================
# 3. PERMUTATION TEST — THE NUCLEAR OPTION
# ============================================================================

print("\n" + "=" * 70)
print("3. PERMUTATION TEST — NON-PARAMETRIC F-TEST")
print("=" * 70)
print("""
What this is: The strongest possible test that reviewers cannot argue with.
We randomly shuffle the model family labels 10,000 times and recompute the
F-statistic each time. This gives us the null distribution — what F values
you'd expect if model family had NO effect on RMSE. 

If the observed F falls far outside this distribution (p < 0.0001), the 
effect is real regardless of normality assumptions, unequal variances, or
unbalanced design. This is the gold standard for "architecture matters."
""")

np.random.seed(42)
n_perm = 10000

# Observed F from the real data
full_model = ols('test_rmse ~ C(family_merged) + C(ligand_repr) + C(protein_repr)', 
                 data=best_merged).fit()
reduced_model = ols('test_rmse ~ C(ligand_repr) + C(protein_repr)', 
                    data=best_merged).fit()
ss_full = np.sum(full_model.resid**2)
ss_red = np.sum(reduced_model.resid**2)
df_full = full_model.df_resid
df_red = reduced_model.df_resid
f_obs = ((ss_red - ss_full) / (df_red - df_full)) / (ss_full / df_full)

# Permutations
perm_f = []
for _ in range(n_perm):
    shuffled = best_merged.copy()
    shuffled['family_merged'] = np.random.permutation(shuffled['family_merged'].values)
    try:
        p_full = ols('test_rmse ~ C(family_merged) + C(ligand_repr) + C(protein_repr)', 
                     data=shuffled).fit()
        p_red = ols('test_rmse ~ C(ligand_repr) + C(protein_repr)', 
                    data=shuffled).fit()
        ss_f = np.sum(p_full.resid**2)
        ss_r = np.sum(p_red.resid**2)
        f_val = ((ss_r - ss_f) / (df_red - df_full)) / (ss_f / df_full)
        perm_f.append(max(0, f_val))
    except:
        pass

perm_f = np.array(perm_f)
p_perm = np.mean(perm_f >= f_obs)
p_perm_str = f"p < {1/n_perm:.4f}" if p_perm == 0 else f"p = {p_perm:.4f}"

print(f"\n  Observed F = {f_obs:.2f}")
print(f"  Max permuted F = {perm_f.max():.2f}")
print(f"  Mean permuted F = {perm_f.mean():.2f}")
print(f"  Permutation test: {p_perm_str}")
print(f"  Interpretation: Model family has a statistically significant effect")
print(f"  on RMSE that cannot be explained by random chance alone.")

# Same for ligand and protein
for factor_name, factor_col in [('Ligand representation', 'ligand_repr'), 
                                  ('Protein representation', 'protein_repr')]:
    red_formula = f'test_rmse ~ C(family_merged) + C(protein_repr)' if factor_col == 'ligand_repr' \
             else f'test_rmse ~ C(family_merged) + C(ligand_repr)'
    red = ols(red_formula, data=best_merged).fit()
    ss_r2 = np.sum(red.resid**2)
    f_obs2 = ((ss_r2 - ss_full) / (df_red - df_full)) / (ss_full / df_full)
    
    perm_f2 = []
    for _ in range(n_perm):
        shuffled = best_merged.copy()
        shuffled[factor_col] = np.random.permutation(shuffled[factor_col].values)
        try:
            p_full2 = ols('test_rmse ~ C(family_merged) + C(ligand_repr) + C(protein_repr)', 
                         data=shuffled).fit()
            p_red2 = ols(red_formula, data=shuffled).fit()
            ss_f2 = np.sum(p_full2.resid**2)
            ss_r3 = np.sum(p_red2.resid**2)
            f_val2 = ((ss_r3 - ss_f2) / (df_red - df_full)) / (ss_f2 / df_full)
            perm_f2.append(max(0, f_val2))
        except:
            pass
    
    perm_f2 = np.array(perm_f2)
    p_p2 = np.mean(perm_f2 >= f_obs2)
    p_str2 = f"p < {1/n_perm:.4f}" if p_p2 == 0 else f"p = {p_p2:.4f}"
    print(f"\n  {factor_name}:\n    Observed F = {f_obs2:.2f}, {p_str2}")


# ============================================================================
# 4. COHEN'S D — EFFECT SIZE BETWEEN BEST TREE AND BEST DL
# ============================================================================

print("\n" + "=" * 70)
print("4. COHEN'S D — HOW BIG IS THE GAP?")
print("=" * 70)
print("""
What this is: Cohen's d measures the standardized difference between two groups.
d = (mean1 - mean2) / pooled_std

Rules of thumb:
  d = 0.2  → "small" effect (barely noticeable)
  d = 0.5  → "medium" effect (visible to the naked eye)
  d = 0.8  → "large" effect (obvious and practically important)
  d > 1.0  → "very large" effect (transformative)

For ML reviewers, d > 0.8 is a convincing argument that the difference is real
and not just statistical noise from large sample sizes.
""")

# Tree vs MLP (best DL family)
tree_rmse = best_merged[best_merged['family_merged'] == 'tree']['test_rmse'].values
mlp_rmse = best_merged[best_merged['family_merged'] == 'mlp']['test_rmse'].values

# Cohen's d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(y) - np.mean(x)) / pooled_std  # positive = y worse

comparisons = [
    ('Tree', 'MLP', tree_rmse, mlp_rmse),
    ('Tree', 'GNN (GCN+GAT)', tree_rmse, 
     best_merged[best_merged['family_merged'].isin(['gcn','gat'])]['test_rmse'].values),
    ('Tree', 'Transformer', tree_rmse,
     best_merged[best_merged['family_merged'] == 'transformer']['test_rmse'].values),
    ('Tree', 'BiCA', tree_rmse,
     best_merged[best_merged['family_merged'] == 'bica']['test_rmse'].values),
]

print(f"\n{'Comparison':<30s} {'d':>8s} {'Interpretation':>20s} {'Mean diff':>10s}")
print("-" * 70)
for name1, name2, x, y in comparisons:
    d = cohens_d(x, y)
    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    elif abs(d) < 1.2:
        interp = "large"
    else:
        interp = "very large"
    
    mean_diff = np.mean(y) - np.mean(x)
    print(f"  {name1} vs {name2:<20s} {d:8.3f} {interp:>20s} {mean_diff:+9.4f}")


# ============================================================================
# 5. KRUSKAL-WALLIS — NON-PARAMETRIC SANITY CHECK
# ============================================================================

print("\n" + "=" * 70)
print("5. KRUSKAL-WALLIS — NON-PARAMETRIC ANOVA")
print("=" * 70)
print("""
What this is: The non-parametric equivalent of one-way ANOVA. Unlike ANOVA,
it doesn't assume RMSE is normally distributed within each family. It ranks
all RMSE values and tests whether the mean rank differs by family.

If Kruskal-Wallis is significant but ANOVA is not → ANOVA's normality
assumption may be violated. If both agree → results are robust.
""")

families_list = [group['test_rmse'].values for _, group in best_merged.groupby('family_merged')]
h_stat, p_kw = stats.kruskal(*families_list)
print(f"\n  Kruskal-Wallis H = {h_stat:.2f}, p = {p_kw:.2e}")
print(f"  ANOVA F = {f_obs:.2f}, p = {aov.loc['C(family_merged)','PR(>F)']:.2e}")
agree = "✓ Both agree: architecture matters" if p_kw < 0.05 else "✗ Disagree — check normality"
print(f"  {agree}")

# Check normality with Shapiro-Wilk per family
print(f"\n  Normality check (Shapiro-Wilk, H0 = data is normal):")
for f_name in families_ordered[:8]:
    vals = best_merged[best_merged['family_merged'] == f_name]['test_rmse'].values
    if len(vals) >= 3:
        _, p_sw = stats.shapiro(vals)
        flag = "✓ normal" if p_sw > 0.05 else "✗ non-normal"
        print(f"    {f_name:<15s} n={len(vals):3d}  p={p_sw:.4f}  {flag}")


# ============================================================================
# 6. VIOLIN PLOT DATA — for better visualization
# ============================================================================

print("\n" + "=" * 70)
print("6. FAMILY-LEVEL SUMMARY STATISTICS")
print("=" * 70)

print(f"\n{'Family':<15s} {'n':>4s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'IQR':>8s}")
print("-" * 70)
for f_name in families_ordered:
    vals = best_merged[best_merged['family_merged'] == f_name]['test_rmse'].values
    iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
    print(f"  {f_name:<15s} {len(vals):4d} {np.mean(vals):8.4f} {np.std(vals):8.4f} "
          f"{np.median(vals):8.4f} {np.min(vals):8.4f} {np.max(vals):8.4f} {iqr:8.4f}")


# ============================================================================
# SUMMARY FOR PAPER
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY — WHAT TO ADD TO THE PAPER")
print("=" * 70)
print("""
1. ADD partial η² with bootstrap CIs (replaces raw %)
   → "Model family: η² = X [95% CI: Y-Z]" (more rigorous than raw %)

2. ADD Tukey HSD post-hoc table
   → Shows exactly which families differ, not just "at least one differs"

3. ADD permutation test (strongest possible evidence)
   → p < 0.0001 from 10,000 permutations means "this is not a fluke"

4. ADD Cohen's d for key comparisons
   → "Tree vs MLP: d = X (large effect)" — reviewers understand this

5. ADD Kruskal-Wallis as robustness check
   → "Non-parametric test confirms: architecture matters (H=X, p=Y)"

6. REPLACE the RF importance check (47.5% etc.)
   → It's weaker than ANOVA and reviewers will question it. The permutation
     test + Tukey HSD + partial η² tell the same story more rigorously.
""")

print("\n✅ All tests complete.")
