#!/usr/bin/env python3
"""
Phase 4: Learning curves on log-log scale + power law extrapolation.

Runs XGBoost and RF (ECFP4+AAC) on random and scaffold splits
across log-spaced subset sizes. Fits RMSE = a * N^(-b) and 
extrapolates to predict crossover points for a hypothetical deep model.
"""
import csv, os, sys, time, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, linregress
import xgboost as xgb

warnings.filterwarnings('ignore')

sys.path.insert(0, "E:/Drug Discovery")
from harness.featurizers import concat

# ── Config ──
SEEDS = [42, 123, 456]
SPLIT_TYPES = ['random', 'scaffold']  # most informative splits
SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, None]  # None = full

SPLIT_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/splits_bindingdb")
CACHE_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/cache_bindingdb")
OUT_DIR = Path("E:/Drug Discovery/projects/balm-revision/results")
RESULTS_CSV = OUT_DIR / "phase4_learning_curves.csv"
DEVICE = "cpu"  # GPU hangs on Windows — use CPU

print(f"XGBoost device: {DEVICE}")

# ── Load caches ──
print("Loading feature caches...")
with open(CACHE_DIR / "ecfp_cache.pkl", 'rb') as f:
    ecfp_dict = pickle.load(f)
with open(CACHE_DIR / "aac_cache.pkl", 'rb') as f:
    aac_dict = pickle.load(f)
print(f"  ECFP: {len(ecfp_dict):,} molecules, AAC: {len(aac_dict):,} sequences")

def build_features(df):
    ecfp_rows = np.stack([ecfp_dict[s] for s in df['Drug_canonical']])
    aac_rows = np.stack([aac_dict[s] for s in df['Target']])
    return concat(ecfp_rows, aac_rows)

def fit_xgb(Xtr, ytr, Xte, yte, seed):
    m = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', device=DEVICE, random_state=seed, verbosity=0
    )
    t0 = time.time()
    m.fit(Xtr, ytr)
    t = time.time() - t0
    yp = m.predict(Xte)
    return {
        'test_rmse': float(np.sqrt(mean_squared_error(yte, yp))),
        'test_pearson': float(pearsonr(yte, yp)[0]),
        'train_time_s': round(t, 1),
    }

def fit_rf(Xtr, ytr, Xte, yte, seed):
    m = RandomForestRegressor(
        n_estimators=500, max_depth=20, random_state=seed, n_jobs=-1
    )
    t0 = time.time()
    m.fit(Xtr, ytr)
    t = time.time() - t0
    yp = m.predict(Xte)
    return {
        'test_rmse': float(np.sqrt(mean_squared_error(yte, yp))),
        'test_pearson': float(pearsonr(yte, yp)[0]),
        'train_time_s': round(t, 1),
    }

MODELS = [
    ('XGBoost_ECFP4+AAC', fit_xgb),
    ('RF_ECFP4+AAC', fit_rf),
]

# ── Run ──
FIELDS = ['model', 'split_type', 'seed', 'size_label', 'n_train',
          'test_rmse', 'test_pearson', 'train_time_s']

results = []
total = len(SPLIT_TYPES) * len(SEEDS) * len(SIZES) * len(MODELS)
cnt = 0

for st in SPLIT_TYPES:
    for sd in SEEDS:
        sdir = SPLIT_DIR / st / f"seed_{sd}"
        tr = pd.read_csv(sdir / "train.csv")
        te = pd.read_csv(sdir / "test.csv")

        # Full features for test set (same for all sizes)
        Xte = build_features(te)
        yte = te['Y'].values.astype(np.float32)

        # Full train features (will subset)
        Xtr_full = build_features(tr)
        ytr_full = tr['Y'].values.astype(np.float32)
        n_full = len(tr)

        rng = np.random.RandomState(sd)

        for size in SIZES:
            if size is None or size >= n_full:
                size_label = 'full'
                n_actual = n_full
                Xtr_sub = Xtr_full
                ytr_sub = ytr_full
            else:
                size_label = str(size)
                n_actual = size
                idx = rng.choice(n_full, size, replace=False)
                Xtr_sub = Xtr_full[idx]
                ytr_sub = ytr_full[idx]

            for model_name, fit_fn in MODELS:
                cnt += 1
                print(f"  [{cnt}/{total}] {model_name} | {st} | seed={sd} | n={n_actual}", flush=True)
                try:
                    r = fit_fn(Xtr_sub, ytr_sub, Xte, yte, sd)
                    results.append({
                        'model': model_name,
                        'split_type': st,
                        'seed': sd,
                        'size_label': size_label,
                        'n_train': n_actual,
                        **r
                    })
                    print(f"    RMSE={r['test_rmse']:.4f} Pearson={r['test_pearson']:.4f} ({r['train_time_s']:.0f}s)", flush=True)
                except Exception as e:
                    print(f"    FAILED: {e}", flush=True)

# Save results
df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)
print(f"\nResults saved: {RESULTS_CSV} ({len(df)} rows)")

# ── Plotting ──
print("\n=== Plotting ===")

COLORS = {'XGBoost_ECFP4+AAC': '#2ecc71', 'RF_ECFP4+AAC': '#e74c3c'}
MARKERS = {'random': 'o', 'scaffold': 's'}
SPLIT_PRETTY = {'random': 'Random', 'scaffold': 'Scaffold'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (st, ax) in enumerate(zip(SPLIT_TYPES, axes)):
    st_df = df[df['split_type'] == st]

    for model_name in ['XGBoost_ECFP4+AAC', 'RF_ECFP4+AAC']:
        mdf = st_df[st_df['model'] == model_name]

        # Aggregate across seeds: mean ± std per training size
        size_groups = mdf.groupby('n_train')
        sizes = size_groups['test_rmse'].mean().index.values
        means = size_groups['test_rmse'].mean().values
        stds = size_groups['test_rmse'].std().values

        color = COLORS[model_name]
        marker = MARKERS[st]

        ax.errorbar(sizes, means, yerr=stds,
                    fmt=marker + '-', color=color, capsize=3,
                    label=model_name, markersize=6)

        # Fit power law: log10(RMSE) = log10(a) - b * log10(N)
        log_n = np.log10(sizes)
        log_rmse = np.log10(means)
        slope, intercept, r_val, p_val, std_err = linregress(log_n, log_rmse)

        b = -slope  # exponent: RMSE ∝ N^(-b)
        a = 10 ** intercept

        # Extrapolate line
        n_ext = np.logspace(np.log10(min(sizes)), np.log10(max(sizes) * 10), 100)
        rmse_ext = a * n_ext ** (-b)
        ax.plot(n_ext, rmse_ext, '--', color=color, alpha=0.4,
                label=f'{model_name} fit: RMSE={a:.2f}·N^(-{b:.3f}), R²={r_val**2:.3f}')

        # Print power law
        print(f"\n  {model_name} | {st}: RMSE = {a:.3f} · N^(-{b:.4f})  (R²={r_val**2:.4f})")
        # Crossover predictions
        for target_alpha in [0.15, 0.20, 0.30, 0.40]:
            if target_alpha > b:
                # RMSE_target = a * N_cross^(-b) = a_dl * N_cross^(-alpha)
                # a * N_cross^(-b) = RMSE_target
                # We want: a_trees * N^(-b) = RMSE_target
                # For a deep model: a_dl * N^(-alpha) = RMSE_target
                # Crossover where a_trees * N^(-b) = a_dl * N^(-alpha)
                # a_trees / a_dl = N^(b-alpha) → N = (a_trees/a_dl)^(1/(b-alpha))
                # Without a_dl, we can't compute. Instead: "at N=∞, if DL has alpha > b, it wins"
                # More useful: at current dataset sizes what RMSE would DL need?

                # If DL has same intercept a as trees but different slope:
                # crossover N when a*N^(-b) = a*N^(-alpha) → always at N=1 (trivial)
                # More meaningfully: at N=n_full, trees give RMSE_t. If DL has exponent α,
                # what intercept a_dl does it need to match trees at N=n_full?
                rmse_at_full = a * (n_full ** (-b))
                a_dl_needed = rmse_at_full * (n_full ** target_alpha)
                print(f"    To match trees at N={n_full:,} with α={target_alpha:.2f}: need a_dl ≤ {a_dl_needed:.3f}")
                # Crossover N if a_dl = a (same intercept):
                # a * N^(-b) = a * N^(-alpha) → N = 1. So not useful.
                # Better: if a_dl = a_trees * 0.8 (20% better intercept):
                for a_dl_factor in [1.0, 0.8, 0.5]:
                    a_dl = a * a_dl_factor
                    if target_alpha > b:
                        n_cross = (a / a_dl) ** (1.0 / (target_alpha - b))
                        if 100 < n_cross < 1e7:
                            print(f"    DL(α={target_alpha:.2f}, a_dl={a_dl:.3f}={a_dl_factor}×): crossover at N ≈ {n_cross:,.0f}")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training set size N')
    ax.set_ylabel('Test RMSE')
    ax.set_title(f'{SPLIT_PRETTY[st]} Split — Learning Curves')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(80, 2 * n_full)

plt.tight_layout()
plt.savefig(OUT_DIR / "phase4_learning_curves.png", dpi=150)
plt.savefig(OUT_DIR / "phase4_learning_curves.pdf")
plt.close()
print(f"\nPlot saved: {OUT_DIR / 'phase4_learning_curves.png'}")

# ── Summary table ──
print("\n=== Power Law Summary ===")
print(f"{'Model':<22} {'Split':<10} {'a':>8} {'b':>8} {'R²':>8} {'RMSE@full':>10}")
for st in SPLIT_TYPES:
    for model_name in ['XGBoost_ECFP4+AAC', 'RF_ECFP4+AAC']:
        mdf = df[(df['split_type'] == st) & (df['model'] == model_name)]
        size_groups = mdf.groupby('n_train')
        sizes = size_groups['test_rmse'].mean().index.values
        means = size_groups['test_rmse'].mean().values
        log_n = np.log10(sizes)
        log_rmse = np.log10(means)
        slope, intercept, r_val, p_val, std_err = linregress(log_n, log_rmse)
        a = 10**intercept
        b = -slope
        rmse_full = a * (n_full ** (-b))
        print(f"{model_name:<22} {st:<10} {a:8.3f} {b:8.4f} {r_val**2:8.4f} {rmse_full:10.4f}")

print("\n[DONE]")
