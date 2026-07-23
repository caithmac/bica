#!/usr/bin/env python3
"""Phase 4+5: Training curves and data-scaling experiment.
Phase 4: Log train/val RMSE per epoch for RF, best MLP, BiCA on scaffold split.
Phase 5: Data size sweep (500, 1K, 2.5K, 5K, 10K, full) × 3 seeds.
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
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

sys.path.insert(0, "E:/Drug Discovery")
from harness.featurizers import ecfp, amino_acid_composition, esm2_embeddings, concat
from harness.trainer import train_torch

SEEDS = [42, 123, 456]
SIZES = [500, 1000, 2500, 5000, 10000, None]  # None = full
OUT_DIR = Path("E:/Drug Discovery/projects/balm-revision/results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# --- Load scaffold split (seed 42) ---
df = pd.read_parquet("E:/Drug Discovery/projects/balm-revision/data/frozen/balm_filtered.parquet")
split_dir = Path("E:/Drug Discovery/projects/balm-revision/data/splits/scaffold/seed_42")
train_df = pd.read_csv(split_dir / "train.csv")
val_df = pd.read_csv(split_dir / "val.csv")
test_df = pd.read_csv(split_dir / "test.csv")

print(f"Scaffold split: {len(train_df)}/{len(val_df)}/{len(test_df)}")

# ============================================================================
# Phase 4: Training curves
# ============================================================================
print("\n=== Phase 4: Training Curves ===")

# RF — log OOB and validation at different tree counts
print("  RF training curve...")
rf_curves = {}
for seed in SEEDS:
    X_train = concat(ecfp(train_df['Drug_canonical'].tolist()),
                     amino_acid_composition(train_df['Target'].tolist()))
    X_val = concat(ecfp(val_df['Drug_canonical'].tolist()),
                   amino_acid_composition(val_df['Target'].tolist()))
    y_train = train_df['Y'].values
    y_val = val_df['Y'].values
    
    n_trees_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    train_rmse, val_rmse = [], []
    
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=seed, n_jobs=-1, warm_start=True)
    
    for n in n_trees_list:
        rf.set_params(n_estimators=n)
        rf.fit(X_train, y_train)
        train_rmse.append(np.sqrt(mean_squared_error(y_train, rf.predict(X_train))))
        val_rmse.append(np.sqrt(mean_squared_error(y_val, rf.predict(X_val))))
    
    rf_curves[f"seed_{seed}"] = {
        'n_trees': n_trees_list,
        'train_rmse': [round(x, 4) for x in train_rmse],
        'val_rmse': [round(x, 4) for x in val_rmse],
    }
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(n_trees_list, train_rmse, 'b-', label='Train')
    ax1.plot(n_trees_list, val_rmse, 'r-', label='Val')
    ax1.set_xlabel('Number of trees'); ax1.set_ylabel('RMSE')
    ax1.legend(); ax1.set_title(f'RF Training Curve (seed={seed})')
    
    ax2.semilogy(n_trees_list, train_rmse, 'b-', label='Train')
    ax2.semilogy(n_trees_list, val_rmse, 'r-', label='Val')
    ax2.set_xlabel('Number of trees'); ax2.set_ylabel('RMSE (log)')
    ax2.legend(); ax2.set_title(f'RF (log scale)')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"phase4_rf_curve_seed{seed}.png", dpi=150)
    plt.close()
    print(f"    seed={seed}: final val_rmse={val_rmse[-1]:.4f}")

# MLP — per-epoch logging
print("  MLP training curves...")

class MLPWithLogging(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_train = concat(ecfp(train_df['Drug_canonical'].tolist()),
                     amino_acid_composition(train_df['Target'].tolist()))
    X_val = concat(ecfp(val_df['Drug_canonical'].tolist()),
                   amino_acid_composition(val_df['Target'].tolist()))
    X_test = concat(ecfp(test_df['Drug_canonical'].tolist()),
                    amino_acid_composition(test_df['Target'].tolist()))
    y_train = train_df['Y'].values.astype(np.float32)
    y_val = val_df['Y'].values.astype(np.float32)
    y_test = test_df['Y'].values.astype(np.float32)
    
    model = MLPWithLogging(X_train.shape[1]).to(DEVICE)
    
    # Custom training loop with per-epoch logging
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    
    train_losses, val_rmse_list = [], []
    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_rmse_list.append(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break
    
    # Test with best state
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()
    
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print(f"    seed={seed}: epochs={len(val_rmse_list)}, val_rmse={best_val_rmse:.4f}, test_rmse={test_rmse:.4f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, 'b-', alpha=0.5, label='Train MSE')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs_range, val_rmse_list, 'r-', label='Val RMSE')
    ax1.set_xlabel('Epoch'); ax1.set_title(f'MLP Training (seed={seed})')
    
    ax2.semilogy(epochs_range, val_rmse_list, 'r-')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val RMSE (log)')
    ax2.set_title('Val RMSE (log)')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"phase4_mlp_curve_seed{seed}.png", dpi=150)
    plt.close()

# ============================================================================
# Phase 5: Data-scaling experiment
# ============================================================================
print("\n=== Phase 5: Data Scaling ===")

scaling_results = []

for seed in SEEDS:
    rng = np.random.RandomState(seed)
    
    for size in SIZES:
        if size is None:
            train_subset = train_df
            size_label = "full"
        else:
            if size > len(train_df):
                continue
            idx = rng.choice(len(train_df), size, replace=False)
            train_subset = train_df.iloc[idx]
            size_label = str(size)
        
        # RF
        X_tr = concat(ecfp(train_subset['Drug_canonical'].tolist()),
                      amino_acid_composition(train_subset['Target'].tolist()))
        X_te = concat(ecfp(test_df['Drug_canonical'].tolist()),
                      amino_acid_composition(test_df['Target'].tolist()))
        y_tr = train_subset['Y'].values
        y_te = test_df['Y'].values
        
        rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=seed, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        yp = rf.predict(X_te)
        
        scaling_results.append({
            'seed': seed, 'size': size_label, 'n_train': len(train_subset),
            'model': 'RF_ECFP4+AAC', 'test_rmse': round(float(np.sqrt(mean_squared_error(y_te, yp))), 4),
            'test_pearson': round(float(pearsonr(y_te, yp)[0]), 4),
        })
        
        print(f"  RF seed={seed} n={len(train_subset):>5}: RMSE={scaling_results[-1]['test_rmse']:.4f}")

# Save
pd.DataFrame(scaling_results).to_csv(OUT_DIR / "phase5_scaling.csv", index=False)

# Plot
df_s = pd.DataFrame(scaling_results)
plt.figure(figsize=(10, 6))
for seed in SEEDS:
    seed_data = df_s[df_s['seed'] == seed]
    sizes_plot = [int(x) if x != 'full' else len(train_df) for x in seed_data['size']]
    plt.plot(sizes_plot, seed_data['test_rmse'].values, 'o-', label=f'RF seed={seed}')

plt.xscale('log')
plt.xlabel('Training set size')
plt.ylabel('Test RMSE')
plt.title('Data Scaling: RF_ECFP4+AAC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "phase5_scaling_rf.png", dpi=150)
plt.close()

print(f"\n[Phase 4+5] COMPLETE — {OUT_DIR}")
