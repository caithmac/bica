#!/usr/bin/env python3
"""Additional baselines: XGBoost, KNN, Ridge — ECFP4+AAC features, all splits × 3 seeds."""
import csv, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

warnings.filterwarnings('ignore')
sys.path.insert(0, "E:/Drug Discovery")
from harness.featurizers import ecfp, amino_acid_composition, concat

SEEDS = [42, 123, 456]
SPLIT_TYPES = ['random', 'scaffold', 'cold_target', 'seq_clustered/30pct', 'seq_clustered/40pct']
SPLIT_DIR = Path("E:/Drug Discovery/projects/balm-revision/data/splits")
RESULTS_CSV = "E:/Drug Discovery/projects/balm-revision/phase3_results.csv"

def evaluate(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'pearson': float(pearsonr(y_true, y_pred)[0]),
        'spearman': float(spearmanr(y_true, y_pred)[0]),
    }

FIELD_NAMES = ['timestamp', 'model', 'model_family', 'split_type', 'seed',
               'n_train', 'n_val', 'n_test',
               'val_rmse', 'val_mae', 'val_pearson',
               'test_rmse', 'test_mae', 'test_pearson', 'test_spearman',
               'n_params', 'train_time_s', 'epochs_trained', 'device']

existing = set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            existing.add((row['model'], row['split_type'], row['seed']))

NEW_MODELS = ['XGBoost_ECFP4+AAC', 'KNN_ECFP4+AAC', 'Ridge_ECFP4+AAC']

with open(RESULTS_CSV, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)

    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            sdir = SPLIT_DIR / split_type / f"seed_{seed}"
            train_df = pd.read_csv(sdir / "train.csv")
            val_df = pd.read_csv(sdir / "val.csv")
            test_df = pd.read_csv(sdir / "test.csv")
            n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
            y_train = train_df['Y'].values.astype(np.float32)
            y_val = val_df['Y'].values.astype(np.float32)
            y_test = test_df['Y'].values.astype(np.float32)

            X_train = concat(ecfp(train_df['Drug_canonical'].tolist()),
                            amino_acid_composition(train_df['Target'].tolist()))
            X_val = concat(ecfp(val_df['Drug_canonical'].tolist()),
                          amino_acid_composition(val_df['Target'].tolist()))
            X_test = concat(ecfp(test_df['Drug_canonical'].tolist()),
                           amino_acid_composition(test_df['Target'].tolist()))

            # --- XGBoost ---
            model_name = 'XGBoost_ECFP4+AAC'
            if (model_name, split_type, str(seed)) not in existing:
                print(f"{model_name} | {split_type} | seed={seed}", flush=True)
                t0 = time.time()
                xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                                             random_state=seed, n_jobs=-1, verbosity=0)
                xgb_model.fit(X_train, y_train)
                train_time = time.time() - t0
                val_metrics = evaluate(y_val, xgb_model.predict(X_val))
                test_metrics = evaluate(y_test, xgb_model.predict(X_test))
                writer.writerow({
                    'timestamp': datetime.utcnow().isoformat(),
                    'model': model_name, 'model_family': 'xgb',
                    'split_type': split_type, 'seed': seed,
                    'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
                    'val_rmse': val_metrics['rmse'], 'val_mae': val_metrics['mae'],
                    'val_pearson': val_metrics['pearson'],
                    'test_rmse': test_metrics['rmse'], 'test_mae': test_metrics['mae'],
                    'test_pearson': test_metrics['pearson'], 'test_spearman': test_metrics['spearman'],
                    'n_params': '', 'train_time_s': round(train_time, 1),
                    'epochs_trained': '', 'device': 'CPU',
                })
                f.flush()
                print(f"  -> test_rmse={test_metrics['rmse']:.4f}, pearson={test_metrics['pearson']:.4f}, time={train_time:.0f}s", flush=True)

            # --- KNN ---
            model_name = 'KNN_ECFP4+AAC'
            if (model_name, split_type, str(seed)) not in existing:
                print(f"{model_name} | {split_type} | seed={seed}", flush=True)
                t0 = time.time()
                knn = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
                knn.fit(X_train, y_train)
                train_time = time.time() - t0
                val_metrics = evaluate(y_val, knn.predict(X_val))
                test_metrics = evaluate(y_test, knn.predict(X_test))
                writer.writerow({
                    'timestamp': datetime.utcnow().isoformat(),
                    'model': model_name, 'model_family': 'knn',
                    'split_type': split_type, 'seed': seed,
                    'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
                    'val_rmse': val_metrics['rmse'], 'val_mae': val_metrics['mae'],
                    'val_pearson': val_metrics['pearson'],
                    'test_rmse': test_metrics['rmse'], 'test_mae': test_metrics['mae'],
                    'test_pearson': test_metrics['pearson'], 'test_spearman': test_metrics['spearman'],
                    'n_params': '', 'train_time_s': round(train_time, 1),
                    'epochs_trained': '', 'device': 'CPU',
                })
                f.flush()
                print(f"  -> test_rmse={test_metrics['rmse']:.4f}, pearson={test_metrics['pearson']:.4f}, time={train_time:.0f}s", flush=True)

            # --- Ridge ---
            model_name = 'Ridge_ECFP4+AAC'
            if (model_name, split_type, str(seed)) not in existing:
                print(f"{model_name} | {split_type} | seed={seed}", flush=True)
                t0 = time.time()
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train, y_train)
                train_time = time.time() - t0
                val_metrics = evaluate(y_val, ridge.predict(X_val))
                test_metrics = evaluate(y_test, ridge.predict(X_test))
                writer.writerow({
                    'timestamp': datetime.utcnow().isoformat(),
                    'model': model_name, 'model_family': 'linear',
                    'split_type': split_type, 'seed': seed,
                    'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
                    'val_rmse': val_metrics['rmse'], 'val_mae': val_metrics['mae'],
                    'val_pearson': val_metrics['pearson'],
                    'test_rmse': test_metrics['rmse'], 'test_mae': test_metrics['mae'],
                    'test_pearson': test_metrics['pearson'], 'test_spearman': test_metrics['spearman'],
                    'n_params': '', 'train_time_s': round(train_time, 1),
                    'epochs_trained': '', 'device': 'CPU',
                })
                f.flush()
                print(f"  -> test_rmse={test_metrics['rmse']:.4f}, pearson={test_metrics['pearson']:.4f}, time={train_time:.0f}s", flush=True)

print("\n[Extra baselines] DONE.")
