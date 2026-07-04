"""
e4_e5_zeroshot.py
=================
Zero-shot RF evaluation on USP7 and Mpro with CONSISTENT training sets.

E4: Fixes ambiguous training set in Table zeroshot.
E5: ESM-2 8M zero-shot (currently missing from diary entirely).

Design:
- Two training sets: scaffold-train (N=17,312 from seed 42) & full BindingDB (N=24,700)
- Three protein encodings: AAC-20, target_binary, ESM-2 8M
- All use ECFP4 fingerprints + RF (500 trees, max_depth=20)
- All logged to diary with explicit n_train and training_set annotation

Usage:
    cd E:/BICA
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
        python e4_e5_zeroshot.py
"""

import os, sys, time
import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.data import get_splits, load_raw
from harness.config import SMILES_COL, PROTEIN_COL, LABEL_COL
from harness.metrics import compute_metrics
from harness.diary import log_result, save_predictions
import harness.featurizers as F

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from datasets import load_dataset as hf_load
from sklearn.ensemble import RandomForestRegressor


def load_zeroshot_targets():
    """Load USP7 and Mpro from BALM-benchmark."""
    targets = {}
    for name in ["USP7", "Mpro"]:
        ds = hf_load("BALM/BALM-benchmark", name, split="train")
        df = ds.to_pandas()
        df = df.rename(columns={
            "Drug": SMILES_COL, "Target": PROTEIN_COL, "Y": LABEL_COL
        })
        targets[name] = df
    return targets


def build_protein(seqs, repr_name):
    """Build protein features for given representation."""
    if repr_name == "aac_20":
        return F.amino_acid_composition(seqs)
    elif repr_name == "target_binary":
        # Binary target encoding — but for zero-shot, unseen targets get all-zeros
        # We use protein sequence to derive a hash-based binary ID
        # For zero-shot: use all zeros for unseen proteins
        return np.zeros((len(seqs), 11))
    elif repr_name == "esm2_8M_320":
        return F.esm2_embeddings(seqs, model_size="8M")
    else:
        raise ValueError(f"Unknown: {repr_name}")


def run_zeroshot(train_df, train_label, n_train_actual, seed,
                 prot_repr, prot_label,
                 test_targets):
    """
    Train RF on train_df, zero-shot evaluate on USP7 and Mpro.
    Returns dict of results.
    """
    print(f"\n  [{prot_label}] Training on {train_label} (n={len(train_df)})...")
    
    # Build training features
    L_train = F.ecfp(train_df[SMILES_COL].tolist(), radius=2, nbits=1024)
    
    if prot_repr == "target_binary":
        # For training: derive binary from target ID
        # We use unique Target values and assign binary codes
        target_ids = train_df[PROTEIN_COL].unique()
        target_to_binary = {
            tid: np.array(list(np.binary_repr(i, width=11)), dtype=np.float32)
            for i, tid in enumerate(target_ids)
        }
        P_train = np.array([target_to_binary[t] for t in train_df[PROTEIN_COL].tolist()])
    else:
        P_train = build_protein(train_df[PROTEIN_COL].tolist(), prot_repr)
    
    X_train = np.concatenate([L_train, P_train], axis=1)
    y_train = train_df[LABEL_COL].values
    
    # Train RF
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, n_jobs=1, random_state=seed)
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"    Trained in {train_time:.0f}s")
    
    # Zero-shot on USP7 and Mpro
    results = {}
    for zs_name, df_zs in test_targets.items():
        L_test = F.ecfp(df_zs[SMILES_COL].tolist(), radius=2, nbits=1024)
        
        if prot_repr == "target_binary":
            P_test = np.zeros((len(df_zs), 11))  # unseen targets
        else:
            P_test = build_protein(df_zs[PROTEIN_COL].tolist(), prot_repr)
        
        X_test = np.concatenate([L_test, P_test], axis=1)
        y_test = df_zs[LABEL_COL].values
        preds = rf.predict(X_test)
        m = compute_metrics(y_test, preds)
        results[zs_name] = m
        
        print(f"    {zs_name}: RMSE={m['rmse']:.4f}  r={m['pearson_r']:.4f}  "
              f"ρ={m['spearman_r']:.4f}")
    
    # Log to diary
    exp_id = f"rf_ecfp4_{prot_repr}_zeroshot_{train_label}"
    # Log test metrics for USP7 (primary target)
    log_result(
        experiment_id=exp_id,
        model_name=exp_id,
        model_family="tree",
        ligand_repr="ecfp4_1024",
        protein_repr=prot_repr,
        fusion_strategy="concat",
        n_params="N/A",
        epochs_trained=0,
        batch_size=0,
        learning_rate=0,
        split_type=f"zeroshot_{train_label}",
        n_train=n_train_actual,
        n_val=0,
        n_test=len(test_targets["USP7"]),
        val_metrics={"rmse": 0, "pearson_r": 0, "spearman_r": 0},
        test_metrics=results["USP7"],
        train_time_sec=train_time,
        notes=f"E4/E5: Zero-shot RF ({prot_label}) trained on {train_label} "
              f"(n={n_train_actual}) → USP7/Mpro. "
              f"Mpro: RMSE={results['Mpro']['rmse']:.4f} r={results['Mpro']['pearson_r']:.4f}",
    )
    
    # Also save predictions for both targets
    save_predictions(f"{exp_id}_USP7", test_targets["USP7"][LABEL_COL].values,
                     rf.predict(np.concatenate([
                         F.ecfp(test_targets["USP7"][SMILES_COL].tolist(), radius=2, nbits=1024),
                         build_protein(test_targets["USP7"][PROTEIN_COL].tolist(), prot_repr)
                         if prot_repr != "target_binary"
                         else np.zeros((len(test_targets["USP7"]), 11))
                     ], axis=1)))
    
    return results


def main():
    print("=" * 60)
    print("E4+E5: ZERO-SHOT RF — CONSISTENT TRAINING SETS")
    print("=" * 60)
    
    # Load test targets
    print("\n[1] Loading USP7 and Mpro...")
    test_targets = load_zeroshot_targets()
    for name, df in test_targets.items():
        print(f"    {name}: {len(df)} compounds")
    
    # Load BindingDB — both scaffold-train-only and full
    print("\n[2] Loading BindingDB...")
    full_df = load_raw()
    scaffold_train_df, _, _ = get_splits(full_df)
    print(f"    Full BindingDB: {len(full_df)}")
    print(f"    Scaffold-train-only (seed 42): {len(scaffold_train_df)}")
    
    # Configs to run
    prot_configs = [
        ("aac_20", "AAC (20-dim)"),
        ("target_binary", "Binary ID (11-bit)"),
        ("esm2_8M_320", "ESM-2 8M (320-dim)"),
    ]
    
    train_configs = [
        (scaffold_train_df, "scaffold_train_s42", 17312),
        (full_df, "full_bindingdb", 24700),
    ]
    
    all_results = {}
    for train_df, train_label, n_train in train_configs:
        print(f"\n{'='*60}")
        print(f"Training set: {train_label} (n={len(train_df)})")
        
        for prot_repr, prot_label in prot_configs:
            res = run_zeroshot(
                train_df, train_label, n_train, 42,
                prot_repr, prot_label, test_targets
            )
            all_results[f"{train_label}_{prot_label}"] = res
    
    # Final summary
    print("\n" + "=" * 70)
    print("E4+E5 FINAL RESULTS: Zero-Shot RF → USP7 / Mpro")
    print("=" * 70)
    print(f"{'Training set':<22s} {'Protein':<20s} {'USP7 RMSE':>10s} {'USP7 r':>8s} {'Mpro RMSE':>10s} {'Mpro r':>8s}")
    print("-" * 70)
    
    for train_label in ["scaffold_train_s42", "full_bindingdb"]:
        for prot_label, _ in prot_configs:
            key = f"{train_label}_{prot_label}"
            if key in all_results:
                usp7 = all_results[key]["USP7"]
                mpro = all_results[key]["Mpro"]
                print(f"  {train_label:<20s}  {prot_label:<18s}  "
                      f"{usp7['rmse']:10.4f} {usp7['pearson_r']:8.4f}  "
                      f"{mpro['rmse']:10.4f} {mpro['pearson_r']:8.4f}")
    
    # Compare to paper's claims
    print(f"\n  PAPER CLAIMED (full BindingDB):")
    print(f"  {'AAC (20-dim)':<20s}  {'':>10s} {'0.38':>8s}  {'':>10s} {'-0.01':>8s}")
    print(f"  {'ESM-2 8M':<20s}  {'':>10s} {'0.58':>8s}  {'':>10s} {'-0.04':>8s}")
    
    print("\n✅ E4+E5 complete.")


if __name__ == "__main__":
    main()
