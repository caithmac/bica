"""
e6_mlp_capacity_sweep.py
=======================
MLP capacity sweep: how many parameters does an MLP need to match RF?

Design:
  - Fixed features: ECFP4-1024 + AAC-20 (same as best RF)
  - Fixed split: scaffold_bemis_murcko_seed42
  - Sweep MLP hidden dims to span 300K → 10M params
  - Report RMSE, Pearson r, n_params for each
  - Compare against RF baseline: RMSE 1.0065

Usage:
    cd E:/BICA
    python experiments/e6_mlp_capacity_sweep.py
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.data import get_splits_for_seed
from harness.config import SMILES_COL, PROTEIN_COL, LABEL_COL, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE
from harness.metrics import compute_metrics
from harness.diary import log_result
from harness.trainer import count_parameters, train_torch
import harness.featurizers as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ---------------------------------------------------------------------------
# Configurations to sweep — (hidden_dims, label)
# ---------------------------------------------------------------------------
SWEEP = [
    ([64],              "mlp_cap_64"),
    ([128],             "mlp_cap_128"),
    ([256],             "mlp_cap_256"),
    ([256, 128],        "mlp_cap_256_128"),       # ~300K — the "matched" architecture
    ([512, 256],        "mlp_cap_512_256"),       # ~800K
    ([1024, 512],       "mlp_cap_1024_512"),      # ~2M
    ([1024, 512, 256],  "mlp_cap_1024_512_256"),  # ~3M
    ([2048, 1024, 512], "mlp_cap_2048_1024_512"), # ~6M
    ([2048, 2048, 1024],"mlp_cap_2k_2k_1k"),      # ~10M
]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        d_in = input_dim
        for d_out in hidden_dims:
            layers.extend([
                nn.Linear(d_in, d_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            d_in = d_out
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    print("E6: MLP capacity sweep vs RF (ECFP4 + AAC, scaffold seed 42)")
    print(f"  RF baseline: RMSE = 1.0065")
    print(f"  Device: {DEVICE}")
    print()

    train_df, val_df, test_df = get_splits_for_seed(SEED)

    # Features — same as RF
    L_train = F.ecfp(train_df[SMILES_COL].tolist(), radius=2, nbits=1024)
    L_val   = F.ecfp(val_df[SMILES_COL].tolist(), radius=2, nbits=1024)
    L_test  = F.ecfp(test_df[SMILES_COL].tolist(), radius=2, nbits=1024)

    P_train = F.amino_acid_composition(train_df[PROTEIN_COL].tolist())
    P_val   = F.amino_acid_composition(val_df[PROTEIN_COL].tolist())
    P_test  = F.amino_acid_composition(test_df[PROTEIN_COL].tolist())

    X_train = np.concatenate([L_train, P_train], axis=1)
    X_val   = np.concatenate([L_val,   P_val],   axis=1)
    X_test  = np.concatenate([L_test,  P_test],  axis=1)

    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val   = val_df[LABEL_COL].values.astype(np.float32)
    y_test  = test_df[LABEL_COL].values.astype(np.float32)

    input_dim = X_train.shape[1]  # 1024 + 20 = 1044

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Input dim: {input_dim}")
    print()
    print(f"{'Label':<30s} {'Params':>10s} {'Test RMSE':>10s} {'Test r':>8s} {'vs RF Δ':>8s}")
    print("-" * 72)

    results = []

    for hidden_dims, label in SWEEP:
        model = SimpleMLP(input_dim, hidden_dims).to(DEVICE)
        n_params = count_parameters(model)

        t0 = time.time()
        val_metrics, test_metrics, train_time, best_epoch, preds = train_torch(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            lr=LEARNING_RATE,
        )

        model.eval()
        m = test_metrics  # already contains rmse, pearson_r, spearman_r
        delta = m["rmse"] - 1.0065

        print(f"{label:<30s} {n_params:>10,d} {m['rmse']:>10.4f} {m['pearson_r']:>8.3f} {delta:>+8.4f}")

        results.append((label, n_params, m, delta, train_time))

        # Log to diary
        log_result(
            experiment_id=f"{label}_seed{SEED}",
            model_name=label,
            model_family="mlp",
            ligand_repr="ecfp4_1024",
            protein_repr="aac_20",
            fusion_strategy="concat",
            n_params=n_params,
            epochs_trained=best_epoch,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            split_type=f"scaffold_bemis_murcko_seed{SEED}",
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
            val_metrics={k: val_metrics.get(k, 0) for k in ["rmse", "pearson_r", "spearman_r"]},
            test_metrics={k: m[k] for k in ["rmse", "pearson_r", "spearman_r"]},
            train_time_sec=train_time,
            notes=f"E6 capacity sweep: MLP hidden_dims={hidden_dims}, ECFP4+AAC",
        )

    # Summary
    print()
    print("=" * 72)
    print("Summary: MLP vs RF (RMSE = 1.0065)")
    print()
    for label, n_params, m, delta, _ in sorted(results, key=lambda x: x[1]):
        bar = "█" * int(m["rmse"] * 10) if m["rmse"] < 2 else "TOO_HIGH"
        print(f"  {label:<30s} {n_params:>10,d} params  RMSE={m['rmse']:.4f}  {bar}")

    # RF comparison
    print(f"  {'RF + ECFP4 + AAC':<30s} {'(500 trees)':>10s}  RMSE=1.0065  {'█' * 10}")

    # Find best MLP
    best = min(results, key=lambda x: x[2]["rmse"])
    gap = best[2]["rmse"] - 1.0065
    print()
    print(f"  Best MLP: {best[0]} ({best[1]:,d} params) RMSE={best[2]['rmse']:.4f}")
    print(f"  Gap to RF: {gap:+.4f} ({gap/1.0065*100:+.1f}%)")
    if gap > 0:
        print(f"  MLP needs MORE than {best[1]:,d} params to match RF")
    print()
    print("✅ E6 complete.")


if __name__ == "__main__":
    main()
