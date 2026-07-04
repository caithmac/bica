"""
e10_deepdta_baseline.py — DeepDTA baseline using existing CNN1D architecture.

DeepDTA (Öztürk et al. 2018): Dual 1D-CNN on character-one-hot SMILES + protein sequences.
Reuses models/cnn.py CNN1D pattern + harness.featurizers.smiles_char_onehot.

Usage:
    cd E:/BICA
    python e10_deepdta_baseline.py
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.data import get_splits_for_seed
from harness.config import SMILES_COL, PROTEIN_COL, LABEL_COL, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE
from harness.metrics import compute_metrics
from harness.diary import log_result, save_predictions
from harness.trainer import count_parameters, train_torch
import harness.featurizers as F

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ── Protein character one-hot (mirrors smiles_char_onehot) ───────────
PROTEIN_CHARS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids

def protein_char_onehot(sequences, max_len=1200):
    """Character-level one-hot for protein sequences, flattened."""
    vocab_size = len(PROTEIN_CHARS)
    char2idx = {c: i for i, c in enumerate(PROTEIN_CHARS)}
    out = np.zeros((len(sequences), max_len * vocab_size), dtype=np.float32)
    for row_idx, seq in enumerate(sequences):
        for col_idx, ch in enumerate(seq[:max_len]):
            if ch in char2idx:
                out[row_idx, col_idx * vocab_size + char2idx[ch]] = 1.0
    return out


# ── DeepDTA model (reuses CNN1D conv pattern) ────────────────────────
class ConvBranch(nn.Module):
    """Multi-kernel 1D-CNN branch (like CNN1D but without the FC head)."""
    def __init__(self, seq_len, vocab_size, num_filters=128, kernel_sizes=(3,5,9), dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(vocab_size, num_filters, k, padding=k//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            for k in kernel_sizes
        ])
        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, x):
        # x: (B, seq_len * vocab_size)
        x = x.view(x.size(0), self.seq_len, self.vocab_size)
        x = x.transpose(1, 2)  # (B, vocab_size, seq_len)
        pooled = [conv(x).squeeze(-1) for conv in self.convs]
        return torch.cat(pooled, dim=1)


class DeepDTAFull(nn.Module):
    """Dual CNN branches → concat → FC head."""
    def __init__(self, smiles_max_len=100, prot_max_len=1200,
                 num_filters=128, kernel_sizes=(3,5,9), dropout=0.3):
        super().__init__()
        self.smiles_branch = ConvBranch(smiles_max_len, 39, num_filters, kernel_sizes, dropout)
        self.prot_branch = ConvBranch(prot_max_len, 20, num_filters, kernel_sizes, dropout)

        concat_dim = self.smiles_branch.out_dim + self.prot_branch.out_dim
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
        self.smiles_dim = smiles_max_len * 39  # 3900
        self.prot_dim = prot_max_len * 20      # 24000

    def forward(self, x):
        # x: (B, smiles_dim + prot_dim)
        smiles_x = x[:, :self.smiles_dim]
        prot_x = x[:, self.smiles_dim:]
        s = self.smiles_branch(smiles_x)
        p = self.prot_branch(prot_x)
        combined = torch.cat([s, p], dim=1)
        return self.fc(combined)


def run_one(seed):
    """Run DeepDTA for one seed."""
    train_df, val_df, test_df = get_splits_for_seed(seed)

    # Featurize
    L_train = F.smiles_char_onehot(train_df[SMILES_COL].tolist(), max_len=100)
    L_val   = F.smiles_char_onehot(val_df[SMILES_COL].tolist(), max_len=100)
    L_test  = F.smiles_char_onehot(test_df[SMILES_COL].tolist(), max_len=100)

    P_train = protein_char_onehot(train_df[PROTEIN_COL].tolist(), max_len=1200)
    P_val   = protein_char_onehot(val_df[PROTEIN_COL].tolist(), max_len=1200)
    P_test  = protein_char_onehot(test_df[PROTEIN_COL].tolist(), max_len=1200)

    X_train = np.concatenate([L_train, P_train], axis=1)
    X_val   = np.concatenate([L_val, P_val], axis=1)
    X_test  = np.concatenate([L_test, P_test], axis=1)

    y_train = train_df[LABEL_COL].values
    y_val   = val_df[LABEL_COL].values
    y_test  = test_df[LABEL_COL].values

    # Scale (feature values are 0/1 but helps with batchnorm)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"  seed={seed}  X={X_train.shape}")

    model = DeepDTAFull(
        smiles_max_len=100, prot_max_len=1200,
        num_filters=128, kernel_sizes=(3,5,9), dropout=0.3
    )
    n_params = count_parameters(model)
    print(f"  params={n_params:,}")

    t0 = time.time()
    val_m, test_m, train_time, epochs, test_pred = train_torch(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32, max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LEARNING_RATE
    )

    exp_id = f"deepdta_seed{seed}"
    save_predictions(exp_id, y_test, test_pred)
    log_result(
        experiment_id=exp_id,
        model_name="deepdta",
        model_family="deepdta",
        ligand_repr="smiles_char_onehot_3900",
        protein_repr="protein_char_onehot_24000",
        fusion_strategy="concat_cnn_dual",
        n_params=n_params,
        epochs_trained=epochs,
        batch_size=32,
        learning_rate=LEARNING_RATE,
        split_type=f"scaffold_bemis_murcko_seed{seed}",
        n_train=len(train_df), n_val=len(val_df), n_test=len(test_df),
        val_metrics=val_m, test_metrics=test_m,
        train_time_sec=train_time,
        notes=f"E10: DeepDTA (Öztürk 2018), dual 1D-CNN, scaffold split seed{seed}",
    )
    return test_m


if __name__ == "__main__":
    print("=" * 60)
    print("E10: DeepDTA BASELINE (3 seeds, scaffold split)")
    print("=" * 60)

    for seed in [42, 123, 456]:
        print(f"\\n--- seed={seed} ---")
        m = run_one(seed)
        print(f"  test_rmse={m['rmse']:.4f}  pearson_r={m['pearson_r']:.4f}")

    print("\\n✅ E10 complete.")
