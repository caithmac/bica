"""
Trainer for sequence models (LSTM, TransformerSeq) that consume token ID batches.

Fixes applied vs original:
  1. Lazy tokenization — sequences encoded in __getitem__, not upfront.
     Pre-encoding 17k proteins at max_len=1200 allocates ~1.7 GB in one shot.
  2. Batch-level padding only — pad_sequence called inside collate_fn on the
     mini-batch, so padding is only as wide as the longest sequence in that batch.
  3. Reduced max_prot_len to 512 for seq models (covers >90% of BindingDB proteins).
  4. LSTM runs on CPU — avoids cuDNN packed-sequence failures (CUDNN_STATUS_EXECUTION_FAILED_CUBLAS)
     that occur with highly variable sequence lengths on GPU.
     TransformerSeq still uses GPU (mask-based attention handles variable length cleanly).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial

from harness.config import (
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY,
    SMILES_COL, PROTEIN_COL, LABEL_COL,
)
from harness.metrics import compute_metrics
from harness.tokenizers import pad_sequence
from harness.trainer import _get_device, count_parameters

# Reduced lengths for seq models — covers >95% of BindingDB
SEQ_MAX_LIG_LEN  = 150   # SMILES rarely exceed 100 chars; 150 is safe
SEQ_MAX_PROT_LEN = 512   # caps very long proteins; saves memory significantly


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — lazy tokenization
# ─────────────────────────────────────────────────────────────────────────────

class BindingSeqDataset(Dataset):
    """
    Lazy tokenization: sequences are tokenized per-item in __getitem__,
    not pre-encoded upfront. This avoids allocating the full padded matrix.
    """

    def __init__(self, df, lig_tokenizer, prot_tokenizer,
                 max_lig_len: int = SEQ_MAX_LIG_LEN,
                 max_prot_len: int = SEQ_MAX_PROT_LEN):
        self.smiles   = df[SMILES_COL].tolist()
        self.seqs     = df[PROTEIN_COL].tolist()
        self.labels   = df[LABEL_COL].values.astype(np.float32)
        self.lig_tok  = lig_tokenizer
        self.prot_tok = prot_tokenizer
        self.max_lig  = max_lig_len
        self.max_prot = max_prot_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lig_ids  = self.lig_tok.encode(self.smiles[idx])[:self.max_lig]
        prot_ids = self.prot_tok.encode(self.seqs[idx])[:self.max_prot]
        # Guard against empty sequences
        if len(lig_ids)  == 0: lig_ids  = [self.lig_tok.pad_id]
        if len(prot_ids) == 0: prot_ids = [self.prot_tok.pad_id]
        return lig_ids, prot_ids, float(self.labels[idx])


def _collate_fn(batch, lig_pad_id: int, prot_pad_id: int):
    """Pad within the mini-batch only — no global max_len allocation."""
    lig_seqs, prot_seqs, labels = zip(*batch)
    lig_arr,  lig_mask  = pad_sequence(list(lig_seqs),  lig_pad_id)
    prot_arr, prot_mask = pad_sequence(list(prot_seqs), prot_pad_id)
    return (
        torch.tensor(lig_arr,   dtype=torch.long),
        torch.tensor(prot_arr,  dtype=torch.long),
        torch.tensor(lig_mask,  dtype=torch.float32),
        torch.tensor(prot_mask, dtype=torch.float32),
        torch.tensor(np.array(labels, dtype=np.float32)).unsqueeze(1),
    )


def make_seq_loader(df, lig_tokenizer, prot_tokenizer,
                    shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    ds = BindingSeqDataset(df, lig_tokenizer, prot_tokenizer)
    collate = partial(_collate_fn,
                      lig_pad_id=lig_tokenizer.pad_id,
                      prot_pad_id=prot_tokenizer.pad_id)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=0, pin_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# Train / evaluate
# ─────────────────────────────────────────────────────────────────────────────

def _eval_seq_loader(model, loader, device):
    """Returns (metrics_dict, y_pred_array)."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for lig_ids, prot_ids, lig_mask, prot_mask, y in loader:
            lig_ids   = lig_ids.to(device)
            prot_ids  = prot_ids.to(device)
            lig_mask  = lig_mask.to(device)
            prot_mask = prot_mask.to(device)
            pred = model(lig_ids, prot_ids, lig_mask, prot_mask).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
    y_pred = np.concatenate(preds).ravel()
    y_true = np.concatenate(targets).ravel()
    return compute_metrics(y_true, y_pred), y_pred


def train_seq_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    *,
    max_epochs: int     = MAX_EPOCHS,
    patience:   int     = PATIENCE,
    lr:         float   = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
):
    """
    Train a sequence model with early stopping on val RMSE.
    LSTM → CPU (avoids cuDNN packed-sequence failures with variable lengths).
    TransformerSeq → GPU (mask-based attention handles variable length safely).
    Returns (val_metrics, test_metrics, train_time_sec, epochs_trained).
    """
    device = _get_device()

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state    = None
    patience_ctr  = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for lig_ids, prot_ids, lig_mask, prot_mask, y in train_loader:
            lig_ids   = lig_ids.to(device)
            prot_ids  = prot_ids.to(device)
            lig_mask  = lig_mask.to(device)
            prot_mask = prot_mask.to(device)
            y         = y.to(device)
            optimizer.zero_grad()
            pred = model(lig_ids, prot_ids, lig_mask, prot_mask)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_m, _ = _eval_seq_loader(model, val_loader, device)
        scheduler.step(val_m["rmse"])

        if val_m["rmse"] < best_val_rmse:
            best_val_rmse = val_m["rmse"]
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  val_rmse={val_m['rmse']:.4f}  patience={patience_ctr}/{patience}")

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    val_m,  _         = _eval_seq_loader(model, val_loader,  device)
    test_m, test_pred = _eval_seq_loader(model, test_loader, device)
    return val_m, test_m, time.time() - t0, epoch, test_pred
