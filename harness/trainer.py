"""
Generic trainer for sklearn-compatible models and PyTorch nn.Module models.
All models go through the same train/evaluate loop.
"""

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from harness.config import (
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY, DEVICE
)
from harness.metrics import compute_metrics


def _get_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Sklearn interface ─────────────────────────────────────────────────────────

def train_sklearn(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Fit a sklearn-compatible model.
    Returns (val_metrics, test_metrics, train_time, 0, test_pred).
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    test_pred    = model.predict(X_test)
    val_metrics  = compute_metrics(y_val,  model.predict(X_val))
    test_metrics = compute_metrics(y_test, test_pred)
    return val_metrics, test_metrics, train_time, 0, test_pred


# ── PyTorch interface ─────────────────────────────────────────────────────────

def _make_loader(X, y, shuffle: bool, batch_size: int) -> DataLoader:
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def train_torch(
    model: nn.Module,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    *,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience:   int = PATIENCE,
    lr:         float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    aux_loss=None,
    dsm_aux=None,
) -> tuple[dict, dict, float, int]:
    """
    Train a PyTorch model with early stopping on val RMSE.

    Args:
        aux_loss: optional callable(pred, y) -> scalar tensor added to MSE loss.
                  Pass a PairwiseRankingLoss instance for ranking-aware training.
        dsm_aux:  optional callable(z) -> scalar tensor (DualBind DSM loss).
                  If provided, model.encode(x) is called to get embedding z,
                  then model.predict_from_embedding(z) for the prediction.
                  Default None = standard MSE-only training (existing behaviour).

    Returns (val_metrics, test_metrics, train_time_sec, epochs_trained, test_pred).
    """
    device = _get_device()
    model = model.to(device)
    if aux_loss is not None and hasattr(aux_loss, 'to'):
        aux_loss = aux_loss.to(device)

    train_loader = _make_loader(X_train, y_train, shuffle=True,  batch_size=batch_size)
    val_loader   = _make_loader(X_val,   y_val,   shuffle=False, batch_size=batch_size * 4)
    test_loader  = _make_loader(X_test,  y_test,  shuffle=False, batch_size=batch_size * 4)

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
        # ── Train ──
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if dsm_aux is not None:
                z    = model.encode(X_batch)
                pred = model.predict_from_embedding(z)
                loss = criterion(pred, y_batch) + dsm_aux(z)
            else:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
            if aux_loss is not None:
                loss = loss + aux_loss(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── Validate ──
        val_metrics = _evaluate_loader(model, val_loader, device)
        scheduler.step(val_metrics["rmse"])

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  val_rmse={val_metrics['rmse']:.4f}  "
                  f"patience={patience_ctr}/{patience}")

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights and evaluate on test
    model.load_state_dict(best_state)
    val_metrics  = _evaluate_loader(model, val_loader,  device)
    test_metrics, test_pred = _evaluate_loader_with_pred(model, test_loader, device)
    train_time   = time.time() - t0

    return val_metrics, test_metrics, train_time, epoch, test_pred


def _evaluate_loader(model: nn.Module, loader: DataLoader, device) -> dict:
    metrics, _ = _evaluate_loader_with_pred(model, loader, device)
    return metrics


def _evaluate_loader_with_pred(model: nn.Module, loader: DataLoader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            preds.append(pred)
            targets.append(y_batch.numpy())
    y_pred = np.concatenate(preds).ravel()
    y_true = np.concatenate(targets).ravel()
    return compute_metrics(y_true, y_pred), y_pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
