"""
Training loop for GNN models (GCN, GAT) that use PyTorch Geometric.

Key difference from trainer.py: the data loader yields PyG Batch objects
instead of plain tensors. The protein vector is carried as a separate
tensor alongside the graph batch.

Entry points:
  make_gnn_loader()       — build a DataLoader of (PyG Data, prot_vec, y) triples
  train_gnn_model()       — full train / early-stop / evaluate loop
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from harness.metrics import compute_metrics
from harness.config import MAX_EPOCHS, PATIENCE, BATCH_SIZE, LEARNING_RATE


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GNNDataset(Dataset):
    """
    Each item: (PyG Data object for the ligand, prot_vec, y_value)
    Invalid SMILES → node feature tensor of shape (1, 78) with zeros.
    """

    def __init__(self, smiles_list: List[str], prot_vecs: np.ndarray,
                 labels: np.ndarray):
        from models.gnn import mol_to_graph_data
        import torch
        from torch_geometric.data import Data

        self.items = []
        zero_graph = Data(
            x          = torch.zeros(1, 78),
            edge_index = torch.zeros(2, 1, dtype=torch.long),
            edge_attr  = torch.zeros(1, 10),
        )

        for smi, pv, y in zip(smiles_list, prot_vecs, labels):
            result = mol_to_graph_data(smi)
            if result is None:
                graph = zero_graph
            else:
                x, edge_index, edge_attr = result
                graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            prot_t = torch.tensor(pv, dtype=torch.float32)
            y_t    = torch.tensor([y], dtype=torch.float32)
            self.items.append((graph, prot_t, y_t))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _gnn_collate(batch):
    """Collate a list of (Data, prot_vec, y) into a PyG Batch."""
    from torch_geometric.data import Batch
    import torch

    graphs, prot_vecs, ys = zip(*batch)
    batched_graph = Batch.from_data_list(graphs)
    prot_tensor   = torch.stack(prot_vecs)
    y_tensor      = torch.cat(ys)
    return batched_graph, prot_tensor, y_tensor


def make_gnn_loader(smiles_list: List[str], prot_vecs: np.ndarray,
                    labels: np.ndarray, shuffle: bool = True,
                    batch_size: int = BATCH_SIZE) -> DataLoader:
    ds = GNNDataset(smiles_list, prot_vecs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_gnn_collate, num_workers=0)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _eval_gnn(model, loader, device):
    """Returns (metrics_dict, y_pred_array)."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_graph, prot_t, y_t in loader:
            batch_graph = batch_graph.to(device)
            prot_t = prot_t.to(device)
            out = model(batch_graph.x, batch_graph.edge_index,
                        batch_graph.batch, prot_t)
            preds.append(out.cpu().numpy().ravel())
            targets.append(y_t.numpy().ravel())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return compute_metrics(y_true, y_pred), y_pred


def train_gnn_model(
    model:           nn.Module,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    test_loader:     DataLoader,
    lr:              float = LEARNING_RATE,
    max_epochs:      int   = MAX_EPOCHS,
    patience:        int   = PATIENCE,
    aux_loss=None,
    use_node_recon:  bool  = False,
    recon_lambda:    float = 0.1,
    mask_rate:       float = 0.15,
) -> Tuple[dict, dict, float, int]:
    """
    Train a GNN model with early stopping on val RMSE.

    Args:
        aux_loss:       optional callable(pred, y) → scalar, e.g. PairwiseRankingLoss.
        use_node_recon: if True, adds GraphMAE-style node reconstruction auxiliary loss.
                        Requires the model to have an encode_nodes() method and the
                        harness to attach a NodeDecoder to model.node_decoder.
        recon_lambda:   weight for reconstruction loss (default 0.1).
        mask_rate:      fraction of nodes to mask per forward pass (default 0.15).

    Returns: (val_metrics, test_metrics, train_time_sec, epochs_done, test_pred)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    if aux_loss is not None and hasattr(aux_loss, 'to'):
        aux_loss = aux_loss.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_rmse    = float("inf")
    best_state   = None
    patience_ctr = 0
    t0           = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_graph, prot_t, y_t in train_loader:
            batch_graph = batch_graph.to(device)
            prot_t = prot_t.to(device)
            y_t    = y_t.to(device)

            optimizer.zero_grad()

            if use_node_recon and hasattr(model, 'node_decoder'):
                # ── GraphMAE dual-forward ──────────────────────────────────
                from models.gnn import mask_node_features
                x_orig = batch_graph.x
                x_masked, mask_idx = mask_node_features(
                    x_orig, mask_rate=mask_rate, training=True)

                # Forward on masked graph for affinity prediction
                pred = model(x_masked, batch_graph.edge_index,
                             batch_graph.batch, prot_t)
                loss = criterion(pred.squeeze(-1), y_t)

                # Node reconstruction loss at masked positions
                h_masked = model.encode_nodes(x_masked, batch_graph.edge_index)
                x_recon  = model.node_decoder(h_masked[mask_idx])
                loss_recon = criterion(x_recon, x_orig[mask_idx])
                loss = loss + recon_lambda * loss_recon
            else:
                pred = model(batch_graph.x, batch_graph.edge_index,
                             batch_graph.batch, prot_t)
                loss = criterion(pred.squeeze(-1), y_t)

            if aux_loss is not None:
                loss = loss + aux_loss(pred.squeeze(-1), y_t)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_m, _ = _eval_gnn(model, val_loader, device)
        scheduler.step(val_m["rmse"])

        if val_m["rmse"] < best_rmse:
            best_rmse    = val_m["rmse"]
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  val_rmse={val_m['rmse']:.4f}  "
                  f"pearson={val_m['pearson_r']:.4f}  patience={patience_ctr}")

        if patience_ctr >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    val_m,  _         = _eval_gnn(model, val_loader,  device)
    test_m, test_pred = _eval_gnn(model, test_loader, device)
    return val_m, test_m, time.time() - t0, epoch, test_pred
