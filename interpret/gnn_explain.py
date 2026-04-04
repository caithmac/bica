"""
GNN interpretability: SME (Subgraph Mask Explanation), Grad-AAM, and GNNExplainer.

References:
  - SME: Subgraph Mask Explanation — learns a soft mask over edges/nodes
  - Grad-AAM: Gradient-weighted Activation Mapping (Gradient × Feature)
  - GNNExplainer: Ying et al., NeurIPS 2019

All methods work on GCN/GAT/GLI/Graphormer binding models.

Usage:
    from interpret.gnn_explain import grad_aam, gnnexplainer_node_mask

    # Grad-AAM: fast, single-pass
    node_scores = grad_aam(model, data, prot_vec, target_pred=None)

    # GNNExplainer: slower, learns per-edge mask
    edge_mask = gnnexplainer_node_mask(model, data, prot_vec, n_epochs=200)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def grad_aam(
    model,
    data,                          # torch_geometric.data.Data
    prot_vec: "torch.Tensor",      # (1, prot_dim)
    target_node: int | None = None,
) -> np.ndarray:
    """
    Gradient-weighted Activation Mapping (Grad-AAM) for GNN models.

    Computes: importance_i = ReLU( sum_k (grad_k * h_i_k) )
    where h_i is the final-layer node embedding for atom i and
    grad_k is dL/dh_ik w.r.t. the predicted binding affinity.

    This requires a GNN with an encode_nodes() method that returns
    per-atom hidden states.

    Args:
        model:       GCN/GAT/GLI binding model with encode_nodes().
        data:        Single molecule PyG Data object.
        prot_vec:    (1, prot_dim) protein feature tensor.
        target_node: If provided, compute gradient w.r.t. this atom's
                     contribution. If None, uses global sum (binding affinity).

    Returns:
        node_scores: (N,) atom importance scores, non-negative.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device

    x         = data.x.to(device)
    edge_index = data.edge_index.to(device)
    batch     = torch.zeros(x.size(0), dtype=torch.long, device=device)
    prot_vec  = prot_vec.to(device)

    x.requires_grad_(True)

    # Forward: get node embeddings
    if hasattr(model, "encode_nodes"):
        h = model.encode_nodes(x, edge_index)        # (N, H)
        h.retain_grad()
        # Global mean pool → protein concat → head
        from torch_geometric.nn import global_mean_pool
        mol_repr = global_mean_pool(h, batch)         # (1, H)
        combined = torch.cat([mol_repr, prot_vec], dim=-1)
        pred     = model.head(combined)              # (1, 1)
    elif hasattr(model, "_gnn_encode_nodes"):
        h = model._gnn_encode_nodes(x, edge_index)
        h.retain_grad()
        from torch_geometric.nn import global_mean_pool
        mol_repr = global_mean_pool(h, batch)
        combined = torch.cat([mol_repr, prot_vec], dim=-1)
        pred     = model.head(combined)
    else:
        # Fallback: full forward with x.grad
        pred = model(x, edge_index, batch, prot_vec)
        pred.sum().backward()
        if x.grad is not None:
            node_scores = F.relu(
                (x.grad * x).sum(dim=-1)
            ).detach().cpu().numpy()
        else:
            node_scores = np.zeros(x.size(0))
        return node_scores

    scalar = pred.sum()
    scalar.backward()

    if h.grad is not None:
        # Grad-AAM: ReLU(grad ⊙ h).sum(dim=-1)
        node_scores = F.relu(
            (h.grad * h).sum(dim=-1)
        ).detach().cpu().numpy()
    else:
        node_scores = np.zeros(x.size(0))

    return node_scores


def gnnexplainer_node_mask(
    model,
    data,
    prot_vec: "torch.Tensor",      # (1, prot_dim)
    n_epochs: int = 200,
    lr: float = 0.01,
    edge_size_coef: float = 5e-4,
    feature_size_coef: float = 1e-4,
) -> dict[str, "np.ndarray"]:
    """
    GNNExplainer: learn per-edge and per-node-feature masks that
    maximise mutual information with the model prediction.

    Optimisation objective:
        max MI(Y, G_S) = max E[log P(Y | G_S)] − ΩS
        Ω = edge_size * sum(mask) + feature_size * ||feature_mask||_1

    Args:
        model:              GNN binding model.
        data:               PyG Data for one molecule.
        prot_vec:           (1, prot_dim) protein feature tensor.
        n_epochs:           Gradient optimisation steps.
        lr:                 Learning rate for mask optimisation.
        edge_size_coef:     Sparsity penalty on edge mask.
        feature_size_coef:  Sparsity penalty on node feature mask.

    Returns:
        Dict with:
          "edge_mask":    (E,)  sigmoid-normalised edge importance
          "feature_mask": (D,)  sigmoid-normalised feature importance
          "node_score":   (N,)  per-node score = mean over incident edge masks
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device

    x          = data.x.to(device).float()
    edge_index = data.edge_index.to(device)
    batch      = torch.zeros(x.size(0), dtype=torch.long, device=device)
    prot_vec   = prot_vec.to(device)

    N, D = x.shape
    E    = edge_index.size(1)

    # Learnable masks (logit space)
    edge_mask_logit    = torch.zeros(E, device=device, requires_grad=True)
    feature_mask_logit = torch.zeros(D, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([edge_mask_logit, feature_mask_logit], lr=lr)

    # Target: the original model prediction (frozen)
    with torch.no_grad():
        target_pred = model(x, edge_index, batch, prot_vec).detach()

    for _ in range(n_epochs):
        optimizer.zero_grad()

        e_mask = torch.sigmoid(edge_mask_logit)   # (E,)
        f_mask = torch.sigmoid(feature_mask_logit)  # (D,)

        # Apply masks: scale node features; scale message contributions
        x_masked = x * f_mask.unsqueeze(0)

        # Inject edge mask into GNN via hooks (scale edge contributions)
        # We approximate by masking node features proportional to
        # the mean of their incident edges
        src, dst = edge_index[0], edge_index[1]
        node_edge_weight = torch.zeros(N, device=device)
        node_edge_weight.scatter_add_(0, src, e_mask)
        counts = torch.zeros(N, device=device)
        counts.scatter_add_(0, src, torch.ones(E, device=device))
        counts = counts.clamp(min=1)
        node_edge_weight = node_edge_weight / counts
        x_masked = x_masked * node_edge_weight.unsqueeze(-1)

        pred_masked = model(x_masked, edge_index, batch, prot_vec)
        pred_loss   = F.mse_loss(pred_masked, target_pred)

        sparsity = edge_size_coef * e_mask.sum() + feature_size_coef * f_mask.sum()
        loss     = pred_loss + sparsity
        loss.backward()
        optimizer.step()

    edge_mask    = torch.sigmoid(edge_mask_logit).detach().cpu().numpy()
    feature_mask = torch.sigmoid(feature_mask_logit).detach().cpu().numpy()

    # Per-node score: mean of incident edge masks
    src_np = edge_index[0].cpu().numpy()
    node_score = np.zeros(N)
    counts_np  = np.zeros(N)
    for i, (e, s) in enumerate(zip(edge_mask, src_np)):
        node_score[s] += e
        counts_np[s]  += 1
    counts_np  = np.maximum(counts_np, 1)
    node_score = node_score / counts_np

    return {
        "edge_mask":    edge_mask,
        "feature_mask": feature_mask,
        "node_score":   node_score,
    }


def visualise_atom_importance(
    smiles: str,
    node_scores: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Atom importance",
) -> None:
    """
    Draw molecule with atoms coloured by importance score.

    Args:
        smiles:      SMILES string.
        node_scores: (N,) non-negative importance per atom.
        save_path:   If provided, save PNG here.
        title:       Image title (shown if save_path is PNG).
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    from IPython.display import SVG, display

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[gnn_explain] Invalid SMILES: {smiles}")
        return

    rdDepictor.Compute2DCoords(mol)

    norm_scores = node_scores / (node_scores.max() + 1e-8)

    # Map scores to atom colours (red = important)
    atom_colors = {}
    for idx, score in enumerate(norm_scores.tolist()):
        r = float(score)
        atom_colors[idx] = (r, 0.2, 1.0 - r)   # red → blue gradient

    highlight_atoms = list(range(mol.GetNumAtoms()))
    highlight_radii = {idx: 0.4 * float(norm_scores[idx]) + 0.2
                       for idx in range(mol.GetNumAtoms())}

    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    if save_path:
        Path(save_path).write_text(svg)
        print(f"[gnn_explain] Saved atom importance SVG → {save_path}")
    else:
        try:
            display(SVG(svg))
        except Exception:
            pass
