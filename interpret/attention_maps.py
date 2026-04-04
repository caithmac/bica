"""
Cross-attention heatmap extraction for BiCA and GLI models.

For each test sample, extracts the protein×ligand attention weight matrix
and produces:
  - Per-sample heatmaps (residue on Y axis, ligand atom/token on X axis)
  - Averaged heatmap across all test samples
  - Per-residue and per-atom importance scores (row/col mean of attention)

Usage:
    from interpret.attention_maps import extract_bica_attention, plot_attention_heatmap

    # Single sample
    attn_dict = extract_bica_attention(model, lig_tensor, prot_tensor)
    plot_attention_heatmap(
        attn_dict["protein_to_ligand"],   # (n_heads, L_prot, L_lig)
        row_labels=residue_ids,
        col_labels=atom_ids,
        save_path="attn_sample0.png",
    )
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def extract_bica_attention(
    model,
    protein_feat: "torch.Tensor",   # (1, D_prot) or (1, L_prot, D_prot)
    ligand_feat:  "torch.Tensor",   # (1, D_lig)  or (1, L_lig,  D_lig)
    protein_mask: "torch.Tensor | None" = None,
    ligand_mask:  "torch.Tensor | None" = None,
) -> dict[str, "np.ndarray"]:
    """
    Forward pass through BiCA with return_attention=True.

    Args:
        model:        BiCA_VariableHeads instance (eval mode).
        protein_feat: (1, …) protein feature tensor.
        ligand_feat:  (1, …) ligand feature tensor.
        protein_mask: optional (1, L_prot) 1=real 0=pad.
        ligand_mask:  optional (1, L_lig)  1=real 0=pad.

    Returns:
        Dict with keys:
          "protein_to_ligand": (n_heads, L_prot, L_lig) ndarray
          "ligand_to_protein": (n_heads, L_lig,  L_prot) ndarray
          "prot_importance":   (L_prot,) mean attention weight per residue
          "lig_importance":    (L_lig,)  mean attention weight per atom/token
    """
    import torch
    model.eval()
    with torch.no_grad():
        _, attn = model(
            protein_feat, ligand_feat,
            protein_mask=protein_mask, ligand_mask=ligand_mask,
            return_attention=True,
        )

    p2l = attn["protein_to_ligand"].squeeze(0).cpu().numpy()   # (H, L_p, L_l)
    l2p = attn["ligand_to_protein"].squeeze(0).cpu().numpy()   # (H, L_l, L_p)

    # Mean over heads → (L_p, L_l) and (L_l, L_p)
    p2l_mean = p2l.mean(axis=0)
    l2p_mean = l2p.mean(axis=0)

    prot_importance = p2l_mean.mean(axis=1)   # (L_p,) — how much each residue attends to ligand
    lig_importance  = l2p_mean.mean(axis=1)   # (L_l,) — how much each atom attends to protein

    return {
        "protein_to_ligand": p2l,
        "ligand_to_protein": l2p,
        "prot_importance":   prot_importance,
        "lig_importance":    lig_importance,
    }


def extract_gli_attention(
    model,
    x:          "torch.Tensor",    # (N_total, node_dim) batched graph nodes
    edge_index: "torch.Tensor",    # (2, E_total)
    batch:      "torch.Tensor",    # (N_total,)
    prot_vec:   "torch.Tensor",    # (B, prot_dim)
    mol_idx:    int = 0,
) -> dict[str, "np.ndarray"]:
    """
    Extract per-atom importance scores from GLI's cross-attention branch.

    Hooks into GLI's p2l_attn and l2p_attn to capture attention weights
    for molecule `mol_idx` in the batch.

    Returns:
        Dict with keys:
          "p2l_attn_weights": (n_heads, 1, n_atoms) — protein→atom attention
          "l2p_attn_weights": (n_heads, n_atoms, 1) — atom→protein attention
          "atom_importance":  (n_atoms,) — combined per-atom importance
    """
    import torch

    captured = {}

    def _hook_p2l(module, input, output):
        # output = (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            w = output[1]
            if w is not None:
                captured["p2l"] = w.detach().cpu().numpy()

    def _hook_l2p(module, input, output):
        if isinstance(output, tuple) and len(output) == 2:
            w = output[1]
            if w is not None:
                captured["l2p"] = w.detach().cpu().numpy()

    h1 = model.p2l_attn.register_forward_hook(_hook_p2l)
    h2 = model.l2p_attn.register_forward_hook(_hook_l2p)

    model.eval()
    with torch.no_grad():
        model(x, edge_index, batch, prot_vec)

    h1.remove()
    h2.remove()

    p2l_w = captured.get("p2l")   # (1, 1, n_atoms) or (1, n_heads, 1, n_atoms)
    l2p_w = captured.get("l2p")   # (1, n_atoms, 1)

    if p2l_w is None or l2p_w is None:
        return {}

    # MultiheadAttention returns (batch, tgt_len, src_len) when average_attn_weights=True
    # Squeeze batch dim
    p2l_w = p2l_w.squeeze(0)   # (1, n_atoms) or (n_heads, 1, n_atoms)
    l2p_w = l2p_w.squeeze(0)   # (n_atoms, 1)

    atom_importance = (p2l_w.mean() if p2l_w.ndim == 1 else p2l_w.mean(axis=0)).ravel()
    atom_importance = atom_importance + l2p_w.ravel()

    return {
        "p2l_attn_weights": p2l_w,
        "l2p_attn_weights": l2p_w,
        "atom_importance":  atom_importance,
    }


def batch_attention_importance(
    model,
    data_loader,
    device: str | None = None,
    model_type: str = "bica",   # "bica" | "gli"
    max_samples: int = 500,
) -> dict[str, "np.ndarray"]:
    """
    Aggregate per-sample importance scores across many samples.

    For BiCA: returns mean protein residue importance and ligand token importance.
    For GLI:  returns mean per-atom importance (averaged across same-size mols).

    Args:
        model:       BiCA or GLI model (eval mode expected).
        data_loader: Yields batches in the model-appropriate format.
        device:      Device string (auto-detects if None).
        model_type:  "bica" or "gli".
        max_samples: Stop after this many samples.

    Returns:
        {"prot_importance": (D_p,), "lig_importance": (D_l,)}
        (averaged importance vectors)
    """
    import torch

    if device is None:
        device = next(model.parameters()).device
    model.eval()

    prot_scores, lig_scores = [], []
    n_seen = 0

    for batch in data_loader:
        if n_seen >= max_samples:
            break
        if model_type == "bica":
            lig_b, prot_b, _ = batch
            B = lig_b.size(0)
            for i in range(B):
                if n_seen >= max_samples:
                    break
                attn = extract_bica_attention(
                    model,
                    prot_b[i:i+1].to(device),
                    lig_b[i:i+1].to(device),
                )
                prot_scores.append(attn["prot_importance"])
                lig_scores.append(attn["lig_importance"])
                n_seen += 1

    result = {}
    if prot_scores:
        result["prot_importance"] = np.stack(prot_scores, axis=0).mean(axis=0)
    if lig_scores:
        result["lig_importance"] = np.stack(lig_scores, axis=0).mean(axis=0)
    return result


def plot_attention_heatmap(
    attn_weights: "np.ndarray",    # (n_heads, L_rows, L_cols) or (L_rows, L_cols)
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    head_idx: int | str = "mean",  # which head or "mean"
    title: str = "Attention weights",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """
    Plot a single attention heatmap (residues × atoms/tokens).

    Args:
        attn_weights: Attention tensor, (H, R, C) or (R, C).
        row_labels:   Optional Y-axis labels (residues).
        col_labels:   Optional X-axis labels (atoms/tokens).
        head_idx:     "mean" to average over heads, or int to select one head.
        title:        Plot title.
        save_path:    If provided, save figure here (PNG).
        figsize:      (width, height) in inches.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if attn_weights.ndim == 3:
        if head_idx == "mean":
            mat = attn_weights.mean(axis=0)
        else:
            mat = attn_weights[int(head_idx)]
    else:
        mat = attn_weights   # (R, C)

    R, C = mat.shape
    if figsize is None:
        figsize = (max(6, C * 0.3), max(4, R * 0.3))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax)

    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6)
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=6)

    ax.set_xlabel("Ligand atoms / tokens")
    ax.set_ylabel("Protein residues")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[attention] Saved heatmap → {save_path}")
    plt.close(fig)
