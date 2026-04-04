"""
Integrated Gradients (IG) for sequence models (LSTM, Mamba, TransformerSeq).

Reference: Sundararajan et al., ICML 2017 (Integrated Gradients).

IG attributes the prediction to input embeddings by integrating gradients
along a straight path from a baseline (zero / PAD embedding) to the input.

  IG_i = (x_i - x'_i) * ∫_0^1 ∂F(x' + α(x-x')) / ∂x_i dα

Approximated with Riemann sum (n_steps=50 is typically sufficient).

Provides:
  ig_for_seq_model(model, lig_ids, prot_ids, lig_mask, prot_mask, ...)
    → {"lig_ig": (L_lig,), "prot_ig": (L_prot,)}

  plot_ig_scores(scores, token_labels, title, save_path)
    → bar plot of per-token IG importance

Usage:
    from interpret.captum_ig import ig_for_seq_model, plot_ig_scores

    result = ig_for_seq_model(model, lig_ids, prot_ids, lig_mask, prot_mask)
    plot_ig_scores(result["lig_ig"], token_labels=smiles_tokens,
                   title="Ligand IG", save_path="lig_ig.png")
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def _interpolate_inputs(
    baseline: "torch.Tensor",
    inp:      "torch.Tensor",
    alpha:    float,
) -> "torch.Tensor":
    return baseline + alpha * (inp - baseline)


def ig_for_seq_model(
    model,
    lig_ids:   "torch.Tensor",    # (1, L_lig)  long
    prot_ids:  "torch.Tensor",    # (1, L_prot) long
    lig_mask:  "torch.Tensor | None" = None,
    prot_mask: "torch.Tensor | None" = None,
    n_steps:   int = 50,
    baseline_token: int = 0,      # PAD token id (used as baseline)
) -> dict[str, np.ndarray]:
    """
    Compute Integrated Gradients attributions for a dual-encoder sequence model.

    The attribution is computed in **embedding space** rather than token ID
    space (token IDs are discrete; IG requires a continuous path).

    Baseline = all-zeros embedding (i.e. the PAD embedding or a zero vector).

    Args:
        model:          LSTMBindingModel / MambaBindingModel / TransformerSeqModel.
                        Must expose lig_enc and prot_enc sub-encoders with
                        an .embed attribute.
        lig_ids:        (1, L_lig)  token id tensor for one ligand.
        prot_ids:       (1, L_prot) token id tensor for one protein.
        lig_mask:       (1, L_lig)  1=real 0=pad; derived from lig_ids if None.
        prot_mask:      (1, L_prot) 1=real 0=pad; derived from prot_ids if None.
        n_steps:        Number of interpolation steps for Riemann sum.
        baseline_token: Token id used as the PAD/zero baseline.

    Returns:
        Dict with:
          "lig_ig":  (L_lig,)  L2 norm of IG attribution per ligand token
          "prot_ig": (L_prot,) L2 norm of IG attribution per protein token
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device

    lig_ids  = lig_ids.to(device)
    prot_ids = prot_ids.to(device)

    if lig_mask is None:
        lig_mask  = (lig_ids != baseline_token).long()
    if prot_mask is None:
        prot_mask = (prot_ids != baseline_token).long()

    # Get embeddings for the actual tokens
    lig_emb  = model.lig_enc.embed(lig_ids).detach()    # (1, L_l, E)
    prot_emb = model.prot_enc.embed(prot_ids).detach()  # (1, L_p, E)

    # Baseline: zero embedding
    lig_base  = torch.zeros_like(lig_emb)
    prot_base = torch.zeros_like(prot_emb)

    lig_grads  = torch.zeros_like(lig_emb)
    prot_grads = torch.zeros_like(prot_emb)

    # Monkey-patch forward to accept embeddings instead of ids
    # We call the encoder components manually
    for step in range(1, n_steps + 1):
        alpha = step / n_steps

        lig_inp  = _interpolate_inputs(lig_base,  lig_emb,  alpha).requires_grad_(True)
        prot_inp = _interpolate_inputs(prot_base, prot_emb, alpha).requires_grad_(True)

        # Project and run through encoder (bypassing embed layer)
        lig_h  = _encode_from_embedding(model.lig_enc,  lig_inp,  lig_mask.to(device))
        prot_h = _encode_from_embedding(model.prot_enc, prot_inp, prot_mask.to(device))

        out  = model.head(torch.cat([lig_h, prot_h], dim=1))  # (1, 1)
        out.sum().backward()

        lig_grads  = lig_grads  + lig_inp.grad.detach()
        prot_grads = prot_grads + prot_inp.grad.detach()

    # Riemann approximation: (x - x') * mean_grad
    lig_ig_vec  = ((lig_emb - lig_base)  * lig_grads  / n_steps).squeeze(0)   # (L_l, E)
    prot_ig_vec = ((prot_emb - prot_base) * prot_grads / n_steps).squeeze(0)  # (L_p, E)

    # Reduce to per-token scalar via L2 norm
    lig_ig  = lig_ig_vec.norm(dim=-1).cpu().numpy()    # (L_l,)
    prot_ig = prot_ig_vec.norm(dim=-1).cpu().numpy()   # (L_p,)

    return {"lig_ig": lig_ig, "prot_ig": prot_ig}


def _encode_from_embedding(
    encoder,
    emb:  "torch.Tensor",    # (B, L, E) already-embedded, grad-tracked
    mask: "torch.Tensor",    # (B, L) 1=real 0=pad
) -> "torch.Tensor":
    """
    Run an encoder's post-embedding layers on a pre-computed embedding.
    Supports LSTMEncoder, TransformerEncoder, MambaEncoder.
    """
    import torch

    # Project from embed_dim to hidden_dim
    if hasattr(encoder, "proj"):
        x = encoder.proj(emb)
    else:
        x = emb

    # LSTM path
    if hasattr(encoder, "lstm"):
        out, _ = encoder.lstm(x)
        # Masked mean pool
        lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        pooled  = (out * mask.float().unsqueeze(-1)).sum(dim=1) / lengths
        return pooled

    # Transformer path
    if hasattr(encoder, "transformer"):
        # Create key_padding_mask: True = ignore
        key_pad = ~mask.bool()
        out = encoder.transformer(x.transpose(0, 1),
                                  src_key_padding_mask=key_pad)
        out = out.transpose(0, 1)
        lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        pooled  = (out * mask.float().unsqueeze(-1)).sum(dim=1) / lengths
        return pooled

    # Mamba / block path
    if hasattr(encoder, "blocks"):
        for block, drop in zip(encoder.blocks, encoder.drops):
            x = drop(block(x))
        x = encoder.norm(x)
        pooled = encoder.pool(x, mask)
        return pooled

    # Generic fallback: mean pool
    lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
    return (x * mask.float().unsqueeze(-1)).sum(dim=1) / lengths


def plot_ig_scores(
    scores: np.ndarray,
    token_labels: list[str] | None = None,
    title: str = "Integrated Gradients attribution",
    max_tokens: int = 60,
    save_path: str | Path | None = None,
) -> None:
    """
    Bar chart of per-token IG importance.

    Args:
        scores:       (L,) IG scores per token.
        token_labels: Optional list of L token strings.
        title:        Plot title.
        max_tokens:   Truncate display at this many tokens.
        save_path:    If provided, save PNG here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = min(len(scores), max_tokens)
    scores_ = scores[:L]
    labels_ = token_labels[:L] if token_labels else [str(i) for i in range(L)]

    fig, ax = plt.subplots(figsize=(max(8, L * 0.2), 4))
    x_pos = np.arange(L)
    ax.bar(x_pos, scores_, color="steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_, rotation=90, fontsize=7)
    ax.set_ylabel("IG attribution (L2)")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[captum_ig] Saved IG plot → {save_path}")
    plt.close(fig)
