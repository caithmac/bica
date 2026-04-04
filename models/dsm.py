"""
DualBind DSM Auxiliary Head — Denoising Score Matching.

Reference: DualBind (combines supervised affinity regression with unsupervised
denoising score matching). The DSM head forces the encoder to learn a smooth
binding energy surface by predicting the score (gradient of log-density) of
corrupted embeddings.

DSM objective:
    L_DSM = E_{σ~p(σ), ε~N(0,σ²I)} [ ||score(z + ε, σ) + ε/σ²||² ]

where z is the encoder embedding, σ is the noise level, and score() is a small
MLP that predicts ε/σ² (the Stein score of the noisy embedding distribution).

Total loss:
    L = L_MSE + lambda_dsm * L_DSM

Usage:
    # In training loop (train_torch or run_bica):
    from models.dsm import DSMAuxHead, dsm_loss
    dsm_head = DSMAuxHead(embed_dim=256)
    # After encoder forward:
    z = model.encode(x)
    loss = mse_criterion(model.head(z), y) + lambda_dsm * dsm_loss(dsm_head, z)

Model interface requirement:
    Models using DSM must expose encode() and head() methods.
    This module also provides encode/head splits for BiCA and MLP.

Noise schedule:
    5 geometric levels: σ ∈ {0.01, 0.05, 0.1, 0.5, 1.0}
    A random σ is sampled per batch.
"""

import torch
import torch.nn as nn
import numpy as np

# Geometric noise schedule (5 levels)
SIGMA_MIN = 0.01
SIGMA_MAX = 1.0
N_SIGMA   = 5
SIGMAS    = torch.tensor(
    np.geomspace(SIGMA_MIN, SIGMA_MAX, N_SIGMA), dtype=torch.float32
)


class DSMAuxHead(nn.Module):
    """
    Score network s_θ(z_noisy, σ) ≈ -ε/σ² (Stein score).

    Takes a noisy embedding and noise level, predicts the score.
    Small MLP: 2 hidden layers, same dim as embedding.

    Args:
        embed_dim:  dimension of the encoder embedding
        hidden_dim: hidden dimension of the score MLP (default = embed_dim)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        # +1 for log(σ) conditioning
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z_noisy: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_noisy:   (B, embed_dim) noisy embedding
            log_sigma: (B, 1) log of noise level (for conditioning)
        Returns:
            score: (B, embed_dim) predicted score
        """
        inp = torch.cat([z_noisy, log_sigma], dim=-1)
        return self.net(inp)


def dsm_loss(
    dsm_head:   DSMAuxHead,
    z:          torch.Tensor,   # (B, embed_dim) clean embedding
    lambda_dsm: float = 0.1,
    sigmas:     torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute the DSM auxiliary loss for a batch of embeddings.

    Samples one σ from SIGMAS uniformly, corrupts z with N(0, σ²I) noise,
    and trains the score head to predict -ε/σ².

    Args:
        dsm_head:   DSMAuxHead instance
        z:          (B, D) clean encoder embeddings (detached from affinity path
                    is NOT required — gradients flow through both)
        lambda_dsm: loss weight
        sigmas:     noise levels tensor (default: module-level SIGMAS)

    Returns:
        Scalar DSM loss (already weighted by lambda_dsm).
    """
    if sigmas is None:
        sigmas = SIGMAS.to(z.device)

    # Sample one noise level for the whole batch
    sigma_idx = torch.randint(len(sigmas), (1,)).item()
    sigma = sigmas[sigma_idx]                                   # scalar

    # Corrupt embedding
    eps = torch.randn_like(z) * sigma                          # (B, D)
    z_noisy = (z + eps).detach()  # stop-gradient on input to score net

    # Condition on log(σ)
    log_sigma = torch.full((z.size(0), 1), sigma.log().item(),
                           device=z.device, dtype=z.dtype)

    # Predicted score vs. target score (-ε / σ²)
    score_pred   = dsm_head(z_noisy, log_sigma)                # (B, D)
    score_target = -eps / (sigma ** 2)                         # (B, D)

    loss = ((score_pred - score_target) ** 2).sum(dim=-1).mean()
    return lambda_dsm * loss
