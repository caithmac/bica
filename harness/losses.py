"""
Auxiliary loss functions for Phase 1 improvements.

PairwiseRankingLoss
-------------------
Samples K pairs (i, j) per batch where y_i > y_j + margin, then penalises
predictions that violate the ranking: pred_i <= pred_j.

Loss = mean over violating pairs of (pred_j - pred_i + margin).clamp(min=0)

This is a margin-based pairwise ranking loss (Ranknet-style) applied to
binding affinity regression.  It directly optimises Spearman rank
correlation in addition to the MSE criterion.

Usage in train_torch / train_gnn_model:
    from harness.losses import PairwiseRankingLoss
    rank_loss = PairwiseRankingLoss(margin=0.5, n_pairs=32, lambda_rank=0.1)
    # Inside the training step:
    loss = mse_criterion(pred, y) + rank_loss(pred, y)

Reference: Multi-task Bioassay Pre-training (MBP) — within-assay ranking
signal as auxiliary objective alongside regression.
"""

import torch
import torch.nn as nn


class PairwiseRankingLoss(nn.Module):
    """
    Margin-based pairwise ranking loss.

    Args:
        margin      (float): minimum pKd difference to form a pair (default 0.5).
                             Pairs where |y_i - y_j| < margin are skipped —
                             avoids penalising ambiguous near-ties.
        n_pairs     (int):   number of pairs to sample per batch (default 32).
                             Capped at the number of valid pairs available.
        lambda_rank (float): weight applied to ranking loss before adding to
                             main MSE loss (default 0.1).
    """

    def __init__(self, margin: float = 0.5, n_pairs: int = 32,
                 lambda_rank: float = 0.1):
        super().__init__()
        self.margin      = margin
        self.n_pairs     = n_pairs
        self.lambda_rank = lambda_rank

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B,) or (B,1) predicted pKd
            y:    (B,) or (B,1) ground-truth pKd
        Returns:
            Scalar ranking loss (already weighted by lambda_rank).
        """
        pred = pred.view(-1)
        y    = y.view(-1)
        B    = y.size(0)

        if B < 2:
            return pred.sum() * 0.0  # zero, keeps grad graph alive

        # Build all (i, j) pairs where y_i > y_j + margin
        with torch.no_grad():
            diff = y.unsqueeze(0) - y.unsqueeze(1)   # (B, B): diff[i,j] = y_i - y_j
            valid_i, valid_j = torch.where(diff > self.margin)

        n_valid = valid_i.size(0)
        if n_valid == 0:
            return pred.sum() * 0.0

        # Sub-sample if more pairs than n_pairs
        if n_valid > self.n_pairs:
            idx = torch.randperm(n_valid, device=pred.device)[:self.n_pairs]
            valid_i = valid_i[idx]
            valid_j = valid_j[idx]

        # Ranking violation: we want pred_i > pred_j; penalise if not
        rank_loss = torch.clamp(
            pred[valid_j] - pred[valid_i] + self.margin, min=0.0
        ).mean()

        return self.lambda_rank * rank_loss
