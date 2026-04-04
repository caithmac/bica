"""
MLP models of varying depth/width for the benchmark.
All accept a flat feature vector as input.
"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Configurable MLP with BatchNorm, Dropout, and GELU activations.

    The architecture is split into:
      self.encoder — all layers except the final Linear(last_hidden, 1)
      self.output  — the final Linear(last_hidden, 1)

    This enables DSM: encode() returns the pre-output embedding,
    predict_from_embedding() applies the output layer.
    forward() is unchanged for all existing experiments.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
    ):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.output  = nn.Linear(in_dim, 1)
        # Kept for backward compatibility
        self.net = nn.Sequential(*encoder_layers, nn.Linear(in_dim, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns pre-output embedding (B, last_hidden_dim). Used by DSM."""
        return self.encoder(x)

    def predict_from_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """Apply output layer to embedding z."""
        return self.output(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mlp_shallow(input_dim: int) -> MLP:
    """Shallow: 1 hidden layer, 256 units."""
    return MLP(input_dim, [256], dropout=0.2)


def mlp_medium(input_dim: int) -> MLP:
    """Medium: 3 hidden layers 512-256-128."""
    return MLP(input_dim, [512, 256, 128], dropout=0.2)


def mlp_deep(input_dim: int) -> MLP:
    """Deep: 5 hidden layers 1024-512-256-128-64."""
    return MLP(input_dim, [1024, 512, 256, 128, 64], dropout=0.3)


def mlp_wide(input_dim: int) -> MLP:
    """Wide: 2 hidden layers 2048-1024."""
    return MLP(input_dim, [2048, 1024], dropout=0.3)
