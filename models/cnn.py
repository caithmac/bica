"""
1D-CNN models operating on sequence token embeddings.
Used with character-level SMILES or amino acid sequences.
"""

import torch
import torch.nn as nn
import numpy as np


class CNN1D(nn.Module):
    """
    1-D CNN over embedded sequences.
    Input: (batch, seq_len, embed_dim) — will be transposed internally to
           (batch, embed_dim, seq_len) for conv layers.
    """
    def __init__(
        self,
        input_dim: int,        # flat input dim (seq_len * embed_dim when pre-flattened)
        seq_len: int,
        embed_dim: int,
        num_filters: int = 128,
        kernel_sizes=(3, 5, 7),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.embed_dim = embed_dim

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),   # global max pooling
            )
            for k in kernel_sizes
        ])

        combined_dim = num_filters * len(kernel_sizes)
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len * embed_dim)
        x = x.view(x.size(0), self.seq_len, self.embed_dim)
        x = x.transpose(1, 2)            # (batch, embed_dim, seq_len)
        pooled = [conv(x).squeeze(-1) for conv in self.convs]
        x = torch.cat(pooled, dim=1)
        return self.fc(x)


def build_smiles_cnn(smiles_max_len: int = 100, input_dim: int | None = None,
                     vocab_size: int | None = None) -> CNN1D:
    """
    CNN for character-level SMILES one-hot encoding.
    Pass input_dim (ligand feature dim only, before protein concat) to derive
    vocab_size automatically, avoiding hardcoded mismatches.
    """
    if input_dim is not None:
        vocab_size = input_dim // smiles_max_len
    elif vocab_size is None:
        vocab_size = 35   # actual deduped vocab size
    return CNN1D(
        input_dim  = smiles_max_len * vocab_size,
        seq_len    = smiles_max_len,
        embed_dim  = vocab_size,
        num_filters= 128,
        kernel_sizes=(3, 5, 9),
        dropout    = 0.3,
    )
