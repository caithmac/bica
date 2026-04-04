"""
Transformer-based models for binding affinity prediction.

Model A: Dual-stream transformer (separate encoders for ligand & protein,
          then cross-attention fusion).
Model B: Concat-then-transformer (concatenate token sequences, single encoder).
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerRegressor(nn.Module):
    """
    Lightweight transformer for flat features.
    Interprets the flat vector as a sequence of tokens for self-attention.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int     = 256,
        nhead: int       = 8,
        num_layers: int  = 3,
        dim_ff: int      = 512,
        dropout: float   = 0.1,
        token_size: int  = 64,   # group input dims into tokens of this size
    ):
        super().__init__()
        # Pad input if needed
        self.token_size = token_size
        pad_to = math.ceil(input_dim / token_size) * token_size
        self.pad_size = pad_to - input_dim
        self.n_tokens = pad_to // token_size

        self.token_proj = nn.Linear(token_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=self.n_tokens + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model))

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if self.pad_size > 0:
            x = torch.cat([x, x.new_zeros(B, self.pad_size)], dim=1)
        # Reshape to (B, n_tokens, token_size)
        x = x.view(B, self.n_tokens, self.token_size)
        x = self.token_proj(x)   # (B, n_tokens, d_model)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_enc(x)
        x   = self.transformer(x)
        return self.head(x[:, 0])   # use CLS representation


class CrossAttentionFusion(nn.Module):
    """
    Dual-stream: separate linear encoders for ligand and protein,
    then cross-attention before regression head.
    """
    def __init__(
        self,
        lig_dim: int,
        prot_dim: int,
        d_model: int    = 256,
        nhead: int      = 8,
        num_layers: int = 2,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.lig_proj  = nn.Linear(lig_dim,  d_model)
        self.prot_proj = nn.Linear(prot_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = concatenation of lig and prot features — we need separate dims
        # This model is driven from run_experiment with separate feats
        raise NotImplementedError("Use forward_separate()")

    def forward_separate(self, lig: torch.Tensor, prot: torch.Tensor) -> torch.Tensor:
        """
        lig:  (B, lig_dim)
        prot: (B, prot_dim)
        """
        lig  = self.lig_proj(lig ).unsqueeze(1)   # (B, 1, d_model)
        prot = self.prot_proj(prot).unsqueeze(1)  # (B, 1, d_model)
        out  = self.cross_attn(lig, prot)         # query=lig, key/value=prot
        return self.head(out.squeeze(1))
