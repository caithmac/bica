"""
Sequence models that consume token IDs directly.
These are paired with the tokenizers in harness/tokenizers.py.

All models take:
  lig_ids:   (B, L_lig)   int64
  prot_ids:  (B, L_prot)  int64
  lig_mask:  (B, L_lig)   float32   (1=real token, 0=pad)
  prot_mask: (B, L_prot)  float32

And return: (B, 1) scalar predictions.

Models in this file:
  - LSTMBindingModel     — bidirectional LSTM encoder for each sequence, concat + MLP
  - TransformerSeqModel  — learned-embedding transformer for each sequence, concat + MLP
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared components
# ─────────────────────────────────────────────────────────────────────────────

class MeanPool(nn.Module):
    """Masked mean pooling over sequence dimension."""
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), mask: (B, L)
        mask = mask.unsqueeze(-1)                        # (B, L, 1)
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bidirectional LSTM
# ─────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.2, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = MeanPool()
        self.out_dim = hidden_dim * 2   # bidirectional

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)          # (B, L, E)
        out, _ = self.lstm(x)            # (B, L, 2H) — plain padded forward, GPU-safe
        return self.pool(out, mask)      # masked mean over real tokens only


class LSTMBindingModel(nn.Module):
    """
    Dual BiLSTM: separate encoders for ligand and protein, concat, MLP head.
    """
    def __init__(
        self,
        lig_vocab_size: int,
        prot_vocab_size: int,
        lig_embed_dim: int   = 64,
        prot_embed_dim: int  = 64,
        hidden_dim: int      = 128,
        num_layers: int      = 2,
        dropout: float       = 0.2,
        lig_pad_id: int      = 0,
        prot_pad_id: int     = 0,
    ):
        super().__init__()
        self.lig_enc  = LSTMEncoder(lig_vocab_size,  lig_embed_dim,  hidden_dim,
                                    num_layers, dropout, lig_pad_id)
        self.prot_enc = LSTMEncoder(prot_vocab_size, prot_embed_dim, hidden_dim,
                                    num_layers, dropout, prot_pad_id)
        self.head = RegressionHead(
            in_dim  = self.lig_enc.out_dim + self.prot_enc.out_dim,
            hidden  = 256,
            dropout = dropout,
        )

    def forward(self, lig_ids, prot_ids, lig_mask, prot_mask) -> torch.Tensor:
        lig_repr  = self.lig_enc(lig_ids,  lig_mask)
        prot_repr = self.prot_enc(prot_ids, prot_mask)
        return self.head(torch.cat([lig_repr, prot_repr], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transformer with learned embeddings
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, nhead: int,
                 num_layers: int, dim_ff: int, dropout: float, pad_id: int,
                 max_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool   = MeanPool()
        self.out_dim = embed_dim

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos   = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)
        x     = self.embedding(ids) + self.pos_embedding(pos)
        # TransformerEncoder expects src_key_padding_mask: True = IGNORE
        pad_mask = (mask == 0)  # (B, L) bool
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.pool(x, mask)


class TransformerSeqModel(nn.Module):
    """
    Dual transformer encoder: separate for ligand and protein, concat, MLP head.
    """
    def __init__(
        self,
        lig_vocab_size: int,
        prot_vocab_size: int,
        lig_embed_dim: int   = 128,
        prot_embed_dim: int  = 128,
        nhead: int           = 4,
        num_layers: int      = 2,
        dim_ff: int          = 256,
        dropout: float       = 0.1,
        lig_pad_id: int      = 0,
        prot_pad_id: int     = 0,
        max_lig_len: int     = 256,
        max_prot_len: int    = 1024,
    ):
        super().__init__()
        self.lig_enc = TransformerEncoder(
            lig_vocab_size,  lig_embed_dim,  nhead, num_layers, dim_ff,
            dropout, lig_pad_id,  max_len=max_lig_len,
        )
        self.prot_enc = TransformerEncoder(
            prot_vocab_size, prot_embed_dim, nhead, num_layers, dim_ff,
            dropout, prot_pad_id, max_len=max_prot_len,
        )
        self.head = RegressionHead(
            in_dim  = lig_embed_dim + prot_embed_dim,
            hidden  = 256,
            dropout = dropout,
        )

    def forward(self, lig_ids, prot_ids, lig_mask, prot_mask) -> torch.Tensor:
        lig_repr  = self.lig_enc(lig_ids,  lig_mask)
        prot_repr = self.prot_enc(prot_ids, prot_mask)
        return self.head(torch.cat([lig_repr, prot_repr], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mamba SSM Encoder (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_mamba():
    """Returns mamba_ssm.Mamba class or None if not installed."""
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError:
        return None


class MambaBlock(nn.Module):
    """Single Mamba or BiLSTM fallback block."""

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        Mamba = _try_import_mamba()
        if Mamba is not None:
            self.layer = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
            self.is_mamba = True
        else:
            import warnings
            warnings.warn(
                "mamba-ssm not installed — MambaEncoder falling back to BiLSTM.",
                ImportWarning, stacklevel=2,
            )
            self.layer = nn.LSTM(d_model, d_model // 2, batch_first=True,
                                 bidirectional=True)
            self.is_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_mamba:
            return self.layer(x)
        out, _ = self.layer(x)
        return out


class MaskedMeanPool(nn.Module):
    """Mean-pool over non-padding positions."""

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return (x * mask.float().unsqueeze(-1)).sum(dim=1) / lengths


class MambaEncoder(nn.Module):
    """
    Sequence encoder using Mamba SSM blocks (or BiLSTM fallback).

    vocab → embed → project → N×MambaBlock → MaskedMeanPool → (B, hidden_dim)

    Args:
        vocab_size:  tokeniser vocabulary size
        embed_dim:   embedding dimension
        hidden_dim:  hidden/output dimension (MambaBlock d_model)
        n_layers:    number of Mamba/BiLSTM blocks
        dropout:     dropout after each block
        pad_id:      padding token id (default 0)
        d_state:     Mamba state dimension
        d_conv:      Mamba convolution width
        expand:      Mamba expansion factor
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        n_layers:   int = 4,
        dropout:    float = 0.1,
        pad_id:     int = 0,
        d_state:    int = 16,
        d_conv:     int = 4,
        expand:     int = 2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.proj   = nn.Linear(embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_dim, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.drops  = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.pool   = MaskedMeanPool()
        self.norm   = nn.LayerNorm(hidden_dim)

    def forward(self, ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            ids:  (B, L) token ids
            mask: (B, L) 1=real 0=pad (optional; derived from pad_id if None)
        Returns:
            (B, hidden_dim) sequence embedding
        """
        if mask is None:
            mask = (ids != self.pad_id).long()
        x = self.proj(self.embed(ids))     # (B, L, hidden_dim)
        for block, drop in zip(self.blocks, self.drops):
            x = drop(block(x))
        x = self.norm(x)
        return self.pool(x, mask)          # (B, hidden_dim)


class MambaBindingModel(nn.Module):
    """
    Dual MambaEncoder for binding affinity prediction.
    Same interface as LSTMBindingModel — drop-in replacement.

    Args:
        lig_vocab_size:  ligand tokeniser vocab size
        prot_vocab_size: protein tokeniser vocab size
        lig_embed_dim:   ligand embedding dim
        prot_embed_dim:  protein embedding dim
        hidden_dim:      shared Mamba hidden dim
        n_layers:        number of Mamba blocks per encoder
        dropout:         dropout rate
        lig_pad_id:      ligand padding token id
        prot_pad_id:     protein padding token id
    """

    def __init__(
        self,
        lig_vocab_size:  int,
        prot_vocab_size: int,
        lig_embed_dim:   int = 128,
        prot_embed_dim:  int = 128,
        hidden_dim:      int = 256,
        n_layers:        int = 4,
        dropout:         float = 0.1,
        lig_pad_id:      int = 0,
        prot_pad_id:     int = 0,
    ):
        super().__init__()
        self.lig_enc = MambaEncoder(
            vocab_size=lig_vocab_size, embed_dim=lig_embed_dim,
            hidden_dim=hidden_dim, n_layers=n_layers,
            dropout=dropout, pad_id=lig_pad_id,
        )
        self.prot_enc = MambaEncoder(
            vocab_size=prot_vocab_size, embed_dim=prot_embed_dim,
            hidden_dim=hidden_dim, n_layers=n_layers,
            dropout=dropout, pad_id=prot_pad_id,
        )
        self.head = RegressionHead(
            in_dim  = hidden_dim * 2,
            hidden  = 256,
            dropout = dropout,
        )

    def forward(self, lig_ids: torch.Tensor, prot_ids: torch.Tensor,
                lig_mask: torch.Tensor | None = None,
                prot_mask: torch.Tensor | None = None) -> torch.Tensor:
        lig_repr  = self.lig_enc(lig_ids,  lig_mask)
        prot_repr = self.prot_enc(prot_ids, prot_mask)
        return self.head(torch.cat([lig_repr, prot_repr], dim=1))
