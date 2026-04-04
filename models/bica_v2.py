"""
BiCA v2 — Enhanced Bidirectional Cross-Attention for binding affinity prediction.

Key improvements over v1:
  1. TRUE SEQUENCE INPUTS — protein as (B, L_prot, 480) per-residue ESM-2 tokens,
     ligand as (B, L_lig, 78) per-atom graph features. Real attention over sequences,
     not degenerate single-token attention.
  2. STACKED CROSS-ATTENTION — 2 bidirectional cross-attention layers (depth matters).
  3. LARGER HIDDEN DIM — 512 default (vs 256) for richer representations.
  4. LEARNED ATTENTION POOLING — gated attention instead of mean pool (weights
     important residues / atoms more heavily).
  5. FEED-FORWARD BLOCK — post-attention FFN (GELU + residual) like a full transformer.
  6. BETTER PREDICTOR — 3-layer MLP with GELU and layer norm instead of batch norm
     (layer norm works at inference batch_size=1, batch norm does not).

Ablation variants (for paper):
  BiCA_v2               — full model (stacked, attention pool, FFN)
  BiCA_v2_MeanPool      — replace attention pool with mean pool
  BiCA_v2_SingleLayer   — only 1 cross-attention layer (depth ablation)
  BiCA_v2_NoFFN         — no FFN block (FFN ablation)
  BiCA_v2_NoResidual    — no residual connections
  BiCA_v2_P2L_only      — unidirectional protein→ligand only
  BiCA_v2_L2P_only      — unidirectional ligand→protein only
  SimpleConcatBaseline  — same projections/MLP but no attention (fair baseline)

All variants share the same forward signature:
  forward(protein_seq, ligand_seq, protein_mask=None, ligand_mask=None,
          return_attention=False) → (B,1) or ((B,1), attn_dict)

Factory:
  build_bica_v2(protein_dim, ligand_dim, **kwargs) → BiCA_v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """
    Learned attention pooling: produces a weighted sum over sequence positions.
    Importance weights are computed by a small MLP: score_i = w^T tanh(W h_i).
    Masked positions receive -inf before softmax.

    Args:
        hidden_dim: input feature dimension
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (B, L, H) sequence hidden states
            mask: (B, L) 1=real 0=pad (optional)
        Returns:
            (B, H) pooled representation
        """
        scores = self.score(x).squeeze(-1)          # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)          # (B, L)
        weights = weights.unsqueeze(-1)              # (B, L, 1)
        return (weights * x).sum(dim=1)             # (B, H)


class DropPath(nn.Module):
    """
    Stochastic depth (DropPath) — drops entire residual branches during training.
    More effective than dropout for transformer residual streams.
    At inference (eval mode) acts as identity.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep)
        return x * random_tensor / keep


class CrossAttentionBlock(nn.Module):
    """
    One bidirectional cross-attention layer with optional FFN.

    Performs (Pre-LN style):
      P' = P + DropPath(MHA(norm(P) queries norm(L)))
      L' = L + DropPath(MHA(norm(L) queries norm(P)))
      optionally followed by a position-wise FFN on each.

    Args:
        hidden_dim: embedding dimension
        num_heads:  MHA heads (must divide hidden_dim)
        ffn_dim:    FFN hidden dimension (default = 4 × hidden_dim)
        dropout:    attention/FFN dropout rate
        drop_path:  stochastic depth drop probability (0 = disabled)
        use_ffn:    whether to include FFN block
    """
    def __init__(self, hidden_dim: int, num_heads: int,
                 ffn_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 drop_path: float = 0.1,
                 use_ffn: bool = True):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.p2l = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.l2p = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)

        # Pre-LN: normalise BEFORE attention (more stable for long sequences)
        self.norm_p1 = nn.LayerNorm(hidden_dim)
        self.norm_l1 = nn.LayerNorm(hidden_dim)

        self.drop      = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        self.use_ffn   = use_ffn

        if use_ffn:
            fdim = ffn_dim or hidden_dim * 4
            self.ffn_p = nn.Sequential(
                nn.Linear(hidden_dim, fdim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fdim, hidden_dim),
                nn.Dropout(dropout),
            )
            self.ffn_l = nn.Sequential(
                nn.Linear(hidden_dim, fdim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fdim, hidden_dim),
                nn.Dropout(dropout),
            )
            self.norm_p2 = nn.LayerNorm(hidden_dim)
            self.norm_l2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        protein_h: torch.Tensor,              # (B, L_p, H)
        ligand_h:  torch.Tensor,              # (B, L_l, H)
        p_key_mask: Optional[torch.Tensor],   # (B, L_p) True=IGNORE
        l_key_mask: Optional[torch.Tensor],   # (B, L_l) True=IGNORE
        return_weights: bool = False,
    ):
        # ── Pre-LN Cross-attention + DropPath residual ─────────────────────
        p2l_out, p2l_w = self.p2l(
            query=self.norm_p1(protein_h), key=self.norm_l1(ligand_h),
            value=self.norm_l1(ligand_h),
            key_padding_mask=l_key_mask,
            need_weights=return_weights, average_attn_weights=False,
        )
        l2p_out, l2p_w = self.l2p(
            query=self.norm_l1(ligand_h), key=self.norm_p1(protein_h),
            value=self.norm_p1(protein_h),
            key_padding_mask=p_key_mask,
            need_weights=return_weights, average_attn_weights=False,
        )

        # DropPath on residual (drops entire attention branch per sample)
        protein_h = protein_h + self.drop_path(self.drop(p2l_out))
        ligand_h  = ligand_h  + self.drop_path(self.drop(l2p_out))

        # ── Pre-LN FFN + DropPath ─────────────────────────────────────────
        if self.use_ffn:
            protein_h = protein_h + self.drop_path(self.ffn_p(self.norm_p2(protein_h)))
            ligand_h  = ligand_h  + self.drop_path(self.ffn_l(self.norm_l2(ligand_h)))

        return protein_h, ligand_h, p2l_w, l2p_w


# ─────────────────────────────────────────────────────────────────────────────
# BiCA v2 — Full model
# ─────────────────────────────────────────────────────────────────────────────

class BiCA_v2(nn.Module):
    """
    BiCA v2: Stacked bidirectional cross-attention with learned attention pooling.

    Designed for TRUE SEQUENCE INPUTS:
      protein_seq:  (B, L_prot, protein_dim)  — per-residue ESM-2 (not mean-pooled)
      ligand_seq:   (B, L_lig,  ligand_dim)   — per-atom features (RDKit/GNN hidden)
    Also accepts flat vectors (B, D) which are unsqueezed to (B, 1, D) as fallback.

    Args:
        protein_dim:  input dimension of protein tokens
        ligand_dim:   input dimension of ligand tokens/atoms
        hidden_dim:   internal projection dimension (default 512)
        num_heads:    attention heads (default 8; must divide hidden_dim)
        num_layers:   number of stacked CrossAttentionBlocks (default 2)
        dropout:      dropout probability
        use_ffn:      include FFN blocks in each cross-attention layer
        pool_type:    "attention" (learned) or "mean"
    """
    def __init__(
        self,
        protein_dim:  int   = 480,
        ligand_dim:   int   = 78,
        hidden_dim:   int   = 512,
        num_heads:    int   = 8,
        num_layers:   int   = 2,
        dropout:      float = 0.1,
        drop_path:    float = 0.1,
        use_ffn:      bool  = True,
        pool_type:    str   = "attention",
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        assert pool_type in ("attention", "mean")
        assert num_layers >= 1

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.num_layers = num_layers
        self.pool_type  = pool_type

        # Input projections
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.ligand_proj  = nn.Linear(ligand_dim,  hidden_dim)
        self.norm_p_in    = nn.LayerNorm(hidden_dim)
        self.norm_l_in    = nn.LayerNorm(hidden_dim)

        # Stacked cross-attention layers — linearly scale drop_path across layers
        dp_rates = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads,
                                ffn_dim=hidden_dim * 4,
                                dropout=dropout,
                                drop_path=dp_rates[i],
                                use_ffn=use_ffn)
            for i in range(num_layers)
        ])

        # Pooling
        if pool_type == "attention":
            self.prot_pool = AttentionPool(hidden_dim)
            self.lig_pool  = AttentionPool(hidden_dim)
        else:
            self.prot_pool = None
            self.lig_pool  = None

        # Post-pool projection
        self.prot_proj2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lig_proj2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Predictor: layer norm instead of batch norm for inference stability
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def _pool(self, x: torch.Tensor,
              mask: Optional[torch.Tensor],
              pooler: Optional[AttentionPool]) -> torch.Tensor:
        if pooler is not None:
            return pooler(x, mask)
        if mask is not None:
            lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            return (x * mask.float().unsqueeze(-1)).sum(1) / lengths
        return x.mean(dim=1)

    def forward(
        self,
        protein_seq:  torch.Tensor,             # (B, L_p, D_p) or (B, D_p)
        ligand_seq:   torch.Tensor,             # (B, L_l, D_l) or (B, D_l)
        protein_mask: Optional[torch.Tensor] = None,   # (B, L_p) 1=real 0=pad
        ligand_mask:  Optional[torch.Tensor] = None,   # (B, L_l) 1=real 0=pad
        return_attention: bool = False,
    ):
        # Ensure 3D
        if protein_seq.dim() == 2:
            protein_seq  = protein_seq.unsqueeze(1)
            protein_mask = None
        if ligand_seq.dim() == 2:
            ligand_seq  = ligand_seq.unsqueeze(1)
            ligand_mask = None

        # Project + norm
        ph = self.norm_p_in(self.protein_proj(protein_seq))   # (B, L_p, H)
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))     # (B, L_l, H)

        # MHA key padding masks: True = ignore position
        p_kpm = (~protein_mask.bool()) if protein_mask is not None else None
        l_kpm = (~ligand_mask.bool())  if ligand_mask  is not None else None

        # Stacked cross-attention
        all_p2l_w, all_l2p_w = [], []
        for layer in self.layers:
            ph, lh, p2l_w, l2p_w = layer(
                ph, lh, p_kpm, l_kpm,
                return_weights=return_attention,
            )
            if return_attention:
                all_p2l_w.append(p2l_w)
                all_l2p_w.append(l2p_w)

        # Pool
        p_pooled = self._pool(ph, protein_mask, self.prot_pool)   # (B, H)
        l_pooled = self._pool(lh, ligand_mask,  self.lig_pool)    # (B, H)

        # Post-pool projection
        p_pooled = self.prot_proj2(p_pooled)
        l_pooled = self.lig_proj2(l_pooled)

        # Predict
        combined = torch.cat([p_pooled, l_pooled], dim=-1)        # (B, 2H)
        out      = self.predictor(combined)                        # (B, 1)

        if return_attention:
            return out, {
                "protein_to_ligand": all_p2l_w,   # list of (B, H, L_p, L_l) per layer
                "ligand_to_protein": all_l2p_w,
            }
        return out

    def encode(
        self,
        protein_seq:  torch.Tensor,
        ligand_seq:   torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, hidden_dim*2) pre-predictor embedding. Used by DSM."""
        if protein_seq.dim() == 2:
            protein_seq  = protein_seq.unsqueeze(1)
            protein_mask = None
        if ligand_seq.dim() == 2:
            ligand_seq  = ligand_seq.unsqueeze(1)
            ligand_mask = None

        ph = self.norm_p_in(self.protein_proj(protein_seq))
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))
        p_kpm = (~protein_mask.bool()) if protein_mask is not None else None
        l_kpm = (~ligand_mask.bool())  if ligand_mask  is not None else None

        for layer in self.layers:
            ph, lh, _, _ = layer(ph, lh, p_kpm, l_kpm, return_weights=False)

        p_pooled = self._pool(ph, protein_mask, self.prot_pool)
        l_pooled = self._pool(lh, ligand_mask,  self.lig_pool)
        p_pooled = self.prot_proj2(p_pooled)
        l_pooled = self.lig_proj2(l_pooled)
        return torch.cat([p_pooled, l_pooled], dim=-1)

    def predict_from_embedding(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variants (all share the same forward signature)
# ─────────────────────────────────────────────────────────────────────────────

class BiCA_v2_MeanPool(BiCA_v2):
    """Ablation: mean pooling instead of learned attention pooling."""
    def __init__(self, **kwargs):
        kwargs["pool_type"] = "mean"
        super().__init__(**kwargs)


class BiCA_v2_SingleLayer(BiCA_v2):
    """Ablation: single cross-attention layer (depth ablation)."""
    def __init__(self, **kwargs):
        kwargs["num_layers"] = 1
        super().__init__(**kwargs)


class BiCA_v2_NoFFN(BiCA_v2):
    """Ablation: no FFN block in cross-attention layers."""
    def __init__(self, **kwargs):
        kwargs["use_ffn"] = False
        super().__init__(**kwargs)


class BiCA_v2_NoResidual(nn.Module):
    """
    Ablation: no residual connections in cross-attention.
    Reimplemented directly (can't inherit and just toggle residuals).
    """
    def __init__(self, protein_dim=480, ligand_dim=78, hidden_dim=512,
                 num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim

        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.ligand_proj  = nn.Linear(ligand_dim,  hidden_dim)
        self.norm_p_in    = nn.LayerNorm(hidden_dim)
        self.norm_l_in    = nn.LayerNorm(hidden_dim)

        self.p2l  = nn.MultiheadAttention(hidden_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.l2p  = nn.MultiheadAttention(hidden_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm_p = nn.LayerNorm(hidden_dim)
        self.norm_l = nn.LayerNorm(hidden_dim)

        self.prot_proj2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.lig_proj2  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(self, protein_seq, ligand_seq,
                protein_mask=None, ligand_mask=None, return_attention=False):
        if protein_seq.dim() == 2:
            protein_seq = protein_seq.unsqueeze(1); protein_mask = None
        if ligand_seq.dim() == 2:
            ligand_seq = ligand_seq.unsqueeze(1);   ligand_mask  = None

        ph = self.norm_p_in(self.protein_proj(protein_seq))
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))
        p_kpm = (~protein_mask.bool()) if protein_mask is not None else None
        l_kpm = (~ligand_mask.bool())  if ligand_mask  is not None else None

        p2l_out, p2l_w = self.p2l(ph, lh, lh, key_padding_mask=l_kpm,
                                   need_weights=return_attention, average_attn_weights=False)
        l2p_out, l2p_w = self.l2p(lh, ph, ph, key_padding_mask=p_kpm,
                                   need_weights=return_attention, average_attn_weights=False)

        # NO residual — just norm
        ph = self.norm_p(p2l_out)
        lh = self.norm_l(l2p_out)

        p_pooled = ph.mean(dim=1)
        l_pooled = lh.mean(dim=1)
        p_pooled = self.prot_proj2(p_pooled)
        l_pooled = self.lig_proj2(l_pooled)
        out = self.predictor(torch.cat([p_pooled, l_pooled], dim=-1))

        if return_attention:
            return out, {"protein_to_ligand": [p2l_w], "ligand_to_protein": [l2p_w]}
        return out

    def encode(self, protein_seq, ligand_seq, protein_mask=None, ligand_mask=None):
        if protein_seq.dim() == 2:
            protein_seq = protein_seq.unsqueeze(1); protein_mask = None
        if ligand_seq.dim() == 2:
            ligand_seq = ligand_seq.unsqueeze(1);   ligand_mask  = None
        ph = self.norm_p_in(self.protein_proj(protein_seq))
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))
        p_kpm = (~protein_mask.bool()) if protein_mask is not None else None
        l_kpm = (~ligand_mask.bool())  if ligand_mask  is not None else None
        p2l_out, _ = self.p2l(ph, lh, lh, key_padding_mask=l_kpm, need_weights=False)
        l2p_out, _ = self.l2p(lh, ph, ph, key_padding_mask=p_kpm, need_weights=False)
        ph = self.norm_p(p2l_out)
        lh = self.norm_l(l2p_out)
        p_pooled = self.prot_proj2(ph.mean(dim=1))
        l_pooled = self.lig_proj2(lh.mean(dim=1))
        return torch.cat([p_pooled, l_pooled], dim=-1)

    def predict_from_embedding(self, z):
        return self.predictor(z)


class BiCA_v2_P2L_only(nn.Module):
    """Ablation: only protein→ligand attention (unidirectional)."""
    def __init__(self, protein_dim=480, ligand_dim=78, hidden_dim=512,
                 num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.ligand_proj  = nn.Linear(ligand_dim,  hidden_dim)
        self.norm_p_in    = nn.LayerNorm(hidden_dim)
        self.norm_l_in    = nn.LayerNorm(hidden_dim)
        self.p2l = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm_p = nn.LayerNorm(hidden_dim)
        self.drop   = nn.Dropout(dropout)
        self.prot_pool = AttentionPool(hidden_dim)
        self.lig_pool  = AttentionPool(hidden_dim)
        self.prot_proj2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.lig_proj2  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(self, protein_seq, ligand_seq,
                protein_mask=None, ligand_mask=None, return_attention=False):
        if protein_seq.dim() == 2:
            protein_seq = protein_seq.unsqueeze(1); protein_mask = None
        if ligand_seq.dim() == 2:
            ligand_seq = ligand_seq.unsqueeze(1);   ligand_mask  = None
        ph = self.norm_p_in(self.protein_proj(protein_seq))
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))
        l_kpm = (~ligand_mask.bool()) if ligand_mask is not None else None
        p2l_out, p2l_w = self.p2l(ph, lh, lh, key_padding_mask=l_kpm,
                                   need_weights=return_attention, average_attn_weights=False)
        ph = self.norm_p(ph + self.drop(p2l_out))
        p_pooled = self.prot_proj2(self.prot_pool(ph, protein_mask))
        l_pooled = self.lig_proj2(self.lig_pool(lh, ligand_mask))
        out = self.predictor(torch.cat([p_pooled, l_pooled], dim=-1))
        if return_attention:
            return out, {"protein_to_ligand": [p2l_w], "ligand_to_protein": [None]}
        return out

    def encode(self, protein_seq, ligand_seq, protein_mask=None, ligand_mask=None):
        if protein_seq.dim() == 2: protein_seq = protein_seq.unsqueeze(1); protein_mask = None
        if ligand_seq.dim() == 2:  ligand_seq  = ligand_seq.unsqueeze(1);  ligand_mask  = None
        ph = self.norm_p_in(self.protein_proj(protein_seq))
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))
        l_kpm = (~ligand_mask.bool()) if ligand_mask is not None else None
        p2l_out, _ = self.p2l(ph, lh, lh, key_padding_mask=l_kpm, need_weights=False)
        ph = self.norm_p(ph + self.drop(p2l_out))
        return torch.cat([self.prot_proj2(self.prot_pool(ph, protein_mask)),
                          self.lig_proj2(self.lig_pool(lh, ligand_mask))], dim=-1)

    def predict_from_embedding(self, z):
        return self.predictor(z)


class BiCA_v2_L2P_only(nn.Module):
    """Ablation: only ligand→protein attention (unidirectional)."""
    def __init__(self, protein_dim=480, ligand_dim=78, hidden_dim=512,
                 num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.ligand_proj  = nn.Linear(ligand_dim,  hidden_dim)
        self.norm_p_in    = nn.LayerNorm(hidden_dim)
        self.norm_l_in    = nn.LayerNorm(hidden_dim)
        self.l2p = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm_l = nn.LayerNorm(hidden_dim)
        self.drop   = nn.Dropout(dropout)
        self.prot_pool = AttentionPool(hidden_dim)
        self.lig_pool  = AttentionPool(hidden_dim)
        self.prot_proj2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.lig_proj2  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(self, protein_seq, ligand_seq,
                protein_mask=None, ligand_mask=None, return_attention=False):
        if protein_seq.dim() == 2:
            protein_seq = protein_seq.unsqueeze(1); protein_mask = None
        if ligand_seq.dim() == 2:
            ligand_seq  = ligand_seq.unsqueeze(1);  ligand_mask  = None
        ph = self.norm_p_in(self.protein_proj(protein_seq))
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))
        p_kpm = (~protein_mask.bool()) if protein_mask is not None else None
        l2p_out, l2p_w = self.l2p(lh, ph, ph, key_padding_mask=p_kpm,
                                   need_weights=return_attention, average_attn_weights=False)
        lh = self.norm_l(lh + self.drop(l2p_out))
        p_pooled = self.prot_proj2(self.prot_pool(ph, protein_mask))
        l_pooled = self.lig_proj2(self.lig_pool(lh, ligand_mask))
        out = self.predictor(torch.cat([p_pooled, l_pooled], dim=-1))
        if return_attention:
            return out, {"protein_to_ligand": [None], "ligand_to_protein": [l2p_w]}
        return out

    def encode(self, protein_seq, ligand_seq, protein_mask=None, ligand_mask=None):
        if protein_seq.dim() == 2: protein_seq = protein_seq.unsqueeze(1); protein_mask = None
        if ligand_seq.dim() == 2:  ligand_seq  = ligand_seq.unsqueeze(1);  ligand_mask  = None
        ph = self.norm_p_in(self.protein_proj(protein_seq))
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))
        p_kpm = (~protein_mask.bool()) if protein_mask is not None else None
        l2p_out, _ = self.l2p(lh, ph, ph, key_padding_mask=p_kpm, need_weights=False)
        lh = self.norm_l(lh + self.drop(l2p_out))
        return torch.cat([self.prot_proj2(self.prot_pool(ph, protein_mask)),
                          self.lig_proj2(self.lig_pool(lh, ligand_mask))], dim=-1)

    def predict_from_embedding(self, z):
        return self.predictor(z)


class SimpleConcatBaseline(nn.Module):
    """
    Fair ablation baseline: same projections + same predictor MLP as BiCA_v2,
    but NO attention — just mean pool and concatenate.
    Proves that attention is doing real work (not just extra parameters).
    """
    def __init__(self, protein_dim=480, ligand_dim=78, hidden_dim=512,
                 dropout=0.1, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.ligand_proj  = nn.Linear(ligand_dim,  hidden_dim)
        self.norm_p_in    = nn.LayerNorm(hidden_dim)
        self.norm_l_in    = nn.LayerNorm(hidden_dim)
        self.prot_proj2   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.lig_proj2    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.predictor    = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def _pool(self, x, mask):
        if mask is not None:
            lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            return (x * mask.float().unsqueeze(-1)).sum(1) / lengths
        return x.mean(dim=1)

    def forward(self, protein_seq, ligand_seq,
                protein_mask=None, ligand_mask=None, return_attention=False):
        if protein_seq.dim() == 2: protein_seq = protein_seq.unsqueeze(1); protein_mask = None
        if ligand_seq.dim() == 2:  ligand_seq  = ligand_seq.unsqueeze(1);  ligand_mask  = None
        ph = self.norm_p_in(self.protein_proj(protein_seq))
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))
        p_pooled = self.prot_proj2(self._pool(ph, protein_mask))
        l_pooled = self.lig_proj2(self._pool(lh, ligand_mask))
        out = self.predictor(torch.cat([p_pooled, l_pooled], dim=-1))
        if return_attention:
            return out, {}
        return out

    def encode(self, protein_seq, ligand_seq, protein_mask=None, ligand_mask=None):
        if protein_seq.dim() == 2: protein_seq = protein_seq.unsqueeze(1); protein_mask = None
        if ligand_seq.dim() == 2:  ligand_seq  = ligand_seq.unsqueeze(1);  ligand_mask  = None
        ph = self.norm_p_in(self.protein_proj(protein_seq))
        lh = self.norm_l_in(self.ligand_proj(ligand_seq))
        return torch.cat([self.prot_proj2(self._pool(ph, protein_mask)),
                          self.lig_proj2(self._pool(lh, ligand_mask))], dim=-1)

    def predict_from_embedding(self, z):
        return self.predictor(z)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

BICA_V2_VARIANTS = {
    "bica_v2":             BiCA_v2,
    "bica_v2_meanpool":    BiCA_v2_MeanPool,
    "bica_v2_singlelayer": BiCA_v2_SingleLayer,
    "bica_v2_noffn":       BiCA_v2_NoFFN,
    "bica_v2_noresidual":  BiCA_v2_NoResidual,
    "bica_v2_p2l":         BiCA_v2_P2L_only,
    "bica_v2_l2p":         BiCA_v2_L2P_only,
    "concat_baseline":     SimpleConcatBaseline,
}


def build_bica_v2(
    protein_dim: int,
    ligand_dim:  int,
    variant:     str   = "bica_v2",
    hidden_dim:  int   = 512,
    num_heads:   int   = 8,
    num_layers:  int   = 2,
    dropout:     float = 0.1,
    drop_path:   float = 0.1,
) -> nn.Module:
    """
    Factory for run_experiment.py.

    Args:
        protein_dim: input protein token dim (e.g. 480 for ESM-2 35M per-residue)
        ligand_dim:  input ligand atom dim   (e.g. 78 for RDKit atom features)
        variant:     one of BICA_V2_VARIANTS keys (default: "bica_v2")
        hidden_dim:  internal dim (auto-adjusted to be divisible by num_heads)
        num_heads:   attention heads
        num_layers:  cross-attention layers (ignored for non-stacked variants)
        dropout:     dropout rate
        drop_path:   stochastic depth rate (linearly scaled across layers)

    Returns:
        Instantiated model.
    """
    if variant not in BICA_V2_VARIANTS:
        raise ValueError(f"Unknown BiCA v2 variant: {variant}. "
                         f"Choose from: {list(BICA_V2_VARIANTS.keys())}")

    # Ensure divisibility
    while hidden_dim % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    cls = BICA_V2_VARIANTS[variant]
    # SimpleConcatBaseline doesn't use drop_path — pass only accepted kwargs
    import inspect
    sig = inspect.signature(cls.__init__)
    kwargs = dict(protein_dim=protein_dim, ligand_dim=ligand_dim,
                  hidden_dim=hidden_dim, num_heads=num_heads,
                  num_layers=num_layers, dropout=dropout)
    if "drop_path" in sig.parameters:
        kwargs["drop_path"] = drop_path
    return cls(**kwargs)
