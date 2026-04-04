"""
BiCA — Bidirectional Cross-Attention model for binding affinity prediction.

Original architecture by user. Bugs fixed:
  1. Return shape: predictor output kept as (B,1) — consistent with MSELoss in trainer.
  2. key_padding_mask convention: PyTorch MHA expects True=IGNORE (padding positions).
     Caller must pass mask where 1=real, 0=pad; we invert internally.
  3. protein_seq_dim / ligand_seq_dim are now constructor args — works with any repr.
  4. Flat vectors (ECFP, ChemBERTa, ESM-2 mean-pool) are unsqueezed to (B,1,D)
     so the attention layers see a "sequence of length 1". This is valid — MHA
     over a single token degenerates to a learned linear transform + residual,
     which is harmless. For true sequence inputs (atom graphs, token sequences)
     pass seq_len > 1.

Two entry points:
  BiCA_VariableHeads  — the original architecture, bugs fixed
  build_bica()        — factory used by run_experiment.py
"""

import torch
import torch.nn as nn


class BiCA_VariableHeads(nn.Module):
    """
    Bidirectional Cross-Attention model.
    Protein and ligand representations attend to each other before pooling.

    Args:
        protein_seq_dim: feature dim of each protein token/position
        ligand_seq_dim:  feature dim of each ligand token/position
        hidden_dim:      internal projection dimension
        num_heads:       number of attention heads (must divide hidden_dim)
        dropout:         dropout probability
    """
    def __init__(
        self,
        protein_seq_dim: int = 480,
        ligand_seq_dim: int  = 82,
        hidden_dim: int      = 256,
        num_heads: int       = 32,
        dropout: float       = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads

        self.protein_proj = nn.Linear(protein_seq_dim, hidden_dim)
        self.ligand_proj  = nn.Linear(ligand_seq_dim,  hidden_dim)

        # Protein queries, Ligand keys/values
        self.protein2ligand = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # Ligand queries, Protein keys/values
        self.ligand2protein = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)

        self.protein_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ligand_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        protein_seq: torch.Tensor,   # (B, L_prot, protein_seq_dim) or (B, protein_seq_dim)
        ligand_atoms: torch.Tensor,  # (B, L_lig,  ligand_seq_dim)  or (B, ligand_seq_dim)
        protein_mask: torch.Tensor | None = None,  # (B, L_prot) 1=real 0=pad
        ligand_mask:  torch.Tensor | None = None,  # (B, L_lig)  1=real 0=pad
        return_attention: bool = False,
    ) -> torch.Tensor:

        # ── Ensure 3D: (B, seq_len, dim) ──────────────────────────────────────
        if protein_seq.dim() == 2:
            protein_seq  = protein_seq.unsqueeze(1)   # (B, 1, D)
            protein_mask = None                        # single token — no padding
        if ligand_atoms.dim() == 2:
            ligand_atoms = ligand_atoms.unsqueeze(1)
            ligand_mask  = None

        # ── Project to hidden_dim ─────────────────────────────────────────────
        protein_h = self.norm1(self.protein_proj(protein_seq))    # (B, L_p, H)
        ligand_h  = self.norm2(self.ligand_proj(ligand_atoms))    # (B, L_l, H)

        # ── Convert masks: 1=real → True=real; MHA needs True=IGNORE (pad) ───
        # So we pass ~(mask.bool()) as key_padding_mask
        p_key_mask = (~protein_mask.bool()) if protein_mask is not None else None
        l_key_mask = (~ligand_mask.bool())  if ligand_mask  is not None else None

        # ── P→L cross-attention: protein queries attend to ligand ─────────────
        p2l_out, p2l_weights = self.protein2ligand(
            query=protein_h, key=ligand_h, value=ligand_h,
            key_padding_mask=l_key_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        p2l_out = self.norm3(p2l_out + protein_h)   # residual
        p2l_out = self.dropout(p2l_out)

        # ── L→P cross-attention: ligand queries attend to protein ─────────────
        l2p_out, l2p_weights = self.ligand2protein(
            query=ligand_h, key=protein_h, value=protein_h,
            key_padding_mask=p_key_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        l2p_out = self.norm4(l2p_out + ligand_h)    # residual
        l2p_out = self.dropout(l2p_out)

        # ── Masked mean pool ──────────────────────────────────────────────────
        if protein_mask is not None:
            lengths = protein_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            p2l_pooled = (p2l_out * protein_mask.float().unsqueeze(-1)).sum(1) / lengths
        else:
            p2l_pooled = p2l_out.mean(dim=1)

        if ligand_mask is not None:
            lengths = ligand_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            l2p_pooled = (l2p_out * ligand_mask.float().unsqueeze(-1)).sum(1) / lengths
        else:
            l2p_pooled = l2p_out.mean(dim=1)

        p2l_pooled = self.protein_pool(p2l_pooled)
        l2p_pooled = self.ligand_pool(l2p_pooled)

        # ── Predict ───────────────────────────────────────────────────────────
        combined = torch.cat([p2l_pooled, l2p_pooled], dim=-1)   # (B, H*2)
        affinity  = self.predictor(combined)                      # (B, 1)  ← kept as (B,1)

        if return_attention:
            return affinity, {
                "protein_to_ligand": p2l_weights,
                "ligand_to_protein": l2p_weights,
            }
        return affinity

    def encode(
        self,
        protein_seq:  torch.Tensor,
        ligand_atoms: torch.Tensor,
        protein_mask: torch.Tensor | None = None,
        ligand_mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns the pre-predictor embedding (B, hidden_dim*2).
        Used by DSM auxiliary loss to access the latent representation.
        """
        if protein_seq.dim() == 2:
            protein_seq  = protein_seq.unsqueeze(1)
            protein_mask = None
        if ligand_atoms.dim() == 2:
            ligand_atoms = ligand_atoms.unsqueeze(1)
            ligand_mask  = None

        protein_h = self.norm1(self.protein_proj(protein_seq))
        ligand_h  = self.norm2(self.ligand_proj(ligand_atoms))

        p_key_mask = (~protein_mask.bool()) if protein_mask is not None else None
        l_key_mask = (~ligand_mask.bool())  if ligand_mask  is not None else None

        p2l_out, _ = self.protein2ligand(
            query=protein_h, key=ligand_h, value=ligand_h,
            key_padding_mask=l_key_mask, need_weights=False)
        p2l_out = self.norm3(p2l_out + protein_h)
        p2l_out = self.dropout(p2l_out)

        l2p_out, _ = self.ligand2protein(
            query=ligand_h, key=protein_h, value=protein_h,
            key_padding_mask=p_key_mask, need_weights=False)
        l2p_out = self.norm4(l2p_out + ligand_h)
        l2p_out = self.dropout(l2p_out)

        if protein_mask is not None:
            lengths = protein_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            p2l_pooled = (p2l_out * protein_mask.float().unsqueeze(-1)).sum(1) / lengths
        else:
            p2l_pooled = p2l_out.mean(dim=1)

        if ligand_mask is not None:
            lengths = ligand_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            l2p_pooled = (l2p_out * ligand_mask.float().unsqueeze(-1)).sum(1) / lengths
        else:
            l2p_pooled = l2p_out.mean(dim=1)

        p2l_pooled = self.protein_pool(p2l_pooled)
        l2p_pooled = self.ligand_pool(l2p_pooled)
        return torch.cat([p2l_pooled, l2p_pooled], dim=-1)   # (B, H*2)

    def predict_from_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the predictor head to a pre-computed embedding z (B, H*2)."""
        return self.predictor(z)


def build_bica(protein_dim: int, ligand_dim: int,
               hidden_dim: int = 256, num_heads: int = 16,
               dropout: float = 0.2) -> BiCA_VariableHeads:
    """
    Factory for run_experiment.py.
    Automatically picks num_heads that divides hidden_dim.
    """
    # Ensure num_heads divides hidden_dim
    while hidden_dim % num_heads != 0 and num_heads > 1:
        num_heads -= 1
    return BiCA_VariableHeads(
        protein_seq_dim=protein_dim,
        ligand_seq_dim=ligand_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
