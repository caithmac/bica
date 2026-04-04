"""
GLI: Gated Global-Local Interaction model for binding affinity prediction.

Reference: Inspired by "Joint Global and Local Interaction Modeling for
Drug-Target Affinity Prediction" (GLI framework).

Architecture:
  Local branch  — GNN per-atom hidden states (encode_nodes, no pooling)
                  global-mean-pooled → mol_repr (hidden_dim,)
  Global branch — BiCA bidirectional cross-attention between:
                    protein vector (unsqueezed to seq_len=1)
                    GNN per-atom hidden states (seq_len = n_atoms)
                  → p2l + l2p pooled → bica_repr (hidden_dim,)
  Gate          — g = sigmoid(W_g * [mol_repr; bica_repr])
  Fused         — fused = g ⊙ mol_repr + (1-g) ⊙ bica_repr
  Head          — MLP(fused) → scalar pKd

Motivation:
  BiCA on flat protein vectors (seq_len=1) degenerates to a linear transform.
  GNNs pool molecules but cannot model protein-atom interaction.
  GLI combines both: the gate lets the model learn which branch to trust.

Forward signature: (x, edge_index, batch, prot_vec) — same as GCN/GAT.
Works with existing GNNDataset / gnn_trainer.py / run_gnn() with no changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLIBindingModel(nn.Module):
    """
    Gated Global-Local Interaction model.

    Args:
        node_dim:   atom node feature dimension (78)
        hidden_dim: shared hidden dimension for GNN + BiCA
        prot_dim:   protein vector dimension
        gnn_type:   "gcn" or "gat" for the local GNN branch
        num_heads:  cross-attention heads in BiCA branch
        dropout:    dropout rate
        num_gnn_layers: depth of GNN in local branch
    """

    def __init__(
        self,
        node_dim:       int = 78,
        hidden_dim:     int = 128,
        prot_dim:       int = 320,
        gnn_type:       str = "gat",
        num_heads:      int = 4,
        dropout:        float = 0.2,
        num_gnn_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Local branch: GNN ─────────────────────────────────────────────────
        if gnn_type == "gcn":
            from torch_geometric.nn import GCNConv
            self.convs = nn.ModuleList()
            self.bns   = nn.ModuleList()
            in_dim = node_dim
            for _ in range(num_gnn_layers):
                self.convs.append(GCNConv(in_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                in_dim = hidden_dim
            self.gnn_type = "gcn"
        else:   # gat (default)
            from torch_geometric.nn import GATConv
            self.convs = nn.ModuleList()
            self.bns   = nn.ModuleList()
            in_dim = node_dim
            out_per_head = hidden_dim // num_heads
            for i in range(num_gnn_layers):
                self.convs.append(
                    GATConv(in_dim, out_per_head, heads=num_heads,
                            dropout=dropout, concat=True))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                in_dim = hidden_dim
            self.gnn_type = "gat"

        self.dropout = nn.Dropout(dropout)

        # ── Global branch: BiCA cross-attention ───────────────────────────────
        # Protein vector projected to hidden_dim (will be seq_len=1)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)

        # Atom hidden states already in hidden_dim (from GNN)
        # BiCA: protein attends to atoms (p2l) and atoms attend to protein (l2p)
        self.p2l_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.l2p_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)

        # ── Gating mechanism ─────────────────────────────────────────────────
        # g = sigmoid(W_g * [mol_repr; bica_repr])  → (hidden_dim,)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Prediction head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _gnn_encode_nodes(self, x, edge_index):
        """GNN forward without pooling — returns per-node hidden states."""
        for conv, bn in zip(self.convs, self.bns):
            if self.gnn_type == "gcn":
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
            else:
                x = conv(x, edge_index)
                x = bn(x)
                x = F.elu(x)
            x = self.dropout(x)
        return x  # (N, hidden_dim)

    def forward(
        self,
        x:          torch.Tensor,   # (N_total, node_dim)
        edge_index: torch.Tensor,   # (2, E_total)
        batch:      torch.Tensor,   # (N_total,) molecule indices
        prot_vec:   torch.Tensor,   # (B, prot_dim)
    ) -> torch.Tensor:
        """Returns (B, 1) predicted pKd."""
        from torch_geometric.nn import global_mean_pool

        B = prot_vec.size(0)
        device = x.device

        # ── Local branch: GNN per-node hidden states ─────────────────────────
        h_nodes = self._gnn_encode_nodes(x, edge_index)   # (N_total, hidden_dim)
        mol_repr = global_mean_pool(h_nodes, batch)        # (B, hidden_dim)

        # ── Global branch: BiCA per molecule ─────────────────────────────────
        prot_h = self.prot_proj(prot_vec)                  # (B, hidden_dim)

        bica_reprs = []
        for mol_idx in range(B):
            mask    = (batch == mol_idx)
            h_mol   = h_nodes[mask]                        # (n_atoms, hidden_dim)
            n_atoms = h_mol.size(0)

            # Unsqueeze protein to (1, 1, hidden_dim) for MHA batch_first
            p_q = prot_h[mol_idx].unsqueeze(0).unsqueeze(0)    # (1, 1, H)
            a_kv = h_mol.unsqueeze(0)                           # (1, n_atoms, H)

            # Protein queries atom keys/values
            p2l_out, _ = self.p2l_attn(p_q, a_kv, a_kv)       # (1, 1, H)
            p2l_pooled = p2l_out.squeeze(0).squeeze(0)         # (H,)

            # Atom queries protein keys/values
            l2p_out, _ = self.l2p_attn(a_kv, p_q, p_q)        # (1, n_atoms, H)
            l2p_pooled = l2p_out.squeeze(0).mean(0)            # (H,)

            bica_repr = p2l_pooled + l2p_pooled                # (H,)
            bica_reprs.append(bica_repr)

        bica_batch = torch.stack(bica_reprs, dim=0)            # (B, hidden_dim)

        # ── Gated fusion ─────────────────────────────────────────────────────
        gate_input = torch.cat([mol_repr, bica_batch], dim=-1) # (B, 2*H)
        g = torch.sigmoid(self.gate(gate_input))               # (B, H)
        fused = g * mol_repr + (1.0 - g) * bica_batch         # (B, H)

        return self.head(fused)                                # (B, 1)
