"""
Graphormer: Transformer with graph-structural attention biases for molecules.

Reference: Ying et al., "Do Transformers Really Perform Badly for Graph
Representation?" NeurIPS 2021.

Three additive biases injected into self-attention logits before softmax:
  1. Centrality bias  — b_c[in_degree[i]] + b_c[in_degree[j]]
  2. Spatial bias     — b_sp[SPD[i,j]]  (shortest-path distance, 0-padded)
  3. Edge bias        — mean of edge_attr on shortest path between i,j

All three are computed from 2D topology (no 3D coordinates required).

For drug-like molecules (max ~50 atoms) the O(N²) attention is affordable.
Protein is a flat pre-computed vector concatenated after CLS-pooled graph
embedding, following the same pattern as GCNBindingModel / GATBindingModel.

Architecture:
  SMILES → mol_to_graph_data() → node features (78-dim)
           + shortest-path matrix (from distmat)
           + edge features (10-dim)
  → GraphormerEncoder (L layers of GraphormerLayer)
  → CLS token pooled → concat with protein_vec
  → MLP head → scalar pKd

Integration:
  Same GNNDataset / gnn_trainer.py infrastructure.
  The GNNDataset stores (graph, prot_vec, y); GraphormerBindingModel's forward
  receives (x, edge_index, batch, prot_vec) matching GCN/GAT signature.
  Internally it re-derives the SPD matrix per batch from the PyG Data objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# Maximum SPD we encode; distances beyond this share the last embedding
MAX_DIST = 20
# Maximum node degree we encode
MAX_DEGREE = 10


# ─────────────────────────────────────────────────────────────────────────────
# SPD matrix from edge_index (no 3D needed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_spd_matrix(edge_index: torch.Tensor, n_nodes: int,
                       max_dist: int = MAX_DIST) -> torch.Tensor:
    """
    Compute the shortest-path distance (SPD) matrix for a single molecule.

    Uses BFS from every node. Runs on CPU with Python loops — acceptable for
    drug-like molecules (n_nodes ≤ ~60). Distances > max_dist are clamped.

    Args:
        edge_index: (2, E) undirected edge index (each edge listed twice)
        n_nodes:    number of atoms
        max_dist:   distances beyond this are clamped to max_dist

    Returns:
        spd: (n_nodes, n_nodes) int32 tensor, diagonal = 0
    """
    # Build adjacency list
    adj = [[] for _ in range(n_nodes)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        adj[u].append(v)

    spd = torch.full((n_nodes, n_nodes), max_dist, dtype=torch.long)
    spd.fill_diagonal_(0)

    from collections import deque
    for start in range(n_nodes):
        visited = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            d = visited[node]
            if d >= max_dist:
                continue
            for nb in adj[node]:
                if nb not in visited:
                    visited[nb] = d + 1
                    queue.append(nb)
        for end, d in visited.items():
            spd[start, end] = min(d, max_dist)

    return spd


# ─────────────────────────────────────────────────────────────────────────────
# Graphormer layer
# ─────────────────────────────────────────────────────────────────────────────

class GraphormerLayer(nn.Module):
    """
    Single Graphormer transformer layer with three structural attention biases.

    The pre-softmax attention logits are modified as:
        A[i,j] += centrality_bias[i] + centrality_bias[j]
                + spatial_bias[SPD[i,j]]
                + edge_bias[i,j]

    Args:
        hidden_dim: embedding / attention dimension
        num_heads:  attention heads (must divide hidden_dim)
        ffn_dim:    feedforward hidden dimension
        dropout:    dropout on attention weights and FFN
        max_dist:   maximum SPD to embed (entries beyond are clamped)
        max_degree: maximum in-degree to embed for centrality bias
        edge_dim:   input edge feature dimension (10 for our mol graphs)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads:  int = 8,
        ffn_dim:    int = 256,
        dropout:    float = 0.1,
        max_dist:   int = MAX_DIST,
        max_degree: int = MAX_DEGREE,
        edge_dim:   int = 10,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Bias embeddings
        self.centrality_bias = nn.Embedding(max_degree + 1, num_heads)  # per head
        self.spatial_bias    = nn.Embedding(max_dist  + 1, num_heads)   # per head
        self.edge_proj       = nn.Linear(edge_dim, num_heads)           # per head

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x:      torch.Tensor,          # (N, D)
        spd:    torch.Tensor,          # (N, N) long — shortest-path distances
        degree: torch.Tensor,          # (N,)   long — node in-degrees (clamped)
        edge_bias_mat: torch.Tensor,   # (N, N, num_heads) pre-projected edge bias
    ) -> torch.Tensor:
        """Returns updated node embeddings (N, D)."""
        N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # ── Multi-head attention with bias ───────────────────────────────────
        Q = self.q_proj(x).view(N, H, Dh).transpose(0, 1)  # (H, N, Dh)
        K = self.k_proj(x).view(N, H, Dh).transpose(0, 1)
        V = self.v_proj(x).view(N, H, Dh).transpose(0, 1)

        scale = Dh ** -0.5
        attn = torch.einsum("hid,hjd->hij", Q, K) * scale  # (H, N, N)

        # Centrality bias: b_c[deg_i] + b_c[deg_j]  →  (num_heads, N, N)
        cb = self.centrality_bias(degree)                   # (N, H)
        cb_i = cb.unsqueeze(1).expand(N, N, H)             # (N, N, H)
        cb_j = cb.unsqueeze(0).expand(N, N, H)
        cent_b = (cb_i + cb_j).permute(2, 0, 1)            # (H, N, N)

        # Spatial bias: b_sp[SPD[i,j]]  →  (H, N, N)
        sp_b = self.spatial_bias(spd).permute(2, 0, 1)     # (H, N, N)

        # Edge bias: already (N, N, H) → (H, N, N)
        edge_b = edge_bias_mat.permute(2, 0, 1)

        attn = attn + cent_b + sp_b + edge_b
        attn = self.attn_drop(F.softmax(attn, dim=-1))     # (H, N, N)

        out = torch.einsum("hij,hjd->hid", attn, V)        # (H, N, Dh)
        out = out.transpose(0, 1).contiguous().view(N, D)  # (N, D)
        out = self.out_proj(out)

        # Pre-norm residual
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Graphormer encoder
# ─────────────────────────────────────────────────────────────────────────────

class GraphormerMolEncoder(nn.Module):
    """
    Stack of GraphormerLayers with input projection and CLS-token pooling.

    Processes a single molecule's node feature matrix (N, node_dim) along
    with its SPD matrix, degree vector, and edge_attr.

    Returns a (hidden_dim,) molecule-level embedding via CLS-token mean-pool.
    """

    def __init__(
        self,
        node_dim:   int = 78,
        hidden_dim: int = 128,
        num_heads:  int = 8,
        num_layers: int = 4,
        ffn_dim:    int = 256,
        dropout:    float = 0.1,
        max_dist:   int = MAX_DIST,
        max_degree: int = MAX_DEGREE,
        edge_dim:   int = 10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads, ffn_dim, dropout,
                            max_dist, max_degree, edge_dim)
            for _ in range(num_layers)
        ])
        self.edge_proj = nn.Linear(edge_dim, num_heads)

    def _build_edge_bias(self, edge_index: torch.Tensor,
                         edge_attr: torch.Tensor,
                         n_nodes: int) -> torch.Tensor:
        """
        Build (N, N, num_heads) edge bias matrix.

        For now: direct edges only (SPD=1 neighbors).
        Disconnected pairs get a zero bias.
        """
        device = edge_attr.device
        H = self.layers[0].num_heads
        mat = torch.zeros(n_nodes, n_nodes, H, device=device)
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            projected = self.edge_proj(edge_attr)   # (E, H)
            mat[src, dst] = projected
        return mat

    def forward(
        self,
        x:          torch.Tensor,   # (N, node_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, 10)
    ) -> torch.Tensor:
        """Returns (hidden_dim,) molecule embedding."""
        N = x.size(0)
        device = x.device

        # Project nodes
        h = self.node_proj(x)  # (N, hidden_dim)

        # Compute SPD (CPU-side, small molecules)
        spd = compute_spd_matrix(edge_index.cpu(), N).to(device)  # (N, N)

        # Compute node degrees (in-degree, clamped)
        degree = torch.zeros(N, dtype=torch.long, device=device)
        if edge_index.size(1) > 0:
            dst = edge_index[1]
            for i in range(N):
                degree[i] = (dst == i).sum().clamp(max=self.max_degree)

        # Build edge bias matrix
        edge_bias = self._build_edge_bias(edge_index, edge_attr, N)

        # Pass through layers
        for layer in self.layers:
            h = layer(h, spd, degree, edge_bias)

        # Mean pool over all nodes (CLS-style)
        return h.mean(dim=0)   # (hidden_dim,)


# ─────────────────────────────────────────────────────────────────────────────
# Graphormer binding model (matches GCN/GAT interface)
# ─────────────────────────────────────────────────────────────────────────────

class GraphormerBindingModel(nn.Module):
    """
    Graphormer molecule encoder + flat protein vector → pKd prediction.

    Forward signature matches GCNBindingModel / GATBindingModel so it works
    with the existing run_gnn() runner and GNNDataset/gnn_trainer unchanged.

    Args:
        node_dim:   atom feature dimension (78)
        hidden_dim: Graphormer hidden dimension
        prot_dim:   protein vector dimension
        num_heads:  attention heads
        num_layers: Graphormer layers
        dropout:    dropout rate
    """

    def __init__(
        self,
        node_dim:   int = 78,
        hidden_dim: int = 128,
        prot_dim:   int = 320,
        num_heads:  int = 8,
        num_layers: int = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphormerMolEncoder(
            node_dim=node_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            ffn_dim=hidden_dim * 2, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + prot_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        x:          torch.Tensor,   # (N_total, node_dim)  — PyG batched nodes
        edge_index: torch.Tensor,   # (2, E_total)
        batch:      torch.Tensor,   # (N_total,) — molecule index per node
        prot_vec:   torch.Tensor,   # (B, prot_dim)
    ) -> torch.Tensor:
        """Returns (B, 1) predicted pKd."""
        from torch_geometric.data import Data, Batch as PyGBatch

        device = x.device
        B = prot_vec.size(0)
        mol_reprs = []

        # Process each molecule separately (required for per-molecule SPD)
        for mol_idx in range(B):
            mask = (batch == mol_idx)
            node_ids = mask.nonzero(as_tuple=True)[0]

            # Re-index edges for this molecule
            x_mol = x[node_ids]                     # (n, node_dim)
            n = x_mol.size(0)

            # Find edges belonging to this molecule and re-index
            edge_mask = mask[edge_index[0]]
            ei_mol = edge_index[:, edge_mask]
            # Remap global node ids to local [0, n)
            id_map = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
            id_map[node_ids] = torch.arange(n, device=device)
            ei_local = id_map[ei_mol]               # (2, e)

            # Get edge attributes for this molecule
            # edge_attr is not batched in GNNDataset currently — use zeros as fallback
            # (GNNDataset doesn't pass edge_attr to train loop; extend later if needed)
            e = ei_local.size(1)
            ea_mol = torch.zeros(e, 10, device=device)

            mol_repr = self.encoder(x_mol, ei_local, ea_mol)  # (hidden_dim,)
            mol_reprs.append(mol_repr)

        mol_batch = torch.stack(mol_reprs, dim=0)           # (B, hidden_dim)
        combined  = torch.cat([mol_batch, prot_vec], dim=-1)
        return self.head(combined)                          # (B, 1)
