"""
Graph Neural Network models for molecular binding affinity prediction.

Ligand is represented as a molecular graph (atoms=nodes, bonds=edges).
Protein is represented as a flat vector (AAC, ESM-2, etc.) concatenated
after GNN pooling.

Node features (78-dim per atom):
  - One-hot atomic number (44 most common atoms + other)
  - One-hot degree (0-10)
  - One-hot implicit valence (0-5)
  - Formal charge, num radical electrons
  - One-hot hybridisation (SP, SP2, SP3, SP3D, SP3D2)
  - Aromaticity flag
  - One-hot num Hs (0-4)

Edge features (10-dim per bond):
  - One-hot bond type (single, double, triple, aromatic)
  - Conjugated flag
  - In-ring flag
  - One-hot stereo (STEREONONE, STEREOANY, STEREOZ, STEREOE)

Requires: torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# Atom / Bond featurization (from SMILES, no 3D needed)
# ─────────────────────────────────────────────────────────────────────────────

ATOM_LIST = [
    'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe',
    'As','Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co',
    'Se','Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn',
    'Zr','Cr','Pt','Hg','Pb','other'
]
ATOM2IDX = {a: i for i, a in enumerate(ATOM_LIST)}

def one_hot(val, choices):
    vec = [0] * len(choices)
    idx = choices.index(val) if val in choices else len(choices) - 1
    vec[idx] = 1
    return vec

def atom_features(atom) -> List[float]:
    from rdkit.Chem import rdchem
    sym = atom.GetSymbol()
    feats = (
        one_hot(sym, ATOM_LIST) +                                    # 44
        one_hot(atom.GetDegree(), list(range(11))) +                  # 11
        one_hot(atom.GetImplicitValence(), list(range(6))) +          # 6
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +     # 2
        one_hot(str(atom.GetHybridization()), [                       # 6
            "SP","SP2","SP3","SP3D","SP3D2","other"]) +
        [int(atom.GetIsAromatic())] +                                 # 1
        one_hot(atom.GetTotalNumHs(), [0,1,2,3,4])                   # 5
    )
    return feats   # total = 44+11+6+2+6+1+5 = 75... padded to 78 below


def bond_features(bond) -> List[float]:
    from rdkit.Chem import rdchem
    bt = bond.GetBondType()
    feats = (
        one_hot(bt, [
            rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
            rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC,
        ]) +                                                           # 4
        [int(bond.GetIsConjugated())] +                               # 1
        [int(bond.IsInRing())] +                                      # 1
        one_hot(str(bond.GetStereo()), [                              # 4
            "STEREONONE","STEREOANY","STEREOZ","STEREOE"])
    )
    return feats   # total = 10


def mol_to_graph_data(smiles: str, max_atoms: int = 100):
    """
    Convert SMILES to graph tensors.
    Returns:
      x:         (n_atoms, 78) node features
      edge_index: (2, n_edges) connectivity
      edge_attr: (n_edges, 10) edge features
    Returns None if SMILES is invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = []
    for atom in mol.GetAtoms():
        f = atom_features(atom)
        # Pad to 78 if needed
        f = f + [0] * (78 - len(f))
        atom_feats.append(f[:78])
    x = torch.tensor(atom_feats, dtype=torch.float32)  # (n_atoms, 78)

    # Edge index + edge features (undirected: add both directions)
    rows, cols, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        rows += [i, j]; cols += [j, i]
        edge_feats += [bf, bf]

    if len(rows) == 0:
        # Single atom molecule — self loop
        rows, cols = [0], [0]
        edge_feats = [[0] * 10]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr  = torch.tensor(edge_feats, dtype=torch.float32)

    return x, edge_index, edge_attr


# ─────────────────────────────────────────────────────────────────────────────
# GCN model
# ─────────────────────────────────────────────────────────────────────────────

class GCNBindingModel(nn.Module):
    """
    3-layer GCN on molecular graph + MLP head.
    Protein is a flat vector concatenated after GNN pooling.

    Requires torch_geometric.
    """
    def __init__(
        self,
        node_dim:   int = 78,
        hidden_dim: int = 128,
        prot_dim:   int = 320,
        dropout:    float = 0.2,
        num_layers: int = 3,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + prot_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def encode_nodes(self, x, edge_index):
        """Returns per-node hidden states (no pooling). Used by GraphMAE recon."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x  # (N, hidden_dim)

    def forward(self, x, edge_index, batch, prot_vec):
        from torch_geometric.nn import global_mean_pool
        h = self.encode_nodes(x, edge_index)
        mol_repr = global_mean_pool(h, batch)       # (B, hidden_dim)
        combined = torch.cat([mol_repr, prot_vec], dim=-1)
        return self.head(combined)


class GATBindingModel(nn.Module):
    """
    3-layer GAT (Graph Attention Network) on molecular graph + MLP head.
    Uses multi-head attention on edges — better than GCN for molecules.
    """
    def __init__(
        self,
        node_dim:   int = 78,
        hidden_dim: int = 128,
        prot_dim:   int = 320,
        heads:      int = 4,
        dropout:    float = 0.2,
        num_layers: int = 3,
    ):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        in_dim = node_dim
        for i in range(num_layers):
            out_per_head = hidden_dim // heads
            self.convs.append(GATConv(in_dim, out_per_head, heads=heads,
                                       dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + prot_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def encode_nodes(self, x, edge_index):
        """Returns per-node hidden states (no pooling). Used by GraphMAE recon."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)
        return x  # (N, hidden_dim)

    def forward(self, x, edge_index, batch, prot_vec):
        from torch_geometric.nn import global_mean_pool
        h = self.encode_nodes(x, edge_index)
        mol_repr = global_mean_pool(h, batch)
        combined = torch.cat([mol_repr, prot_vec], dim=-1)
        return self.head(combined)


# ─────────────────────────────────────────────────────────────────────────────
# GraphMAE-style node feature reconstruction (Phase 1 auxiliary loss)
# Reference: Hou et al., NeurIPS 2022 — "GraphMAE: Self-supervised masked
# graph autoencoders"
# ─────────────────────────────────────────────────────────────────────────────

NODE_FEAT_DIM = 78   # must match atom_features() padded output


def mask_node_features(x: torch.Tensor, mask_rate: float = 0.15,
                       training: bool = True):
    """
    Randomly zero-out mask_rate fraction of node feature vectors.

    Args:
        x:         (N, node_feat_dim) node feature tensor
        mask_rate: fraction of nodes to mask (default 0.15)
        training:  only mask during training; returns (x, None) at eval time

    Returns:
        x_masked:  (N, node_feat_dim) with masked nodes zeroed
        mask_idx:  1-D LongTensor of masked node indices, or None at eval time
    """
    if not training or mask_rate <= 0.0:
        return x, None

    N = x.size(0)
    n_mask = max(1, int(N * mask_rate))
    mask_idx = torch.randperm(N, device=x.device)[:n_mask]

    x_masked = x.clone()
    x_masked[mask_idx] = 0.0
    return x_masked, mask_idx


class NodeDecoder(nn.Module):
    """
    Lightweight MLP that reconstructs original node features from GNN hidden
    states at masked positions.

        L_recon = MSE(decoder(h[mask_idx]), x_original[mask_idx])

    Args:
        hidden_dim:    GNN hidden dimension (input to decoder)
        node_feat_dim: original node feature dimension to reconstruct
    """
    def __init__(self, hidden_dim: int = 128,
                 node_feat_dim: int = NODE_FEAT_DIM):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (N, hidden_dim) → (N, node_feat_dim)"""
        return self.decoder(h)
