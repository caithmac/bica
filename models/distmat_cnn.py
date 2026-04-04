"""
2D CNN on topological distance matrix for binding affinity prediction.

Ligand is represented as a (max_atoms × max_atoms) topological distance matrix
from RDKit's GetDistanceMatrix() — integer bond-hop distances between every
pair of atoms. The matrix is normalized and treated as a single-channel image.

Protein is a flat vector (AAC, ESM-2, etc.) concatenated after CNN pooling.

Architecture (inspired by DGraphDTA / GNN-DTI):
  - Input: (B, 1, max_atoms, max_atoms)
  - 3 × Conv2d blocks with BatchNorm + ReLU
  - Adaptive average pooling → (B, channels, 1, 1) → flatten
  - Concat with protein vector
  - MLP head → scalar pKd

Requires: rdkit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


MAX_ATOMS = 100   # molecules with more atoms are truncated/padded


def smiles_to_distmat(smiles: str, max_atoms: int = MAX_ATOMS) -> Optional[np.ndarray]:
    """
    Convert SMILES to a (max_atoms, max_atoms) topological distance matrix.
    Returns None if the SMILES is invalid.

    Values are bond-hop distances (integers). Atoms beyond max_atoms are
    ignored (rare for drug-like molecules). The matrix is padded with zeros.
    Diagonal is 0. Normalised to [0, 1] by dividing by max_atoms.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdmolops
    except ImportError:
        raise ImportError("rdkit is required for distmat featurization")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n = mol.GetNumAtoms()
    dm = rdmolops.GetDistanceMatrix(mol)       # (n, n) float64, bond-hop distances

    # Clip to max_atoms × max_atoms
    n_clip = min(n, max_atoms)
    mat = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    mat[:n_clip, :n_clip] = dm[:n_clip, :n_clip]

    # Normalise to [0, 1]
    if mat.max() > 0:
        mat /= mat.max()

    return mat


def smiles_list_to_distmat(smiles_list: List[str],
                           max_atoms: int = MAX_ATOMS) -> np.ndarray:
    """
    Batch conversion: list of SMILES → (N, max_atoms, max_atoms) array.
    Invalid SMILES yield an all-zero matrix.
    """
    out = np.zeros((len(smiles_list), max_atoms, max_atoms), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mat = smiles_to_distmat(smi, max_atoms)
        if mat is not None:
            out[i] = mat
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2D CNN model
# ─────────────────────────────────────────────────────────────────────────────

class DistMatCNN(nn.Module):
    """
    2D CNN that takes a topological distance matrix as input.
    Protein is a flat vector concatenated after CNN pooling.

    Args:
        max_atoms:   matrix side length (default 100)
        prot_dim:    protein feature dimension
        channels:    list of output channels for each Conv2d block
        dropout:     dropout probability
    """
    def __init__(
        self,
        max_atoms: int = MAX_ATOMS,
        prot_dim:  int = 320,
        channels:  List[int] = None,
        dropout:   float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                        padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch

        self.pool    = nn.AdaptiveAvgPool2d((4, 4))   # → (B, 128, 4, 4)
        cnn_out_dim  = channels[-1] * 4 * 4           # 128 * 16 = 2048

        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + prot_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, dm: torch.Tensor, prot_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dm:       (B, max_atoms, max_atoms) — will be unsqueezed to (B,1,H,W)
            prot_vec: (B, prot_dim)
        Returns:
            (B, 1) scalar prediction
        """
        x = dm.unsqueeze(1)                          # (B, 1, H, W)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
            x = self.dropout(x)
        x = self.pool(x)                             # (B, 128, 4, 4)
        x = x.flatten(1)                             # (B, 2048)
        combined = torch.cat([x, prot_vec], dim=-1)  # (B, 2048 + prot_dim)
        return self.head(combined)                   # (B, 1)


def build_distmat_cnn(prot_dim: int = 320, dropout: float = 0.2) -> DistMatCNN:
    """Factory used by run_experiment.py."""
    return DistMatCNN(max_atoms=MAX_ATOMS, prot_dim=prot_dim,
                      channels=[32, 64, 128], dropout=dropout)
