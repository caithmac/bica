"""
Main experiment runner.

Usage examples:
  python run_experiment.py --exp rf_ecfp4_aac
  python run_experiment.py --group baselines
  python run_experiment.py --group all
  python run_experiment.py --list

  # Multi-seed (BindingDB only)
  python run_experiment.py --exp rf_ecfp4_aac --seed 123

  # LeakyPDB dataset
  python run_experiment.py --exp rf_ecfp4_aac --dataset leakypdb

Each experiment is fully self-contained: it defines its featurization,
model, and calls log_result() at the end.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # suppress libiomp5md.dll conflict warning

import argparse
import sys
import time
import numpy as np
import torch.nn as nn

# ── Harness imports ───────────────────────────────────────────────────────────
from harness.config import SPLIT_DIR, CACHE_DIR, BATCH_SIZE, LEARNING_RATE, SPLIT_SEED
from harness.data import get_splits_for_seed, get_leakypdb_splits
from harness.metrics import compute_metrics, format_metrics
from harness.diary import log_result, print_leaderboard, save_predictions
from harness.trainer import train_sklearn, train_torch, count_parameters
import harness.featurizers as F

from rdkit import RDLogger

# Disable all logs from the rdApp (warnings, info, etc.)
RDLogger.DisableLog('rdApp.*')

# ── Active dataset/seed — set by CLI args before any run() call ───────────────
_ACTIVE_DATASET = "bindingdb"   # "bindingdb" | "leakypdb"
_ACTIVE_SEED    = SPLIT_SEED    # default 42


def _get_splits():
    """Return (train_df, val_df, test_df) for the currently active dataset/seed."""
    if _ACTIVE_DATASET == "leakypdb":
        return get_leakypdb_splits()
    else:
        return get_splits_for_seed(_ACTIVE_SEED)


def _split_tag() -> str:
    """Short string used in experiment_id and split_type field."""
    if _ACTIVE_DATASET == "leakypdb":
        return "leakypdb_cl1cl2"
    return f"scaffold_bemis_murcko_seed{_ACTIVE_SEED}"


def _exp_id(base_name: str) -> str:
    """Append dataset/seed suffix to experiment_id when not default."""
    if _ACTIVE_DATASET == "leakypdb":
        return f"{base_name}__leakypdb"
    if _ACTIVE_SEED != SPLIT_SEED:
        return f"{base_name}__seed{_ACTIVE_SEED}"
    return base_name


def _feat_cache_prefix() -> str:
    """Prefix for feature cache files, so different datasets don't collide."""
    if _ACTIVE_DATASET == "leakypdb":
        return "leakypdb_"
    if _ACTIVE_SEED != SPLIT_SEED:
        return f"s{_ACTIVE_SEED}_"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS: dict[str, dict] = {}   # name → config dict


def register(name: str, group: str, **kwargs):
    EXPERIMENTS[name] = {"name": name, "group": group, **kwargs}


# ── Group: baselines ──────────────────────────────────────────────────────────

register("ridge_ecfp4_aac",
    group="baselines",
    model_family="linear",
    ligand_repr="ecfp4_1024", protein_repr="aac_20", fusion="concat",
)
register("ridge_ecfp4_dipeptide",
    group="baselines",
    model_family="linear",
    ligand_repr="ecfp4_1024", protein_repr="dipeptide_400", fusion="concat",
)
register("ridge_maccs_aac",
    group="baselines",
    model_family="linear",
    ligand_repr="maccs_167", protein_repr="aac_20", fusion="concat",
)
register("ridge_rdkit_aac",
    group="baselines",
    model_family="linear",
    ligand_repr="rdkit_200", protein_repr="aac_20", fusion="concat",
)

# ── Group: trees ──────────────────────────────────────────────────────────────

register("rf_ecfp4_aac",
    group="trees",
    model_family="tree",
    ligand_repr="ecfp4_1024", protein_repr="aac_20", fusion="concat",
)
register("rf_ecfp4_dipeptide",
    group="trees",
    model_family="tree",
    ligand_repr="ecfp4_1024", protein_repr="dipeptide_400", fusion="concat",
)
register("xgb_ecfp4_aac",
    group="trees",
    model_family="tree",
    ligand_repr="ecfp4_1024", protein_repr="aac_20", fusion="concat",
)
register("xgb_ecfp6_kmer3",
    group="trees",
    model_family="tree",
    ligand_repr="ecfp6_1024", protein_repr="kmer3_8000", fusion="concat",
)
register("lgbm_ecfp4_aac",
    group="trees",
    model_family="tree",
    ligand_repr="ecfp4_1024", protein_repr="aac_20", fusion="concat",
)

# ── Group: gp (Gaussian Processes — reviewer-requested baseline) ─────────

register("gp_ecfp4_tanimoto",
    group="gp",
    model_family="gp",
    ligand_repr="ecfp4_1024", protein_repr="none", fusion="ligand_only",
    gp_kernel="tanimoto",
    notes="Reviewer: GP + ECFP4 + Tanimoto kernel — missing virtual screening baseline",
)
register("gp_ecfp4_rbf",
    group="gp",
    model_family="gp",
    ligand_repr="ecfp4_1024", protein_repr="none", fusion="ligand_only",
    gp_kernel="rbf",
)
register("gp_ecfp4_matern",
    group="gp",
    model_family="gp",
    ligand_repr="ecfp4_1024", protein_repr="none", fusion="ligand_only",
    gp_kernel="matern",
)
register("gp_ecfp4_rq",
    group="gp",
    model_family="gp",
    ligand_repr="ecfp4_1024", protein_repr="none", fusion="ligand_only",
    gp_kernel="rq",
)
register("gp_ecfp4_aac",
    group="gp",
    model_family="gp",
    ligand_repr="ecfp4_1024", protein_repr="aac_20", fusion="concat",
    gp_kernel="rbf",
    notes="Reviewer: GP + ECFP4+AAC + RBF kernel",
)

register("gp_ecfp4_esm2_8M",
    group="gp",
    model_family="gp",
    ligand_repr="ecfp4_1024", protein_repr="esm2_8M_320", fusion="concat",
    gp_kernel="rbf",
    notes="Reviewer: GP + ECFP4 + ESM-2 8M frozen — tests if rich protein features help GP",
)
register("gp_ecfp4_esm2_35M",
    group="gp",
    model_family="gp",
    ligand_repr="ecfp4_1024", protein_repr="esm2_35M_480", fusion="concat",
    gp_kernel="rbf",
    notes="Reviewer: GP + ECFP4 + ESM-2 35M frozen",
)

# ── Group: mlp ────────────────────────────────────────────────────────────────

register("mlp_shallow_ecfp4_aac",
    group="mlp",
    model_family="mlp",
    ligand_repr="ecfp4_1024", protein_repr="aac_20", fusion="concat",
    mlp_arch="shallow",
)
register("mlp_medium_ecfp4_dipeptide",
    group="mlp",
    model_family="mlp",
    ligand_repr="ecfp4_1024", protein_repr="dipeptide_400", fusion="concat",
    mlp_arch="medium",
)
register("mlp_deep_ecfp6_kmer3",
    group="mlp",
    model_family="mlp",
    ligand_repr="ecfp6_1024", protein_repr="kmer3_8000", fusion="concat",
    mlp_arch="deep",
)
register("mlp_wide_maccs_dipeptide",
    group="mlp",
    model_family="mlp",
    ligand_repr="maccs_167", protein_repr="dipeptide_400", fusion="concat",
    mlp_arch="wide",
)

# ── Group: pretrained ─────────────────────────────────────────────────────────

register("mlp_chemberta_aac",
    group="pretrained",
    model_family="mlp",
    ligand_repr="chemberta_600", protein_repr="aac_20", fusion="concat",
    mlp_arch="medium",
)
register("mlp_chemberta_dipeptide",
    group="pretrained",
    model_family="mlp",
    ligand_repr="chemberta_600", protein_repr="dipeptide_400", fusion="concat",
    mlp_arch="medium",
)
register("mlp_ecfp4_esm2_8M",
    group="pretrained",
    model_family="mlp",
    ligand_repr="ecfp4_1024", protein_repr="esm2_8M_320", fusion="concat",
    mlp_arch="medium",
)
register("mlp_chemberta_esm2_8M",
    group="pretrained",
    model_family="mlp",
    ligand_repr="chemberta_600", protein_repr="esm2_8M_320", fusion="concat",
    mlp_arch="deep",
)
register("mlp_chemberta_esm2_35M",
    group="pretrained",
    model_family="mlp",
    ligand_repr="chemberta_600", protein_repr="esm2_35M_480", fusion="concat",
    mlp_arch="deep",
)
register("xgb_chemberta_esm2_8M",
    group="pretrained",
    model_family="tree",
    ligand_repr="chemberta_600", protein_repr="esm2_8M_320", fusion="concat",
)
# ── Ablation: isolate ligand vs protein representation gain ──────────────────
# These two fill the 2×2 grid: (ecfp4 vs chemberta) × (aac vs esm2) for trees
register("xgb_ecfp4_esm2_8M",
    group="pretrained",
    model_family="tree",
    ligand_repr="ecfp4_1024",    protein_repr="esm2_8M_320", fusion="concat",
    notes="Ablation: ECFP4 + ESM-2 (protein upgrade only)",
)
register("xgb_chemberta_aac",
    group="pretrained",
    model_family="tree",
    ligand_repr="chemberta_600", protein_repr="aac_20",      fusion="concat",
    notes="Ablation: ChemBERTa + AAC (ligand upgrade only)",
)

# ── Group: transformer ────────────────────────────────────────────────────────

register("transformer_ecfp4_aac",
    group="transformer",
    model_family="transformer",
    ligand_repr="ecfp4_1024", protein_repr="aac_20", fusion="concat",
    transformer_arch="self_attn",
)
register("transformer_chemberta_esm2_8M",
    group="transformer",
    model_family="transformer",
    ligand_repr="chemberta_600", protein_repr="esm2_8M_320", fusion="cross_attention",
    transformer_arch="cross_attn",
)
register("transformer_chemberta_esm2_35M",
    group="transformer",
    model_family="transformer",
    ligand_repr="chemberta_600", protein_repr="esm2_35M_480", fusion="cross_attention",
    transformer_arch="cross_attn",
)

# ── Group: cnn ────────────────────────────────────────────────────────────────

register("cnn_smiles_onehot_aac",
    group="cnn",
    model_family="cnn",
    ligand_repr="smiles_onehot", protein_repr="aac_20", fusion="concat",
)

# ── Group: seq_models ─────────────────────────────────────────────────────────
# These experiments use token ID sequences + learned embeddings.
# Tokenizer strategies: char, atom-level, BPE-512, BPE-1000, WordPiece-512
# Model strategies: BiLSTM, Transformer-with-learned-embeddings

register("lstm_smiles_char_protein_char",
    group="seq_models",
    model_family="lstm",
    lig_tok="smiles_char",       # character-level SMILES
    prot_tok="protein_char",     # character-level protein
    fusion="dual_encoder",
    notes="Simplest tokenization: 1 char = 1 token",
)
register("lstm_smiles_atom_protein_char",
    group="seq_models",
    model_family="lstm",
    lig_tok="smiles_atom",       # chemistry-aware atom-level
    prot_tok="protein_char",
    fusion="dual_encoder",
    notes="Atom-aware SMILES tokenization (Br,Cl,[NH2+] as single tokens)",
)
register("lstm_smiles_bpe512_protein_bpe512",
    group="seq_models",
    model_family="lstm",
    lig_tok="smiles_bpe_512",    # BPE trained on dataset, vocab=512
    prot_tok="protein_bpe_512",
    fusion="dual_encoder",
    notes="BPE small vocab: learns sub-structure patterns",
)
register("lstm_smiles_bpe1000_protein_bpe1000",
    group="seq_models",
    model_family="lstm",
    lig_tok="smiles_bpe_1000",
    prot_tok="protein_bpe_1000",
    fusion="dual_encoder",
    notes="BPE larger vocab: richer sub-structure patterns",
)
register("lstm_smiles_atom_protein_wordpiece512",
    group="seq_models",
    model_family="lstm",
    lig_tok="smiles_atom",
    prot_tok="protein_wordpiece_512",
    fusion="dual_encoder",
    notes="WordPiece protein tokenization (BERT-style)",
)
register("transformer_seq_smiles_char_protein_char",
    group="seq_models",
    model_family="transformer_seq",
    lig_tok="smiles_char",
    prot_tok="protein_char",
    fusion="dual_encoder",
    notes="Learned-embedding transformer, character tokenization",
)
register("transformer_seq_smiles_atom_protein_char",
    group="seq_models",
    model_family="transformer_seq",
    lig_tok="smiles_atom",
    prot_tok="protein_char",
    fusion="dual_encoder",
    notes="Learned-embedding transformer, atom-level SMILES",
)
register("transformer_seq_smiles_bpe512_protein_bpe512",
    group="seq_models",
    model_family="transformer_seq",
    lig_tok="smiles_bpe_512",
    prot_tok="protein_bpe_512",
    fusion="dual_encoder",
    notes="Learned-embedding transformer, BPE-512",
)
register("transformer_seq_smiles_bpe1000_protein_bpe1000",
    group="seq_models",
    model_family="transformer_seq",
    lig_tok="smiles_bpe_1000",
    prot_tok="protein_bpe_1000",
    fusion="dual_encoder",
    notes="Learned-embedding transformer, BPE-1000",
)
register("transformer_seq_smiles_atom_protein_wordpiece512",
    group="seq_models",
    model_family="transformer_seq",
    lig_tok="smiles_atom",
    prot_tok="protein_wordpiece_512",
    fusion="dual_encoder",
    notes="Learned-embedding transformer, WordPiece protein",
)

# ── Group: distmat ────────────────────────────────────────────────────────────
# 2D CNN on topological distance matrix (bond-hop distances from RDKit).

register("distmat_cnn_aac",
    group="distmat",
    model_family="distmat_cnn",
    ligand_repr="distmat_100",    protein_repr="aac_20",       fusion="distmat",
    notes="2D CNN on RDKit topological distance matrix + AAC",
)
register("distmat_cnn_esm2_8M",
    group="distmat",
    model_family="distmat_cnn",
    ligand_repr="distmat_100",    protein_repr="esm2_8M_320",  fusion="distmat",
    notes="2D CNN on RDKit distance matrix + ESM-2 8M",
)

# ── Group: graph_models ───────────────────────────────────────────────────────
# GCN and GAT with PyTorch Geometric (full molecular graph, no 3D needed).

register("gcn_ecfp_aac",
    group="graph_models",
    model_family="gcn",
    ligand_repr="mol_graph",      protein_repr="aac_20",       fusion="graph",
    notes="3-layer GCN + AAC protein vector",
)
register("gcn_ecfp_esm2_8M",
    group="graph_models",
    model_family="gcn",
    ligand_repr="mol_graph",      protein_repr="esm2_8M_320",  fusion="graph",
    notes="3-layer GCN + ESM-2 8M protein",
)
register("gat_ecfp_aac",
    group="graph_models",
    model_family="gat",
    ligand_repr="mol_graph",      protein_repr="aac_20",       fusion="graph",
    notes="3-layer GAT (4 heads) + AAC protein vector",
)
register("gat_ecfp_esm2_8M",
    group="graph_models",
    model_family="gat",
    ligand_repr="mol_graph",      protein_repr="esm2_8M_320",  fusion="graph",
    notes="3-layer GAT (4 heads) + ESM-2 8M protein",
)

# ── Group: prot_electra ───────────────────────────────────────────────────────
# ProtElectra (RTD pre-training on BFD) as protein encoder.

register("mlp_ecfp4_prot_electra",
    group="prot_electra",
    model_family="mlp",
    ligand_repr="ecfp4_1024",     protein_repr="prot_electra_256",  fusion="concat",
    mlp_arch="medium",
    notes="MLP + ECFP4 + ProtElectra-BFD 256-dim",
)
register("mlp_chemberta_prot_electra",
    group="prot_electra",
    model_family="mlp",
    ligand_repr="chemberta_600",  protein_repr="prot_electra_256",  fusion="concat",
    mlp_arch="deep",
    notes="MLP + ChemBERTa + ProtElectra-BFD 256-dim",
)
register("xgb_chemberta_prot_electra",
    group="prot_electra",
    model_family="tree",
    ligand_repr="chemberta_600",  protein_repr="prot_electra_256",  fusion="concat",
    notes="XGBoost + ChemBERTa + ProtElectra-BFD 256-dim",
)

# ── Group: bica ───────────────────────────────────────────────────────────────
# Bidirectional Cross-Attention (BiCA) — protein and ligand attend to each other.

register("bica_ecfp4_aac",
    group="bica",
    model_family="bica",
    ligand_repr="ecfp4_1024",     protein_repr="aac_20",       fusion="bidirectional_cross_attn",
    notes="BiCA flat vectors: ECFP4 + AAC (each unsqueezed to seq_len=1)",
)
register("bica_chemberta_esm2_8M",
    group="bica",
    model_family="bica",
    ligand_repr="chemberta_600",  protein_repr="esm2_8M_320",  fusion="bidirectional_cross_attn",
    notes="BiCA flat vectors: ChemBERTa + ESM-2 8M",
)
register("bica_chemberta_prot_electra",
    group="bica",
    model_family="bica",
    ligand_repr="chemberta_600",  protein_repr="prot_electra_256",  fusion="bidirectional_cross_attn",
    notes="BiCA flat vectors: ChemBERTa + ProtElectra",
)

# ── Group: phase1_ranking — Pairwise Ranking Loss (from MBP paper) ────────────
# Adds ranking-aware training to top deep models.
# Loss = MSE + 0.1 * PairwiseRankingLoss(margin=0.5, n_pairs=32)
# Directly optimises Spearman R in addition to RMSE.

register("mlp_chemberta_esm2_8M_ranked",
    group="phase1_ranking",
    model_family="mlp",
    ligand_repr="chemberta_600",  protein_repr="esm2_8M_320",  fusion="concat",
    mlp_arch="deep",
    use_ranking_loss=True,
    notes="Phase1: MLP + ChemBERTa + ESM-2 8M with pairwise ranking loss (MBP)",
)
register("bica_chemberta_esm2_8M_ranked",
    group="phase1_ranking",
    model_family="bica",
    ligand_repr="chemberta_600",  protein_repr="esm2_8M_320",  fusion="bidirectional_cross_attn",
    use_ranking_loss=True,
    notes="Phase1: BiCA + ChemBERTa + ESM-2 8M with pairwise ranking loss (MBP)",
)
register("gat_ecfp_esm2_8M_ranked",
    group="phase1_ranking",
    model_family="gat",
    ligand_repr="mol_graph",      protein_repr="esm2_8M_320",  fusion="graph",
    use_ranking_loss=True,
    notes="Phase1: GAT + ESM-2 8M with pairwise ranking loss (MBP)",
)

# ── Group: phase1_recon — GraphMAE Node Reconstruction Auxiliary Loss ─────────
# Masks 15% of atom node features during training; reconstructs them via a
# NodeDecoder head.  Loss = MSE + 0.1 * recon_MSE.
# Reference: Hou et al., NeurIPS 2022 (GraphMAE).

register("gcn_ecfp_esm2_8M_recon",
    group="phase1_recon",
    model_family="gcn",
    ligand_repr="mol_graph",      protein_repr="esm2_8M_320",  fusion="graph",
    use_node_recon=True,
    notes="Phase1: GCN + ESM-2 8M with GraphMAE node reconstruction aux loss",
)
register("gat_ecfp_esm2_8M_recon",
    group="phase1_recon",
    model_family="gat",
    ligand_repr="mol_graph",      protein_repr="esm2_8M_320",  fusion="graph",
    use_node_recon=True,
    notes="Phase1: GAT + ESM-2 8M with GraphMAE node reconstruction aux loss",
)

# ── Group: phase2_graphormer — Graph Transformer with structural biases ───────
# Reference: Ying et al., NeurIPS 2021 (Graphormer)
# 3 additive biases to attention logits: centrality (degree), spatial (SPD),
# edge type.  Same GNNDataset/gnn_trainer interface as GCN/GAT.

register("graphormer_mol_aac",
    group="phase2_graphormer",
    model_family="graphormer",
    ligand_repr="mol_graph",      protein_repr="aac_20",        fusion="graph",
    notes="Phase2: Graphormer (4 layers) + AAC protein",
)
register("graphormer_mol_esm2_8M",
    group="phase2_graphormer",
    model_family="graphormer",
    ligand_repr="mol_graph",      protein_repr="esm2_8M_320",   fusion="graph",
    notes="Phase2: Graphormer (4 layers) + ESM-2 8M — key new architecture",
)
register("graphormer_mol_prot_electra",
    group="phase2_graphormer",
    model_family="graphormer",
    ligand_repr="mol_graph",      protein_repr="prot_electra_256", fusion="graph",
    notes="Phase2: Graphormer (4 layers) + ProtElectra",
)
register("graphormer_mol_esm2_35M",
    group="phase2_graphormer",
    model_family="graphormer",
    ligand_repr="mol_graph",      protein_repr="esm2_35M_480",  fusion="graph",
    notes="Phase2: Graphormer (4 layers) + ESM-2 35M",
)

# ── Group: phase2_gli — Gated Global-Local Interaction ────────────────────────
# Reference: Inspired by GLI framework (joint global/local interaction for DTA)
# Local branch: GNN per-atom hidden states + global mean pool
# Global branch: BiCA cross-attention between protein (seq=1) and GNN atoms
# Gate: g = sigmoid(W_g * [mol_repr; bica_repr])
# Same forward signature as GCN/GAT → works with run_gnn() unchanged.

register("gli_mol_aac",
    group="phase2_gli",
    model_family="gli",
    ligand_repr="mol_graph",      protein_repr="aac_20",        fusion="graph",
    notes="Phase2: GLI gated fusion (GNN + BiCA) + AAC protein",
)
register("gli_mol_esm2_8M",
    group="phase2_gli",
    model_family="gli",
    ligand_repr="mol_graph",      protein_repr="esm2_8M_320",   fusion="graph",
    notes="Phase2: GLI gated fusion + ESM-2 8M — main GLI experiment",
)
register("gli_mol_prot_electra",
    group="phase2_gli",
    model_family="gli",
    ligand_repr="mol_graph",      protein_repr="prot_electra_256", fusion="graph",
    notes="Phase2: GLI gated fusion + ProtElectra",
)
register("gli_mol_esm2_35M",
    group="phase2_gli",
    model_family="gli",
    ligand_repr="mol_graph",      protein_repr="esm2_35M_480",  fusion="graph",
    notes="Phase2: GLI gated fusion + ESM-2 35M",
)

# ── Group: phase2_dsm — DualBind DSM Auxiliary Loss ─────────────────────────
# Reference: DualBind (Denoising Score Matching)
# Forces encoder to learn smooth binding energy surface via noise corruption.
# Score network predicts ε/σ² on corrupted embeddings (5 geometric noise levels).
# Total loss: L = MSE + lambda_dsm * L_DSM

register("bica_chemberta_esm2_8M_dsm",
    group="phase2_dsm",
    model_family="bica",
    ligand_repr="chemberta_600",  protein_repr="esm2_8M_320",   fusion="bidirectional_cross_attn",
    use_dsm=True,
    notes="Phase2: BiCA + ChemBERTa + ESM-2 8M + DualBind DSM auxiliary loss",
)
register("bica_chemberta_prot_electra_dsm",
    group="phase2_dsm",
    model_family="bica",
    ligand_repr="chemberta_600",  protein_repr="prot_electra_256", fusion="bidirectional_cross_attn",
    use_dsm=True,
    notes="Phase2: BiCA + ChemBERTa + ProtElectra + DSM auxiliary loss",
)
register("mlp_chemberta_esm2_8M_dsm",
    group="phase2_dsm",
    model_family="mlp",
    ligand_repr="chemberta_600",  protein_repr="esm2_8M_320",   fusion="concat",
    mlp_arch="deep",
    use_dsm=True,
    notes="Phase2: MLP deep + ChemBERTa + ESM-2 8M + DualBind DSM auxiliary loss",
)

# ── Group: phase2_mamba — Mamba SSM sequence encoder ─────────────────────────
# Reference: Gu & Dao, NeurIPS 2023 (Mamba: Linear-Time Sequence Modeling)
# O(N) selective state space model replacing transformer attention.
# Falls back to BiLSTM if mamba-ssm not installed.

register("mamba_smiles_char_protein_char",
    group="phase2_mamba",
    model_family="mamba",
    lig_tok="smiles_char",        prot_tok="protein_char",      fusion="dual_encoder",
    notes="Phase2: Mamba SSM character-level tokenization",
)
register("mamba_smiles_atom_protein_char",
    group="phase2_mamba",
    model_family="mamba",
    lig_tok="smiles_atom",        prot_tok="protein_char",      fusion="dual_encoder",
    notes="Phase2: Mamba SSM atom-level SMILES + char protein",
)
register("mamba_smiles_bpe512_protein_bpe512",
    group="phase2_mamba",
    model_family="mamba",
    lig_tok="smiles_bpe_512",     prot_tok="protein_bpe_512",   fusion="dual_encoder",
    notes="Phase2: Mamba SSM BPE-512 tokenization",
)
register("mamba_smiles_bpe1000_protein_bpe1000",
    group="phase2_mamba",
    model_family="mamba",
    lig_tok="smiles_bpe_1000",    prot_tok="protein_bpe_1000",  fusion="dual_encoder",
    notes="Phase2: Mamba SSM BPE-1000 tokenization",
)

# ── Group: bica_v2 — True-sequence BiCA ablations ────────────────────────────
# All use ESM-2 35M per-residue protein tokens (480-dim) + per-atom RDKit
# features (78-dim, max 100 atoms) as true sequence inputs.

register("bica_v2_full",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2",
    esm2_size="35M",
    max_atoms=100,
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.3,
    drop_path=0.1,
    peak_lr=1e-4,
    notes="BiCA v2 full: Pre-LN + stacked cross-attn + attn pool + FFN + DropPath + warmup",
)
register("bica_v2_meanpool",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2_meanpool",
    esm2_size="35M",
    max_atoms=100,
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.2,
    peak_lr=1e-4,
    notes="BiCA v2 ablation: mean pooling instead of attention pooling",
)
register("bica_v2_singlelayer",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2_singlelayer",
    esm2_size="35M",
    max_atoms=100,
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=1,
    dropout=0.2,
    peak_lr=1e-4,
    notes="BiCA v2 ablation: 1 cross-attention layer (depth ablation)",
)
register("bica_v2_noffn",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2_noffn",
    esm2_size="35M",
    max_atoms=100,
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.2,
    peak_lr=1e-4,
    notes="BiCA v2 ablation: no FFN block (pure cross-attention)",
)
register("bica_v2_noresidual",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2_noresidual",
    esm2_size="35M",
    max_atoms=100,
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.2,
    peak_lr=1e-4,
    notes="BiCA v2 ablation: no residual connections",
)
register("bica_v2_p2l",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2_p2l",
    esm2_size="35M",
    max_atoms=100,
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.2,
    peak_lr=1e-4,
    notes="BiCA v2 ablation: unidirectional protein→ligand only",
)
register("bica_v2_l2p",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2_l2p",
    esm2_size="35M",
    max_atoms=100,
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.2,
    peak_lr=1e-4,
    notes="BiCA v2 ablation: unidirectional ligand→protein only",
)
register("concat_baseline",
    group="bica_v2",
    model_family="bica_v2",
    variant="concat_baseline",
    esm2_size="35M",
    max_atoms=100,
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.2,
    peak_lr=1e-4,
    notes="BiCA v2 ablation: no attention — concat projected features only",
)
register("bica_v2_chemberta_tokens",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2",
    esm2_size="35M",
    ligand_repr="chemberta_tokens",
    chemberta_model="seyonec/ChemBERTa-zinc-base-v1",
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.3,
    drop_path=0.1,
    peak_lr=1e-4,
    notes="BiCA v2: ESM-2 35M per-residue × ChemBERTa-zinc per-token cross-attention",
)
register("bica_v2_chemberta77M_tokens",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2",
    esm2_size="35M",
    ligand_repr="chemberta_tokens",
    chemberta_model="DeepChem/ChemBERTa-77M-MTR",
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.3,
    drop_path=0.1,
    peak_lr=1e-4,
    notes="BiCA v2: ESM-2 35M per-residue × ChemBERTa-77M-MTR per-token cross-attention",
)
register("bica_v2_chemberta_tokens_esm2_150M",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2",
    esm2_size="150M",
    ligand_repr="chemberta_tokens",
    chemberta_model="seyonec/ChemBERTa-zinc-base-v1",
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.3,
    drop_path=0.1,
    peak_lr=1e-4,
    notes="BiCA v2: ESM-2 150M (640-dim) per-residue × ChemBERTa-zinc per-token",
)
register("bica_v2_cb77M_mtr_esm2_150M",
    group="bica_v2",
    model_family="bica_v2",
    variant="bica_v2",
    esm2_size="150M",
    ligand_repr="chemberta_tokens",
    chemberta_model="DeepChem/ChemBERTa-77M-MTR",
    max_prot_len=512,
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    dropout=0.3,
    drop_path=0.1,
    peak_lr=1e-4,
    notes="BiCA v2 best combo: ESM-2 150M per-residue × ChemBERTa-77M-MTR per-token",
)

# ── Group: targeted — Best ligand × best protein combos not yet run ───────────
# Generated by generate_targeted_experiments.py:
#   Top ligand reprs: ecfp6_1024, chemberta_600, smiles_char (lowest mean RMSE)
#   Top protein reprs: kmer3_8000, prot_electra_256, protein_char
# Covers all model families × these top representations.

# Ridge (linear)
register("ridge_ecfp6_1024_kmer3_8000",
    group="targeted", model_family="linear",
    ligand_repr="ecfp6_1024", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: Ridge + ECFP6 + kmer3")
register("ridge_ecfp6_1024_prot_electra_256",
    group="targeted", model_family="linear",
    ligand_repr="ecfp6_1024", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: Ridge + ECFP6 + ProtElectra")
register("ridge_ecfp6_1024_protein_char",
    group="targeted", model_family="linear",
    ligand_repr="ecfp6_1024", protein_repr="protein_char", fusion="concat",
    notes="Targeted: Ridge + ECFP6 + protein_char")
register("ridge_chemberta_600_kmer3_8000",
    group="targeted", model_family="linear",
    ligand_repr="chemberta_600", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: Ridge + ChemBERTa + kmer3")
register("ridge_chemberta_600_prot_electra_256",
    group="targeted", model_family="linear",
    ligand_repr="chemberta_600", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: Ridge + ChemBERTa + ProtElectra")
register("ridge_chemberta_600_protein_char",
    group="targeted", model_family="linear",
    ligand_repr="chemberta_600", protein_repr="protein_char", fusion="concat",
    notes="Targeted: Ridge + ChemBERTa + protein_char")
register("ridge_smiles_char_kmer3_8000",
    group="targeted", model_family="linear",
    ligand_repr="smiles_char", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: Ridge + smiles_char + kmer3")
register("ridge_smiles_char_prot_electra_256",
    group="targeted", model_family="linear",
    ligand_repr="smiles_char", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: Ridge + smiles_char + ProtElectra")
register("ridge_smiles_char_protein_char",
    group="targeted", model_family="linear",
    ligand_repr="smiles_char", protein_repr="protein_char", fusion="concat",
    notes="Targeted: Ridge + smiles_char + protein_char")

# XGBoost
register("xgb_ecfp6_1024_kmer3_8000",
    group="targeted", model_family="tree",
    ligand_repr="ecfp6_1024", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: XGB + ECFP6 + kmer3")
register("xgb_ecfp6_1024_prot_electra_256",
    group="targeted", model_family="tree",
    ligand_repr="ecfp6_1024", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: XGB + ECFP6 + ProtElectra")
register("xgb_ecfp6_1024_protein_char",
    group="targeted", model_family="tree",
    ligand_repr="ecfp6_1024", protein_repr="protein_char", fusion="concat",
    notes="Targeted: XGB + ECFP6 + protein_char")
register("xgb_chemberta_600_kmer3_8000",
    group="targeted", model_family="tree",
    ligand_repr="chemberta_600", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: XGB + ChemBERTa + kmer3")
register("xgb_chemberta_600_prot_electra_256",
    group="targeted", model_family="tree",
    ligand_repr="chemberta_600", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: XGB + ChemBERTa + ProtElectra")
register("xgb_chemberta_600_protein_char",
    group="targeted", model_family="tree",
    ligand_repr="chemberta_600", protein_repr="protein_char", fusion="concat",
    notes="Targeted: XGB + ChemBERTa + protein_char")
register("xgb_smiles_char_kmer3_8000",
    group="targeted", model_family="tree",
    ligand_repr="smiles_char", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: XGB + smiles_char + kmer3")
register("xgb_smiles_char_prot_electra_256",
    group="targeted", model_family="tree",
    ligand_repr="smiles_char", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: XGB + smiles_char + ProtElectra")
register("xgb_smiles_char_protein_char",
    group="targeted", model_family="tree",
    ligand_repr="smiles_char", protein_repr="protein_char", fusion="concat",
    notes="Targeted: XGB + smiles_char + protein_char")

# MLP
register("mlp_ecfp6_1024_kmer3_8000",
    group="targeted", model_family="mlp", mlp_arch="medium",
    ligand_repr="ecfp6_1024", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: MLP + ECFP6 + kmer3")
register("mlp_ecfp6_1024_prot_electra_256",
    group="targeted", model_family="mlp", mlp_arch="medium",
    ligand_repr="ecfp6_1024", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: MLP + ECFP6 + ProtElectra")
register("mlp_ecfp6_1024_protein_char",
    group="targeted", model_family="mlp", mlp_arch="medium",
    ligand_repr="ecfp6_1024", protein_repr="protein_char", fusion="concat",
    notes="Targeted: MLP + ECFP6 + protein_char")
register("mlp_chemberta_600_kmer3_8000",
    group="targeted", model_family="mlp", mlp_arch="medium",
    ligand_repr="chemberta_600", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: MLP + ChemBERTa + kmer3")
register("mlp_chemberta_600_prot_electra_256",
    group="targeted", model_family="mlp", mlp_arch="medium",
    ligand_repr="chemberta_600", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: MLP + ChemBERTa + ProtElectra")
register("mlp_chemberta_600_protein_char",
    group="targeted", model_family="mlp", mlp_arch="medium",
    ligand_repr="chemberta_600", protein_repr="protein_char", fusion="concat",
    notes="Targeted: MLP + ChemBERTa + protein_char")
register("mlp_smiles_char_kmer3_8000",
    group="targeted", model_family="mlp", mlp_arch="medium",
    ligand_repr="smiles_char", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: MLP + smiles_char + kmer3")
register("mlp_smiles_char_prot_electra_256",
    group="targeted", model_family="mlp", mlp_arch="medium",
    ligand_repr="smiles_char", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: MLP + smiles_char + ProtElectra")
register("mlp_smiles_char_protein_char",
    group="targeted", model_family="mlp", mlp_arch="medium",
    ligand_repr="smiles_char", protein_repr="protein_char", fusion="concat",
    notes="Targeted: MLP + smiles_char + protein_char")

# Transformer (flat self-attention)
register("transformer_ecfp6_1024_kmer3_8000",
    group="targeted", model_family="transformer",
    ligand_repr="ecfp6_1024", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: Transformer + ECFP6 + kmer3")
register("transformer_ecfp6_1024_prot_electra_256",
    group="targeted", model_family="transformer",
    ligand_repr="ecfp6_1024", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: Transformer + ECFP6 + ProtElectra")
register("transformer_ecfp6_1024_protein_char",
    group="targeted", model_family="transformer",
    ligand_repr="ecfp6_1024", protein_repr="protein_char", fusion="concat",
    notes="Targeted: Transformer + ECFP6 + protein_char")
register("transformer_chemberta_600_kmer3_8000",
    group="targeted", model_family="transformer",
    ligand_repr="chemberta_600", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: Transformer + ChemBERTa + kmer3")
register("transformer_chemberta_600_prot_electra_256",
    group="targeted", model_family="transformer",
    ligand_repr="chemberta_600", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: Transformer + ChemBERTa + ProtElectra")
register("transformer_chemberta_600_protein_char",
    group="targeted", model_family="transformer",
    ligand_repr="chemberta_600", protein_repr="protein_char", fusion="concat",
    notes="Targeted: Transformer + ChemBERTa + protein_char")
register("transformer_smiles_char_kmer3_8000",
    group="targeted", model_family="transformer",
    ligand_repr="smiles_char", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: Transformer + smiles_char + kmer3")
register("transformer_smiles_char_prot_electra_256",
    group="targeted", model_family="transformer",
    ligand_repr="smiles_char", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: Transformer + smiles_char + ProtElectra")
register("transformer_smiles_char_protein_char",
    group="targeted", model_family="transformer",
    ligand_repr="smiles_char", protein_repr="protein_char", fusion="concat",
    notes="Targeted: Transformer + smiles_char + protein_char")

# BiCA
register("bica_ecfp6_1024_kmer3_8000",
    group="targeted", model_family="bica",
    ligand_repr="ecfp6_1024", protein_repr="kmer3_8000", fusion="bidirectional_cross_attn",
    notes="Targeted: BiCA + ECFP6 + kmer3")
register("bica_ecfp6_1024_prot_electra_256",
    group="targeted", model_family="bica",
    ligand_repr="ecfp6_1024", protein_repr="prot_electra_256", fusion="bidirectional_cross_attn",
    notes="Targeted: BiCA + ECFP6 + ProtElectra")
register("bica_ecfp6_1024_protein_char",
    group="targeted", model_family="bica",
    ligand_repr="ecfp6_1024", protein_repr="protein_char", fusion="bidirectional_cross_attn",
    notes="Targeted: BiCA + ECFP6 + protein_char")
register("bica_chemberta_600_kmer3_8000",
    group="targeted", model_family="bica",
    ligand_repr="chemberta_600", protein_repr="kmer3_8000", fusion="bidirectional_cross_attn",
    notes="Targeted: BiCA + ChemBERTa + kmer3")
register("bica_chemberta_600_prot_electra_256",
    group="targeted", model_family="bica",
    ligand_repr="chemberta_600", protein_repr="prot_electra_256", fusion="bidirectional_cross_attn",
    notes="Targeted: BiCA + ChemBERTa + ProtElectra")
register("bica_chemberta_600_protein_char",
    group="targeted", model_family="bica",
    ligand_repr="chemberta_600", protein_repr="protein_char", fusion="bidirectional_cross_attn",
    notes="Targeted: BiCA + ChemBERTa + protein_char")
register("bica_smiles_char_kmer3_8000",
    group="targeted", model_family="bica",
    ligand_repr="smiles_char", protein_repr="kmer3_8000", fusion="bidirectional_cross_attn",
    notes="Targeted: BiCA + smiles_char + kmer3")
register("bica_smiles_char_prot_electra_256",
    group="targeted", model_family="bica",
    ligand_repr="smiles_char", protein_repr="prot_electra_256", fusion="bidirectional_cross_attn",
    notes="Targeted: BiCA + smiles_char + ProtElectra")
register("bica_smiles_char_protein_char",
    group="targeted", model_family="bica",
    ligand_repr="smiles_char", protein_repr="protein_char", fusion="bidirectional_cross_attn",
    notes="Targeted: BiCA + smiles_char + protein_char")

# CNN (ligand fixed to smiles_onehot)
register("cnn_smiles_onehot_kmer3_8000",
    group="targeted", model_family="cnn",
    ligand_repr="smiles_onehot", protein_repr="kmer3_8000", fusion="concat",
    notes="Targeted: CNN + smiles_onehot + kmer3")
register("cnn_smiles_onehot_prot_electra_256",
    group="targeted", model_family="cnn",
    ligand_repr="smiles_onehot", protein_repr="prot_electra_256", fusion="concat",
    notes="Targeted: CNN + smiles_onehot + ProtElectra")
register("cnn_smiles_onehot_protein_char",
    group="targeted", model_family="cnn",
    ligand_repr="smiles_onehot", protein_repr="protein_char", fusion="concat",
    notes="Targeted: CNN + smiles_onehot + protein_char")

# DistmatCNN (ligand fixed to distmat_100)
register("distmat_cnn_kmer3_8000",
    group="targeted", model_family="distmat_cnn",
    ligand_repr="distmat_100", protein_repr="kmer3_8000", fusion="distmat",
    notes="Targeted: DistmatCNN + kmer3")
register("distmat_cnn_prot_electra_256",
    group="targeted", model_family="distmat_cnn",
    ligand_repr="distmat_100", protein_repr="prot_electra_256", fusion="distmat",
    notes="Targeted: DistmatCNN + ProtElectra")
register("distmat_cnn_protein_char",
    group="targeted", model_family="distmat_cnn",
    ligand_repr="distmat_100", protein_repr="protein_char", fusion="distmat",
    notes="Targeted: DistmatCNN + protein_char")

# GCN (ligand fixed to mol_graph)
register("gcn_mol_graph_kmer3_8000",
    group="targeted", model_family="gcn",
    ligand_repr="mol_graph", protein_repr="kmer3_8000", fusion="graph",
    notes="Targeted: GCN + kmer3")
register("gcn_mol_graph_prot_electra_256",
    group="targeted", model_family="gcn",
    ligand_repr="mol_graph", protein_repr="prot_electra_256", fusion="graph",
    notes="Targeted: GCN + ProtElectra")
register("gcn_mol_graph_protein_char",
    group="targeted", model_family="gcn",
    ligand_repr="mol_graph", protein_repr="protein_char", fusion="graph",
    notes="Targeted: GCN + protein_char")

# GAT (ligand fixed to mol_graph)
register("gat_mol_graph_kmer3_8000",
    group="targeted", model_family="gat",
    ligand_repr="mol_graph", protein_repr="kmer3_8000", fusion="graph",
    notes="Targeted: GAT + kmer3")
register("gat_mol_graph_prot_electra_256",
    group="targeted", model_family="gat",
    ligand_repr="mol_graph", protein_repr="prot_electra_256", fusion="graph",
    notes="Targeted: GAT + ProtElectra")
register("gat_mol_graph_protein_char",
    group="targeted", model_family="gat",
    ligand_repr="mol_graph", protein_repr="protein_char", fusion="graph",
    notes="Targeted: GAT + protein_char")

# Sequence models — new tokenizer combos (lstm)
register("lstm_smiles_char_protein_bpe_1000",
    group="targeted", model_family="lstm",
    lig_tok="smiles_char", prot_tok="protein_bpe_1000", fusion="dual_encoder",
    notes="Targeted: LSTM + smiles_char + protein_bpe_1000")
register("lstm_smiles_atom_protein_bpe_1000",
    group="targeted", model_family="lstm",
    lig_tok="smiles_atom", prot_tok="protein_bpe_1000", fusion="dual_encoder",
    notes="Targeted: LSTM + smiles_atom + protein_bpe_1000")
register("lstm_smiles_bpe_1000_protein_bpe_1000",
    group="targeted", model_family="lstm",
    lig_tok="smiles_bpe_1000", prot_tok="protein_bpe_1000", fusion="dual_encoder",
    notes="Targeted: LSTM + smiles_bpe_1000 + protein_bpe_1000")
register("lstm_smiles_bpe_1000_protein_char",
    group="targeted", model_family="lstm",
    lig_tok="smiles_bpe_1000", prot_tok="protein_char", fusion="dual_encoder",
    notes="Targeted: LSTM + smiles_bpe_1000 + protein_char")

# Sequence models — new tokenizer combos (transformer_seq)
register("transformer_seq_smiles_char_protein_bpe_1000",
    group="targeted", model_family="transformer_seq",
    lig_tok="smiles_char", prot_tok="protein_bpe_1000", fusion="dual_encoder",
    notes="Targeted: TransformerSeq + smiles_char + protein_bpe_1000")
register("transformer_seq_smiles_atom_protein_bpe_1000",
    group="targeted", model_family="transformer_seq",
    lig_tok="smiles_atom", prot_tok="protein_bpe_1000", fusion="dual_encoder",
    notes="Targeted: TransformerSeq + smiles_atom + protein_bpe_1000")
register("transformer_seq_smiles_bpe_1000_protein_bpe_1000",
    group="targeted", model_family="transformer_seq",
    lig_tok="smiles_bpe_1000", prot_tok="protein_bpe_1000", fusion="dual_encoder",
    notes="Targeted: TransformerSeq + smiles_bpe_1000 + protein_bpe_1000")
register("transformer_seq_smiles_bpe_1000_protein_char",
    group="targeted", model_family="transformer_seq",
    lig_tok="smiles_bpe_1000", prot_tok="protein_char", fusion="dual_encoder",
    notes="Targeted: TransformerSeq + smiles_bpe_1000 + protein_char")


# ─────────────────────────────────────────────────────────────────────────────
# Feature builders
# ─────────────────────────────────────────────────────────────────────────────

def build_features(cfg: dict, train_df, val_df, test_df):
    """Build and return (X_train, X_val, X_test, y_train, y_val, y_test)."""
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL
    from pathlib import Path

    FEAT_CACHE = Path("cache/features")
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    pfx = _feat_cache_prefix()   # dataset/seed prefix so caches don't collide

    def _get_or_compute(tag, fn, df_part):
        fpath = FEAT_CACHE / f"{pfx}{tag}_{df_part}.npy"
        if fpath.exists():
            return np.load(fpath)
        arr = fn()
        np.save(fpath, arr)
        return arr

    lig_repr  = cfg["ligand_repr"]
    prot_repr = cfg["protein_repr"]

    def lig_feat(df, part):
        smiles = df[SMILES_COL].tolist()
        tag = f"lig_{lig_repr}"
        return _get_or_compute(f"{tag}_{part}", lambda: _compute_lig(smiles, lig_repr), part)

    def prot_feat(df, part):
        seqs = df[PROTEIN_COL].tolist()
        tag = f"prot_{prot_repr}"
        return _get_or_compute(f"{tag}_{part}", lambda: _compute_prot(seqs, prot_repr), part)

    print(f"[features] Computing ligand features: {lig_repr}")
    L_train = lig_feat(train_df, "train")
    L_val   = lig_feat(val_df,   "val")
    L_test  = lig_feat(test_df,  "test")

    if prot_repr == "none":
        # ponytail: ligand-only features — no protein
        X_train = L_train
        X_val   = L_val
        X_test  = L_test
        P_train = np.zeros((len(train_df), 0))
        P_val   = np.zeros((len(val_df), 0))
        P_test  = np.zeros((len(test_df), 0))
    else:
        print(f"[features] Computing protein features: {prot_repr}")
        P_train = prot_feat(train_df, "train")
        P_val   = prot_feat(val_df,   "val")
        P_test  = prot_feat(test_df,  "test")
        X_train = F.concat(L_train, P_train)
        X_val   = F.concat(L_val,   P_val)
        X_test  = F.concat(L_test,  P_test)

    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val   = val_df[LABEL_COL].values.astype(np.float32)
    y_test  = test_df[LABEL_COL].values.astype(np.float32)

    # Also return separate lig/prot for cross-attention models
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            L_train, L_val, L_test, P_train, P_val, P_test)


def _compute_lig(smiles, repr_name):
    if repr_name.startswith("chemberta_"):
        model_map = {
            "chemberta_5M": "DeepChem/ChemBERTa-5M-MLM",
            "chemberta_77M": "DeepChem/ChemBERTa-77M-MLM",
            "chemberta_100M": "DeepChem/ChemBERTa-100M-MLM",
            "chemberta_600": "seyonec/ChemBERTa-zinc-base-v1",
        }
        model = model_map.get(repr_name, "seyonec/ChemBERTa-zinc-base-v1")
        return F.chemberta_embeddings(smiles, model_name=model)
    elif repr_name.startswith("esmc_"):
        model = "EvolutionaryScale/esmc-300m-2024-12"
        return F.esmc_embeddings(smiles, model_name=model)
    elif repr_name == "ecfp2_1024":
        return F.ecfp(smiles, radius=1, nbits=1024)
    elif repr_name == "ecfp4_1024":
        return F.ecfp(smiles, radius=2, nbits=1024)
    elif repr_name == "ecfp6_1024":
        return F.ecfp(smiles, radius=3, nbits=1024)
    elif repr_name == "maccs_167":
        return F.maccs_keys(smiles)
    elif repr_name == "rdkit_200":
        return F.rdkit_descriptors(smiles)
    elif repr_name in ("smiles_onehot", "smiles_char"):
        # smiles_char = character-level one-hot (same as smiles_onehot for flat models)
        return F.smiles_char_onehot(smiles, max_len=100)
    elif repr_name == "chemberta_600":
        return F.chemberta_embeddings(smiles)
    elif repr_name == "distmat_100":
        return F.smiles_distmat(smiles, max_atoms=100)
    # mol_graph is handled separately in GNN path — not a flat array
    else:
        raise ValueError(f"Unknown ligand repr: {repr_name}")


def _compute_prot(seqs, repr_name):
    if repr_name == "aac_20":
        return F.amino_acid_composition(seqs)
    elif repr_name == "dipeptide_400":
        return F.dipeptide_composition(seqs)
    elif repr_name == "kmer3_8000":
        return F.kmer_frequency(seqs, k=3, max_features=8000)
    elif repr_name.startswith("esm2_"):
        size = repr_name.split("_")[1]  # "8M", "35M", "150M", "650M"
        return F.esm2_embeddings(seqs, model_size=size)
    elif repr_name == "prot_electra_256":
        return F.prot_electra_embeddings(seqs)
    elif repr_name == "protein_char":
        # protein_char as flat repr = amino-acid composition (20-dim)
        # It's a tokenizer for seq models, but for flat models we use AAC
        return F.amino_acid_composition(seqs)
    elif repr_name == "esmc_300M":
        return F.esmc_embeddings(seqs, model_name="esmc_300m")
    else:
        raise ValueError(f"Unknown protein repr: {repr_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_sklearn_model(cfg: dict, exp_name: str):
    from models import sklearn_models as M
    name = exp_name
    if name.startswith("ridge"):
        return M.ridge()
    elif name.startswith("lasso"):
        return M.lasso()
    elif name.startswith("svr"):
        return M.svr_rbf()
    elif name.startswith("rf"):
        return M.random_forest()
    elif name.startswith("xgb"):
        return M.xgboost_model()
    elif name.startswith("lgbm"):
        return M.lightgbm_model()
    elif name.startswith("gp"):
        kernel = cfg.get("gp_kernel", "rbf")
        builders = {
            "tanimoto": M.gp_tanimoto,
            "rbf": M.gp_rbf,
            "matern": M.gp_matern,
            "rq": M.gp_rq,
        }
        return builders[kernel]()
    else:
        raise ValueError(f"Cannot infer sklearn model from name: {name}")


def build_torch_model(cfg: dict, input_dim: int, lig_dim: int = 0, prot_dim: int = 0):
    from models.mlp import mlp_shallow, mlp_medium, mlp_deep, mlp_wide
    from models.cnn import build_smiles_cnn
    from models.transformer import TransformerRegressor, CrossAttentionFusion

    mf   = cfg["model_family"]
    arch = cfg.get("mlp_arch", "medium")
    ta   = cfg.get("transformer_arch", "self_attn")

    if mf == "mlp":
        builders = {"shallow": mlp_shallow, "medium": mlp_medium,
                    "deep": mlp_deep, "wide": mlp_wide}
        return builders[arch](input_dim)

    elif mf == "cnn":
        # Pass lig_dim so vocab_size is derived from actual feature dim, not hardcoded
        return build_smiles_cnn(smiles_max_len=100, input_dim=lig_dim)

    elif mf == "transformer":
        if ta == "cross_attn":
            return CrossAttentionFusion(lig_dim, prot_dim, d_model=256, nhead=8, num_layers=2)
        else:
            return TransformerRegressor(input_dim, d_model=256, nhead=8, num_layers=3, token_size=64)

    else:
        raise ValueError(f"Unknown model_family: {mf}")


# ─────────────────────────────────────────────────────────────────────────────
# Sequence experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_distmat(exp_name: str):
    """Run a distance-matrix CNN experiment."""
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL, BATCH_SIZE, LEARNING_RATE
    from models.distmat_cnn import build_distmat_cnn, MAX_ATOMS
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from harness.trainer import _get_device, count_parameters
    from harness.config import MAX_EPOCHS, PATIENCE

    cfg = EXPERIMENTS[exp_name]
    print(f"\n{'='*60}\n  Experiment : {exp_name}\n{'='*60}")

    train_df, val_df, test_df = _get_splits()

    from harness.config import SMILES_COL, PROTEIN_COL, LABEL_COL
    from pathlib import Path
    FEAT_CACHE = Path("cache/features")
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    pfx = _feat_cache_prefix()

    prot_repr = cfg["protein_repr"]

    def _load_prot(df_part, df_):
        tag   = f"{pfx}prot_{prot_repr}_{df_part}"
        fpath = FEAT_CACHE / f"{tag}.npy"
        print(f"  [debug] Checking {fpath}")

        if fpath.exists():
            print(f"  [debug] Loading from cache")

            return np.load(fpath)
        print(f"  [debug] Computing from scratch for {len(df_)} sequences")

        arr = _compute_prot(df_[PROTEIN_COL].tolist(), prot_repr)
        np.save(fpath, arr)
        return arr

    def _load_distmat(df_part, df_):
        tag   = f"{pfx}lig_distmat_100_{df_part}"
        fpath = FEAT_CACHE / f"{tag}.npy"
        if fpath.exists():
            return np.load(fpath).reshape(-1, MAX_ATOMS, MAX_ATOMS)
        arr = F.smiles_distmat(df_[SMILES_COL].tolist(), max_atoms=MAX_ATOMS)
        np.save(fpath, arr.reshape(-1, MAX_ATOMS * MAX_ATOMS))
        return arr.reshape(-1, MAX_ATOMS, MAX_ATOMS)

    print("[distmat] Loading distance matrices …")
    DM_train = _load_distmat("train", train_df)
    DM_val   = _load_distmat("val",   val_df)
    DM_test  = _load_distmat("test",  test_df)

    print(f"[distmat] Loading protein features: {prot_repr} …")
    P_train = _load_prot("train", train_df)
    P_val   = _load_prot("val",   val_df)
    P_test  = _load_prot("test",  test_df)

    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val   = val_df[LABEL_COL].values.astype(np.float32)
    y_test  = test_df[LABEL_COL].values.astype(np.float32)

    prot_dim = P_train.shape[1]
    model    = build_distmat_cnn(prot_dim=prot_dim)
    n_params = count_parameters(model)
    device   = _get_device()
    model    = model.to(device)

    # Build loaders
    def _make_loader(DM, P, y, shuffle):
        ds = TensorDataset(
            torch.tensor(DM, dtype=torch.float32),
            torch.tensor(P,  dtype=torch.float32),
            torch.tensor(y,  dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = _make_loader(DM_train, P_train, y_train, shuffle=True)
    val_loader   = _make_loader(DM_val,   P_val,   y_val,   shuffle=False)
    test_loader  = _make_loader(DM_test,  P_test,  y_test,  shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )

    best_rmse, best_state, patience_ctr = float("inf"), None, 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for dm_b, p_b, y_b in train_loader:
            dm_b, p_b, y_b = dm_b.to(device), p_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(dm_b, p_b).squeeze(-1)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = []
            for dm_b, p_b, _ in val_loader:
                val_preds.append(model(dm_b.to(device), p_b.to(device)).cpu().numpy().ravel())
        val_pred = np.concatenate(val_preds)
        val_m    = compute_metrics(y_val, val_pred)
        scheduler.step(val_m["rmse"])

        if val_m["rmse"] < best_rmse:
            best_rmse, patience_ctr = val_m["rmse"], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch}  val_rmse={val_m['rmse']:.4f}  patience={patience_ctr}")
        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds, test_preds = [], []
        for dm_b, p_b, _ in val_loader:
            val_preds.append(model(dm_b.to(device), p_b.to(device)).cpu().numpy().ravel())
        for dm_b, p_b, _ in test_loader:
            test_preds.append(model(dm_b.to(device), p_b.to(device)).cpu().numpy().ravel())
    test_pred_arr = np.concatenate(test_preds)
    val_m  = compute_metrics(y_val,  np.concatenate(val_preds))
    test_m = compute_metrics(y_test, test_pred_arr)
    train_time  = time.time() - t0
    save_predictions(_exp_id(exp_name), y_test, test_pred_arr)

    print(f"[distmat] Val  → RMSE={val_m['rmse']:.4f}  Pearson={val_m['pearson_r']:.4f}")
    print(f"[distmat] Test → RMSE={test_m['rmse']:.4f}  Pearson={test_m['pearson_r']:.4f}")

    log_result(
        experiment_id  = _exp_id(exp_name),
        model_name     = _exp_id(exp_name),
        model_family   = "distmat_cnn",
        ligand_repr    = cfg["ligand_repr"],
        protein_repr   = cfg["protein_repr"],
        fusion_strategy= cfg.get("fusion", "distmat"),
        n_params       = n_params,
        epochs_trained = epoch,
        batch_size     = BATCH_SIZE,
        learning_rate  = LEARNING_RATE,
        split_type     = _split_tag(),
        n_train        = len(y_train),
        n_val          = len(y_val),
        n_test         = len(y_test),
        val_metrics    = val_m,
        test_metrics   = test_m,
        train_time_sec = train_time,
        notes          = cfg.get("notes", ""),
    )


def run_gnn(exp_name: str):
    """Run a GNN experiment (GCN or GAT) using PyTorch Geometric."""
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL, BATCH_SIZE, LEARNING_RATE
    from harness.gnn_trainer import make_gnn_loader, train_gnn_model
    from harness.trainer import count_parameters
    from models.gnn import GCNBindingModel, GATBindingModel
    from pathlib import Path

    cfg = EXPERIMENTS[exp_name]
    mf  = cfg["model_family"]
    print(f"\n{'='*60}\n  Experiment : {exp_name}\n{'='*60}")

    train_df, val_df, test_df = _get_splits()

    prot_repr = cfg["protein_repr"]
    FEAT_CACHE = Path("cache/features")
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    pfx = _feat_cache_prefix()

    def _load_prot(df_part, df_):
        tag   = f"{pfx}prot_{prot_repr}_{df_part}"
        fpath = FEAT_CACHE / f"{tag}.npy"
        if fpath.exists():
            return np.load(fpath)
        arr = _compute_prot(df_[PROTEIN_COL].tolist(), prot_repr)
        np.save(fpath, arr)
        return arr

    print(f"[gnn] Loading protein features: {prot_repr} …")
    P_train = _load_prot("train", train_df)
    P_val   = _load_prot("val",   val_df)
    P_test  = _load_prot("test",  test_df)

    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val   = val_df[LABEL_COL].values.astype(np.float32)
    y_test  = test_df[LABEL_COL].values.astype(np.float32)

    prot_dim = P_train.shape[1]

    if mf == "gcn":
        model = GCNBindingModel(node_dim=78, hidden_dim=128, prot_dim=prot_dim)
    elif mf == "gat":
        model = GATBindingModel(node_dim=78, hidden_dim=128, prot_dim=prot_dim, heads=4)
    else:
        raise ValueError(f"Unknown GNN family: {mf}")

    # Attach NodeDecoder if GraphMAE reconstruction is requested
    use_recon = cfg.get("use_node_recon", False)
    if use_recon:
        from models.gnn import NodeDecoder
        model.node_decoder = NodeDecoder(hidden_dim=128)
        print("[gnn] GraphMAE node reconstruction enabled (mask_rate=0.15, lambda=0.1)")

    # Pairwise ranking loss
    gnn_aux_loss = None
    if cfg.get("use_ranking_loss"):
        from harness.losses import PairwiseRankingLoss
        gnn_aux_loss = PairwiseRankingLoss(margin=0.5, n_pairs=32, lambda_rank=0.1)
        print("[gnn] Pairwise ranking loss enabled (margin=0.5, lambda=0.1)")

    n_params = count_parameters(model)
    print(f"[gnn] Params: {n_params:,}")
    print("[gnn] Building data loaders (graph conversion) …")

    train_loader = make_gnn_loader(
        train_df[SMILES_COL].tolist(), P_train, y_train, shuffle=True,  batch_size=BATCH_SIZE)
    val_loader   = make_gnn_loader(
        val_df[SMILES_COL].tolist(),   P_val,   y_val,   shuffle=False, batch_size=BATCH_SIZE * 4)
    test_loader  = make_gnn_loader(
        test_df[SMILES_COL].tolist(),  P_test,  y_test,  shuffle=False, batch_size=BATCH_SIZE * 4)

    val_m, test_m, train_time, epoch, test_pred = train_gnn_model(
        model, train_loader, val_loader, test_loader, lr=LEARNING_RATE,
        aux_loss=gnn_aux_loss,
        use_node_recon=use_recon,
    )
    save_predictions(_exp_id(exp_name), y_test, test_pred)

    print(f"[gnn] Val  → RMSE={val_m['rmse']:.4f}  Pearson={val_m['pearson_r']:.4f}")
    print(f"[gnn] Test → RMSE={test_m['rmse']:.4f}  Pearson={test_m['pearson_r']:.4f}")

    log_result(
        experiment_id  = _exp_id(exp_name),
        model_name     = _exp_id(exp_name),
        model_family   = mf,
        ligand_repr    = cfg["ligand_repr"],
        protein_repr   = cfg["protein_repr"],
        fusion_strategy= cfg.get("fusion", "graph"),
        n_params       = n_params,
        epochs_trained = epoch,
        batch_size     = BATCH_SIZE,
        learning_rate  = LEARNING_RATE,
        split_type     = _split_tag(),
        n_train        = len(y_train),
        n_val          = len(y_val),
        n_test         = len(y_test),
        val_metrics    = val_m,
        test_metrics   = test_m,
        train_time_sec = train_time,
        notes          = cfg.get("notes", ""),
    )


def run_bica(exp_name: str):
    """Run a BiCA bidirectional cross-attention experiment."""
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL, BATCH_SIZE, LEARNING_RATE
    from harness.trainer import _get_device, count_parameters
    from harness.config import MAX_EPOCHS, PATIENCE
    from models.bica import build_bica
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from pathlib import Path

    cfg = EXPERIMENTS[exp_name]
    print(f"\n{'='*60}\n  Experiment : {exp_name}\n{'='*60}")

    train_df, val_df, test_df = _get_splits()

    FEAT_CACHE = Path("cache/features")
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    pfx = _feat_cache_prefix()

    lig_repr  = cfg["ligand_repr"]
    prot_repr = cfg["protein_repr"]

    def _load(tag, fn):
        fpath = FEAT_CACHE / f"{pfx}{tag}.npy"
        if fpath.exists():
            return np.load(fpath)
        arr = fn()
        np.save(fpath, arr)
        return arr

    print(f"[bica] Ligand features: {lig_repr}")
    L_train = _load(f"lig_{lig_repr}_train",
                    lambda: _compute_lig(train_df[SMILES_COL].tolist(), lig_repr))
    L_val   = _load(f"lig_{lig_repr}_val",
                    lambda: _compute_lig(val_df[SMILES_COL].tolist(), lig_repr))
    L_test  = _load(f"lig_{lig_repr}_test",
                    lambda: _compute_lig(test_df[SMILES_COL].tolist(), lig_repr))

    print(f"[bica] Protein features: {prot_repr}")
    P_train = _load(f"prot_{prot_repr}_train",
                    lambda: _compute_prot(train_df[PROTEIN_COL].tolist(), prot_repr))
    P_val   = _load(f"prot_{prot_repr}_val",
                    lambda: _compute_prot(val_df[PROTEIN_COL].tolist(), prot_repr))
    P_test  = _load(f"prot_{prot_repr}_test",
                    lambda: _compute_prot(test_df[PROTEIN_COL].tolist(), prot_repr))

    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val   = val_df[LABEL_COL].values.astype(np.float32)
    y_test  = test_df[LABEL_COL].values.astype(np.float32)

    lig_dim  = L_train.shape[1]
    prot_dim = P_train.shape[1]
    model    = build_bica(protein_dim=prot_dim, ligand_dim=lig_dim,
                          hidden_dim=256, num_heads=8, dropout=0.1)
    n_params = count_parameters(model)
    device   = _get_device()
    model    = model.to(device)
    print(f"[bica] Params: {n_params:,}  lig_dim={lig_dim}  prot_dim={prot_dim}")

    def _make_loader(L, P, y, shuffle):
        ds = TensorDataset(
            torch.tensor(L, dtype=torch.float32),
            torch.tensor(P, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = _make_loader(L_train, P_train, y_train, shuffle=True)
    val_loader   = _make_loader(L_val,   P_val,   y_val,   shuffle=False)
    test_loader  = _make_loader(L_test,  P_test,  y_test,  shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    bica_aux_loss = None
    if cfg.get("use_ranking_loss"):
        from harness.losses import PairwiseRankingLoss
        bica_aux_loss = PairwiseRankingLoss(margin=0.5, n_pairs=32, lambda_rank=0.1)
        bica_aux_loss = bica_aux_loss.to(device)
        print("[bica] Pairwise ranking loss enabled (margin=0.5, lambda=0.1)")

    dsm_head = None
    if cfg.get("use_dsm"):
        from models.dsm import DSMAuxHead, dsm_loss as _dsm_loss
        embed_dim = model.hidden_dim * 2   # BiCA embedding is (hidden_dim*2,)
        dsm_head  = DSMAuxHead(embed_dim=embed_dim).to(device)
        print(f"[bica] DualBind DSM auxiliary loss enabled (embed_dim={embed_dim})")

    best_rmse, best_state, patience_ctr = float("inf"), None, 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        if dsm_head is not None:
            dsm_head.train()
        for l_b, p_b, y_b in train_loader:
            l_b, p_b, y_b = l_b.to(device), p_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            if dsm_head is not None:
                z    = model.encode(p_b, l_b)          # (B, H*2)
                pred = model.predict_from_embedding(z).squeeze(-1)
                loss = criterion(pred, y_b) + _dsm_loss(dsm_head, z)
            else:
                pred = model(p_b, l_b).squeeze(-1)   # BiCA: (protein, ligand)
                loss = criterion(pred, y_b)
            if bica_aux_loss is not None:
                loss = loss + bica_aux_loss(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = []
            for l_b, p_b, _ in val_loader:
                val_preds.append(model(p_b.to(device), l_b.to(device)).cpu().numpy().ravel())
        val_m = compute_metrics(y_val, np.concatenate(val_preds))
        scheduler.step(val_m["rmse"])

        if val_m["rmse"] < best_rmse:
            best_rmse, patience_ctr = val_m["rmse"], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch}  val_rmse={val_m['rmse']:.4f}  patience={patience_ctr}")
        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds, test_preds = [], []
        for l_b, p_b, _ in val_loader:
            val_preds.append(model(p_b.to(device), l_b.to(device)).cpu().numpy().ravel())
        for l_b, p_b, _ in test_loader:
            test_preds.append(model(p_b.to(device), l_b.to(device)).cpu().numpy().ravel())
    test_pred_arr = np.concatenate(test_preds)
    val_m  = compute_metrics(y_val,  np.concatenate(val_preds))
    test_m = compute_metrics(y_test, test_pred_arr)
    train_time = time.time() - t0
    save_predictions(_exp_id(exp_name), y_test, test_pred_arr)

    print(f"[bica] Val  → RMSE={val_m['rmse']:.4f}  Pearson={val_m['pearson_r']:.4f}")
    print(f"[bica] Test → RMSE={test_m['rmse']:.4f}  Pearson={test_m['pearson_r']:.4f}")

    # Save best model checkpoint
    checkpoint_dir = Path("cache/models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{_exp_id(exp_name)}.pt"
    torch.save(best_state, checkpoint_path)
    print(f"[bica] Saved best model to {checkpoint_path}")

    log_result(
        experiment_id  = _exp_id(exp_name),
        model_name     = _exp_id(exp_name),
        model_family   = "bica",
        ligand_repr    = cfg["ligand_repr"],
        protein_repr   = cfg["protein_repr"],
        fusion_strategy= cfg.get("fusion", "bidirectional_cross_attn"),
        n_params       = n_params,
        epochs_trained = epoch,
        batch_size     = BATCH_SIZE,
        learning_rate  = LEARNING_RATE,
        split_type     = _split_tag(),
        n_train        = len(y_train),
        n_val          = len(y_val),
        n_test         = len(y_test),
        val_metrics    = val_m,
        test_metrics   = test_m,
        train_time_sec = train_time,
        notes          = cfg.get("notes", ""),
    )


def run_graphormer(exp_name: str):
    """Run a Graphormer graph-transformer experiment."""
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL, BATCH_SIZE, LEARNING_RATE
    from harness.gnn_trainer import make_gnn_loader, train_gnn_model
    from harness.trainer import count_parameters
    from models.graphormer import GraphormerBindingModel
    from pathlib import Path

    cfg = EXPERIMENTS[exp_name]
    print(f"\n{'='*60}\n  Experiment : {exp_name}\n{'='*60}")

    train_df, val_df, test_df = _get_splits()

    prot_repr = cfg["protein_repr"]
    FEAT_CACHE = Path("cache/features")
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    pfx = _feat_cache_prefix()

    def _load_prot(df_part, df_):
        tag   = f"{pfx}prot_{prot_repr}_{df_part}"
        fpath = FEAT_CACHE / f"{tag}.npy"
        print(f"  [debug] Checking {fpath}")
        if fpath.exists():
            print(f"  [debug] Loading from cache")
            return np.load(fpath)
        print(f"  [debug] Computing from scratch for {len(df_)} sequences")
        arr = _compute_prot(df_[PROTEIN_COL].tolist(), prot_repr)
        np.save(fpath, arr)
        return arr

    print(f"[graphormer] Loading protein features: {prot_repr} …")
    P_train = _load_prot("train", train_df)
    P_val   = _load_prot("val",   val_df)
    P_test  = _load_prot("test",  test_df)

    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val   = val_df[LABEL_COL].values.astype(np.float32)
    y_test  = test_df[LABEL_COL].values.astype(np.float32)

    prot_dim = P_train.shape[1]
    model    = GraphormerBindingModel(
        node_dim=78, hidden_dim=128, prot_dim=prot_dim,
        num_heads=8, num_layers=4, dropout=0.1,
    )
    n_params = count_parameters(model)
    print(f"[graphormer] Params: {n_params:,}  prot_dim={prot_dim}")

    train_loader = make_gnn_loader(
        train_df[SMILES_COL].tolist(), P_train, y_train, shuffle=True,  batch_size=BATCH_SIZE)
    val_loader   = make_gnn_loader(
        val_df[SMILES_COL].tolist(),   P_val,   y_val,   shuffle=False, batch_size=BATCH_SIZE * 2)
    test_loader  = make_gnn_loader(
        test_df[SMILES_COL].tolist(),  P_test,  y_test,  shuffle=False, batch_size=BATCH_SIZE * 2)

    val_m, test_m, train_time, epoch, test_pred = train_gnn_model(
        model, train_loader, val_loader, test_loader, lr=LEARNING_RATE,
    )
    save_predictions(_exp_id(exp_name), y_test, test_pred)

    print(f"[graphormer] Val  → RMSE={val_m['rmse']:.4f}  Pearson={val_m['pearson_r']:.4f}")
    print(f"[graphormer] Test → RMSE={test_m['rmse']:.4f}  Pearson={test_m['pearson_r']:.4f}")

    log_result(
        experiment_id  = _exp_id(exp_name),
        model_name     = _exp_id(exp_name),
        model_family   = "graphormer",
        ligand_repr    = cfg["ligand_repr"],
        protein_repr   = cfg["protein_repr"],
        fusion_strategy= cfg.get("fusion", "graph"),
        n_params       = n_params,
        epochs_trained = epoch,
        batch_size     = BATCH_SIZE,
        learning_rate  = LEARNING_RATE,
        split_type     = _split_tag(),
        n_train        = len(y_train),
        n_val          = len(y_val),
        n_test         = len(y_test),
        val_metrics    = val_m,
        test_metrics   = test_m,
        train_time_sec = train_time,
        notes          = cfg.get("notes", ""),
    )


def run_gli(exp_name: str):
    """Run a GLI (Gated Global-Local Interaction) experiment."""
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL, BATCH_SIZE, LEARNING_RATE
    from harness.gnn_trainer import make_gnn_loader, train_gnn_model
    from harness.trainer import count_parameters
    from models.gli import GLIBindingModel
    from pathlib import Path

    cfg = EXPERIMENTS[exp_name]
    print(f"\n{'='*60}\n  Experiment : {exp_name}\n{'='*60}")

    train_df, val_df, test_df = _get_splits()

    prot_repr = cfg["protein_repr"]
    FEAT_CACHE = Path("cache/features")
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    pfx = _feat_cache_prefix()

    def _load_prot(df_part, df_):
        tag   = f"{pfx}prot_{prot_repr}_{df_part}"
        fpath = FEAT_CACHE / f"{tag}.npy"
        if fpath.exists():
            return np.load(fpath)
        arr = _compute_prot(df_[PROTEIN_COL].tolist(), prot_repr)
        np.save(fpath, arr)
        return arr

    print(f"[gli] Loading protein features: {prot_repr} …")
    P_train = _load_prot("train", train_df)
    P_val   = _load_prot("val",   val_df)
    P_test  = _load_prot("test",  test_df)

    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val   = val_df[LABEL_COL].values.astype(np.float32)
    y_test  = test_df[LABEL_COL].values.astype(np.float32)

    prot_dim = P_train.shape[1]
    model    = GLIBindingModel(
        node_dim=78, hidden_dim=128, prot_dim=prot_dim,
        gnn_type="gat", num_heads=4, dropout=0.2, num_gnn_layers=3,
    )
    n_params = count_parameters(model)
    print(f"[gli] Params: {n_params:,}  prot_dim={prot_dim}")

    train_loader = make_gnn_loader(
        train_df[SMILES_COL].tolist(), P_train, y_train, shuffle=True,  batch_size=BATCH_SIZE)
    val_loader   = make_gnn_loader(
        val_df[SMILES_COL].tolist(),   P_val,   y_val,   shuffle=False, batch_size=BATCH_SIZE * 4)
    test_loader  = make_gnn_loader(
        test_df[SMILES_COL].tolist(),  P_test,  y_test,  shuffle=False, batch_size=BATCH_SIZE * 4)

    val_m, test_m, train_time, epoch, test_pred = train_gnn_model(
        model, train_loader, val_loader, test_loader, lr=LEARNING_RATE,
    )
    save_predictions(_exp_id(exp_name), y_test, test_pred)

    print(f"[gli] Val  → RMSE={val_m['rmse']:.4f}  Pearson={val_m['pearson_r']:.4f}")
    print(f"[gli] Test → RMSE={test_m['rmse']:.4f}  Pearson={test_m['pearson_r']:.4f}")

    # Save best model checkpoint
    checkpoint_dir = Path("cache/models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{_exp_id(exp_name)}.pt"
    torch.save(best_state, checkpoint_path)
    print(f"[gli] Saved best model to {checkpoint_path}")
    
    log_result(
        experiment_id  = _exp_id(exp_name),
        model_name     = _exp_id(exp_name),
        model_family   = "gli",
        ligand_repr    = cfg["ligand_repr"],
        protein_repr   = cfg["protein_repr"],
        fusion_strategy= cfg.get("fusion", "graph"),
        n_params       = n_params,
        epochs_trained = epoch,
        batch_size     = BATCH_SIZE,
        learning_rate  = LEARNING_RATE,
        split_type     = _split_tag(),
        n_train        = len(y_train),
        n_val          = len(y_val),
        n_test         = len(y_test),
        val_metrics    = val_m,
        test_metrics   = test_m,
        train_time_sec = train_time,
        notes          = cfg.get("notes", ""),
    )


def run_seq(exp_name: str):
    """Run a sequence-model experiment (lstm / transformer_seq / mamba)."""
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL, BATCH_SIZE, LEARNING_RATE
    from harness.tokenizers import build_tokenizer
    from harness.seq_trainer import make_seq_loader, train_seq_model
    from models.sequence_models import LSTMBindingModel, TransformerSeqModel, MambaBindingModel

    cfg = EXPERIMENTS[exp_name]
    mf  = cfg["model_family"]

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Model      : {mf}  Lig tok: {cfg['lig_tok']}  Prot tok: {cfg['prot_tok']}")
    print(f"{'='*60}")

    train_df, val_df, test_df = _get_splits()

    # ── Build tokenizers (trained on train set only) ──────────────────────────
    print("[seq] Building ligand tokenizer …")
    lig_tok  = build_tokenizer(cfg["lig_tok"],  train_df[SMILES_COL].tolist())
    print("[seq] Building protein tokenizer …")
    prot_tok = build_tokenizer(cfg["prot_tok"], train_df[PROTEIN_COL].tolist())

    print(f"[seq] Lig vocab size : {lig_tok.vocab_size}")
    print(f"[seq] Prot vocab size: {prot_tok.vocab_size}")

    # ── Build data loaders ────────────────────────────────────────────────────
    train_loader = make_seq_loader(train_df, lig_tok, prot_tok, shuffle=True,  batch_size=BATCH_SIZE)
    val_loader   = make_seq_loader(val_df,   lig_tok, prot_tok, shuffle=False, batch_size=BATCH_SIZE * 4)
    test_loader  = make_seq_loader(test_df,  lig_tok, prot_tok, shuffle=False, batch_size=BATCH_SIZE * 4)

    # ── Build model ───────────────────────────────────────────────────────────
    if mf == "lstm":
        model = LSTMBindingModel(
            lig_vocab_size  = lig_tok.vocab_size,
            prot_vocab_size = prot_tok.vocab_size,
            lig_embed_dim   = 64,
            prot_embed_dim  = 64,
            hidden_dim      = 256,
            num_layers      = 2,
            dropout         = 0.2,
            lig_pad_id      = lig_tok.pad_id,
            prot_pad_id     = prot_tok.pad_id,
        )
    elif mf == "transformer_seq":
        model = TransformerSeqModel(
            lig_vocab_size  = lig_tok.vocab_size,
            prot_vocab_size = prot_tok.vocab_size,
            lig_embed_dim   = 128,
            prot_embed_dim  = 128,
            nhead           = 4,
            num_layers      = 3,
            dim_ff          = 256,
            dropout         = 0.1,
            lig_pad_id      = lig_tok.pad_id,
            prot_pad_id     = prot_tok.pad_id,
        )
    elif mf == "mamba":
        model = MambaBindingModel(
            lig_vocab_size  = lig_tok.vocab_size,
            prot_vocab_size = prot_tok.vocab_size,
            lig_embed_dim   = 128,
            prot_embed_dim  = 128,
            hidden_dim      = 256,
            n_layers        = 4,
            dropout         = 0.1,
            lig_pad_id      = lig_tok.pad_id,
            prot_pad_id     = prot_tok.pad_id,
        )
    else:
        raise ValueError(f"Unknown seq model family: {mf}")

    n_params = count_parameters(model)
    print(f"[seq] Model params: {n_params:,}")

    val_m, test_m, train_time, epochs_done, test_pred = train_seq_model(
        model, train_loader, val_loader, test_loader, lr=LEARNING_RATE,
    )
    save_predictions(_exp_id(exp_name), test_df[LABEL_COL].values, test_pred)

    print(f"[seq] Val  → {format_metrics(val_m)}")
    print(f"[seq] Test → {format_metrics(test_m)}")

    log_result(
        experiment_id   = _exp_id(exp_name),
        model_name      = _exp_id(exp_name),
        model_family    = mf,
        ligand_repr     = cfg["lig_tok"],
        protein_repr    = cfg["prot_tok"],
        fusion_strategy = cfg.get("fusion", "dual_encoder"),
        n_params        = n_params,
        epochs_trained  = epochs_done,
        batch_size      = BATCH_SIZE,
        learning_rate   = LEARNING_RATE,
        split_type      = _split_tag(),
        n_train         = len(train_df),
        n_val           = len(val_df),
        n_test          = len(test_df),
        val_metrics     = val_m,
        test_metrics    = test_m,
        train_time_sec  = train_time,
        notes           = cfg.get("notes", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BiCA v2 — true-sequence inputs runner
# ─────────────────────────────────────────────────────────────────────────────

def run_bica_v2(exp_name: str):
    """
    Run a BiCA v2 experiment with per-residue ESM-2 protein tokens and
    per-atom RDKit ligand features.  Uses variable-length sequences with
    padding/masks — proper cross-attention, not single-token degenerate.
    """
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL, BATCH_SIZE, LEARNING_RATE
    from harness.trainer import _get_device, count_parameters
    from harness.config import MAX_EPOCHS, PATIENCE
    from models.bica_v2 import build_bica_v2
    from harness.featurizers import (esm2_per_residue_padded, mol_atom_features_padded,
                                     chemberta_per_token_padded)
    import torch
    from torch.utils.data import Dataset, DataLoader
    from pathlib import Path

    cfg = EXPERIMENTS[exp_name]
    print(f"\n{'='*60}\n  Experiment : {exp_name}  [BiCA v2]\n{'='*60}")

    train_df, val_df, test_df = _get_splits()
    FEAT_CACHE = Path("cache/features")
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    pfx = _feat_cache_prefix()

    prot_model_size = cfg.get("esm2_size", "35M")
    max_atoms       = cfg.get("max_atoms",  100)
    max_prot_len    = cfg.get("max_prot_len", 512)

    # ── Featurize proteins ────────────────────────────────────────────────────
    # Cache key includes truncation length so different max_prot_len values
    # don't share the same cache file.
    print(f"[bica_v2] Computing per-residue ESM-2 ({prot_model_size}) features "
          f"(max_len={max_prot_len}) …")
    prot_tag = f"prot_esm2_{prot_model_size}_L{max_prot_len}"
    prot_cache_emb  = FEAT_CACHE / f"{pfx}{prot_tag}_seqemb_train.pt"
    prot_cache_mask = FEAT_CACHE / f"{pfx}{prot_tag}_seqmask_train.pt"
    def _load_prot(emb_path, mask_path, seqs, split_name):
        """Load per-residue ESM-2 embeddings from cache (float16) or compute."""
        if emb_path.exists() and mask_path.exists():
            emb  = torch.load(emb_path).float()   # fp16 on disk → fp32 in memory
            mask = torch.load(mask_path)
            print(f"[bica_v2]  Loaded {split_name} from cache: {emb.shape}")
        else:
            emb, mask = esm2_per_residue_padded(seqs, model_size=prot_model_size,
                                                max_len=max_prot_len)
            print(f"[bica_v2]  Computed {split_name}: {emb.shape}  "
                  f"({emb.numel()*4/1e9:.1f} GB fp32 → saving fp16)")
            torch.save(emb.half(), emb_path)   # save fp16 to halve disk usage
            torch.save(mask, mask_path)
        return emb, mask

    P_train, Pm_train = _load_prot(
        prot_cache_emb,   prot_cache_mask,
        train_df[PROTEIN_COL].tolist(), "train")

    prot_cache_emb_v  = FEAT_CACHE / f"{pfx}{prot_tag}_seqemb_val.pt"
    prot_cache_mask_v = FEAT_CACHE / f"{pfx}{prot_tag}_seqmask_val.pt"
    P_val, Pm_val = _load_prot(
        prot_cache_emb_v, prot_cache_mask_v,
        val_df[PROTEIN_COL].tolist(), "val")

    prot_cache_emb_t  = FEAT_CACHE / f"{pfx}{prot_tag}_seqemb_test.pt"
    prot_cache_mask_t = FEAT_CACHE / f"{pfx}{prot_tag}_seqmask_test.pt"
    P_test, Pm_test = _load_prot(
        prot_cache_emb_t, prot_cache_mask_t,
        test_df[PROTEIN_COL].tolist(), "test")

    prot_dim = P_train.shape[-1]   # e.g. 480 for ESM-2 35M

    # ── Featurize ligands ─────────────────────────────────────────────────────
    lig_repr       = cfg.get("ligand_repr", "mol_atom")   # "mol_atom" | "chemberta_tokens"
    cb_model_name  = cfg.get("chemberta_model", "seyonec/ChemBERTa-zinc-base-v1")

    if lig_repr == "chemberta_tokens":
        cb_tag = cb_model_name.split("/")[-1]
        print(f"[bica_v2] Computing per-token ChemBERTa ({cb_tag}) features …")
        lig_cache_emb  = FEAT_CACHE / f"{pfx}lig_cb_{cb_tag}_seqemb_train.pt"
        lig_cache_mask = FEAT_CACHE / f"{pfx}lig_cb_{cb_tag}_seqmask_train.pt"
        if lig_cache_emb.exists() and lig_cache_mask.exists():
            L_train, Lm_train = torch.load(lig_cache_emb), torch.load(lig_cache_mask)
            print(f"[bica_v2]  Loaded from cache: L_train {L_train.shape}")
        else:
            L_train, Lm_train = chemberta_per_token_padded(
                train_df[SMILES_COL].tolist(), model_name=cb_model_name)
            torch.save(L_train, lig_cache_emb);  torch.save(Lm_train, lig_cache_mask)
            print(f"[bica_v2]  Computed and cached: L_train {L_train.shape}")

        lig_cache_emb_v  = FEAT_CACHE / f"{pfx}lig_cb_{cb_tag}_seqemb_val.pt"
        lig_cache_mask_v = FEAT_CACHE / f"{pfx}lig_cb_{cb_tag}_seqmask_val.pt"
        if lig_cache_emb_v.exists() and lig_cache_mask_v.exists():
            L_val, Lm_val = torch.load(lig_cache_emb_v), torch.load(lig_cache_mask_v)
        else:
            L_val, Lm_val = chemberta_per_token_padded(
                val_df[SMILES_COL].tolist(), model_name=cb_model_name)
            torch.save(L_val, lig_cache_emb_v);  torch.save(Lm_val, lig_cache_mask_v)

        lig_cache_emb_t  = FEAT_CACHE / f"{pfx}lig_cb_{cb_tag}_seqemb_test.pt"
        lig_cache_mask_t = FEAT_CACHE / f"{pfx}lig_cb_{cb_tag}_seqmask_test.pt"
        if lig_cache_emb_t.exists() and lig_cache_mask_t.exists():
            L_test, Lm_test = torch.load(lig_cache_emb_t), torch.load(lig_cache_mask_t)
        else:
            L_test, Lm_test = chemberta_per_token_padded(
                test_df[SMILES_COL].tolist(), model_name=cb_model_name)
            torch.save(L_test, lig_cache_emb_t);  torch.save(Lm_test, lig_cache_mask_t)

    else:
        print(f"[bica_v2] Computing per-atom features (max_atoms={max_atoms}) …")
        lig_cache_emb  = FEAT_CACHE / f"{pfx}lig_atom{max_atoms}_seqemb_train.pt"
        lig_cache_mask = FEAT_CACHE / f"{pfx}lig_atom{max_atoms}_seqmask_train.pt"
        if lig_cache_emb.exists() and lig_cache_mask.exists():
            L_train, Lm_train = torch.load(lig_cache_emb), torch.load(lig_cache_mask)
            print(f"[bica_v2]  Loaded from cache: L_train {L_train.shape}")
        else:
            L_train, Lm_train = mol_atom_features_padded(
                train_df[SMILES_COL].tolist(), max_atoms=max_atoms)
            torch.save(L_train, lig_cache_emb);  torch.save(Lm_train, lig_cache_mask)

        lig_cache_emb_v  = FEAT_CACHE / f"{pfx}lig_atom{max_atoms}_seqemb_val.pt"
        lig_cache_mask_v = FEAT_CACHE / f"{pfx}lig_atom{max_atoms}_seqmask_val.pt"
        if lig_cache_emb_v.exists() and lig_cache_mask_v.exists():
            L_val, Lm_val = torch.load(lig_cache_emb_v), torch.load(lig_cache_mask_v)
            print(f"[bica_v2]  Loaded from cache: L_val {L_val.shape}")
        else:
            L_val, Lm_val = mol_atom_features_padded(
                val_df[SMILES_COL].tolist(), max_atoms=max_atoms)
            torch.save(L_val, lig_cache_emb_v);  torch.save(Lm_val, lig_cache_mask_v)

        lig_cache_emb_t  = FEAT_CACHE / f"{pfx}lig_atom{max_atoms}_seqemb_test.pt"
        lig_cache_mask_t = FEAT_CACHE / f"{pfx}lig_atom{max_atoms}_seqmask_test.pt"
        if lig_cache_emb_t.exists() and lig_cache_mask_t.exists():
            L_test, Lm_test = torch.load(lig_cache_emb_t), torch.load(lig_cache_mask_t)
            print(f"[bica_v2]  Loaded from cache: L_test {L_test.shape}")
        else:
            L_test, Lm_test = mol_atom_features_padded(
                test_df[SMILES_COL].tolist(), max_atoms=max_atoms)
            torch.save(L_test, lig_cache_emb_t);  torch.save(Lm_test, lig_cache_mask_t)

    lig_dim = L_train.shape[-1]   # 78 for mol_atom, 600 for chemberta_tokens

    y_train = torch.tensor(train_df[LABEL_COL].values, dtype=torch.float32)
    y_val   = torch.tensor(val_df[LABEL_COL].values,   dtype=torch.float32)
    y_test  = torch.tensor(test_df[LABEL_COL].values,  dtype=torch.float32)

    # ── Truncate proteins to max_prot_len (safety clip if cache was larger) ──
    if P_train.shape[1] > max_prot_len:
        P_train  = P_train[:, :max_prot_len, :]
        Pm_train = Pm_train[:, :max_prot_len]
        P_val    = P_val[:, :max_prot_len, :]
        Pm_val   = Pm_val[:, :max_prot_len]
        P_test   = P_test[:, :max_prot_len, :]
        Pm_test  = Pm_test[:, :max_prot_len]
        print(f"[bica_v2] Proteins truncated to {max_prot_len} residues")

    prot_dim = P_train.shape[-1]

    # ── DataLoader ────────────────────────────────────────────────────────────
    class SeqAffinityDataset(Dataset):
        def __init__(self, L, Lm, P, Pm, y):
            self.L, self.Lm, self.P, self.Pm, self.y = L, Lm, P, Pm, y
        def __len__(self):  return len(self.y)
        def __getitem__(self, i):
            return self.L[i], self.Lm[i], self.P[i], self.Pm[i], self.y[i]

    def collate_fn(batch):
        """Pad within batch to the maximum length present (more memory-efficient)."""
        L_list, Lm_list, P_list, Pm_list, y_list = zip(*batch)
        max_L = max(x.shape[0] for x in L_list)
        max_P = max(x.shape[0] for x in P_list)
        B, lig_d, prot_d = len(L_list), L_list[0].shape[-1], P_list[0].shape[-1]
        L_out  = torch.zeros(B, max_L, lig_d)
        Lm_out = torch.zeros(B, max_L, dtype=torch.long)
        P_out  = torch.zeros(B, max_P, prot_d)
        Pm_out = torch.zeros(B, max_P, dtype=torch.long)
        for i in range(B):
            ll, lp = L_list[i].shape[0], P_list[i].shape[0]
            L_out[i, :ll]  = L_list[i]
            Lm_out[i, :ll] = Lm_list[i]
            P_out[i, :lp]  = P_list[i]
            Pm_out[i, :lp] = Pm_list[i]
        return L_out, Lm_out, P_out, Pm_out, torch.stack(y_list)

    bs = min(BATCH_SIZE, 32)   # v2 uses more memory per sample; keep stable
    train_loader = DataLoader(
        SeqAffinityDataset(L_train, Lm_train, P_train, Pm_train, y_train),
        batch_size=bs, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(
        SeqAffinityDataset(L_val,   Lm_val,   P_val,   Pm_val,   y_val),
        batch_size=bs*2, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(
        SeqAffinityDataset(L_test,  Lm_test,  P_test,  Pm_test,  y_test),
        batch_size=bs*2, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ── Build model ───────────────────────────────────────────────────────────
    variant    = cfg.get("variant",    "bica_v2")
    hidden_dim = cfg.get("hidden_dim", 256)
    num_heads  = cfg.get("num_heads",  8)
    num_layers = cfg.get("num_layers", 2)
    dropout    = cfg.get("dropout",    0.2)
    drop_path  = cfg.get("drop_path",  0.1)   # stochastic depth regularisation

    model    = build_bica_v2(
        protein_dim=prot_dim, ligand_dim=lig_dim,
        variant=variant, hidden_dim=hidden_dim,
        num_heads=num_heads, num_layers=num_layers,
        dropout=dropout, drop_path=drop_path,
    )
    n_params = count_parameters(model)
    device   = _get_device()
    model    = model.to(device)
    print(f"[bica_v2] variant={variant}  params={n_params:,}  "
          f"prot_dim={prot_dim}(trunc={max_prot_len})  lig_dim={lig_dim}  hidden={hidden_dim}")

    # ── Optimiser: linear warmup + cosine decay ───────────────────────────────
    # Warmup is critical for cross-attention on long sequences — without it,
    # attention weights are random in early epochs and gradients blow up.
    criterion    = nn.MSELoss()
    peak_lr      = cfg.get("peak_lr",      1e-4)   # lower than default 5e-4
    weight_decay = cfg.get("weight_decay", 1e-3)   # strong L2 to counter overfitting
    warmup_steps = max(1, int(0.05 * MAX_EPOCHS * len(train_loader)))  # 5% of total steps
    total_steps  = MAX_EPOCHS * len(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    global_step = 0

    print(f"[bica_v2] peak_lr={peak_lr}  warmup_steps={warmup_steps}  total_steps={total_steps}")

    # Optional DSM auxiliary head
    dsm_head = None
    if cfg.get("use_dsm"):
        from models.dsm import DSMAuxHead, dsm_loss as _dsm_loss
        embed_dim = hidden_dim * 2
        dsm_head  = DSMAuxHead(embed_dim=embed_dim).to(device)
        print(f"[bica_v2] DSM aux loss enabled (embed_dim={embed_dim})")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_rmse, best_state, patience_ctr = float("inf"), None, 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for L_b, Lm_b, P_b, Pm_b, y_b in train_loader:
            L_b  = L_b.to(device);  Lm_b = Lm_b.to(device)
            P_b  = P_b.to(device);  Pm_b = Pm_b.to(device)
            y_b  = y_b.to(device)
            optimizer.zero_grad()
            if dsm_head is not None:
                z    = model.encode(P_b, L_b, Pm_b, Lm_b)
                pred = model.predict_from_embedding(z).squeeze(-1)
                loss = criterion(pred, y_b) + _dsm_loss(dsm_head, z)
            else:
                pred = model(P_b, L_b, Pm_b, Lm_b).squeeze(-1)
                loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_preds = []
        with torch.no_grad():
            for L_b, Lm_b, P_b, Pm_b, _ in val_loader:
                out = model(P_b.to(device), L_b.to(device),
                            Pm_b.to(device), Lm_b.to(device))
                val_preds.append(out.cpu().numpy().ravel())
        val_m = compute_metrics(y_val.numpy(), np.concatenate(val_preds))

        if val_m["rmse"] < best_rmse:
            best_rmse, patience_ctr = val_m["rmse"], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 5 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch:3d}  val_rmse={val_m['rmse']:.4f}  "
                  f"pearson={val_m['pearson_r']:.4f}  patience={patience_ctr}  lr={cur_lr:.2e}")
        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── Evaluate best checkpoint ──────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    val_preds, test_preds = [], []
    with torch.no_grad():
        for L_b, Lm_b, P_b, Pm_b, _ in val_loader:
            out = model(P_b.to(device), L_b.to(device),
                        Pm_b.to(device), Lm_b.to(device))
            val_preds.append(out.cpu().numpy().ravel())
        for L_b, Lm_b, P_b, Pm_b, _ in test_loader:
            out = model(P_b.to(device), L_b.to(device),
                        Pm_b.to(device), Lm_b.to(device))
            test_preds.append(out.cpu().numpy().ravel())

    test_pred_arr = np.concatenate(test_preds)
    val_m   = compute_metrics(y_val.numpy(),  np.concatenate(val_preds))
    test_m  = compute_metrics(y_test.numpy(), test_pred_arr)
    train_time = time.time() - t0
    save_predictions(_exp_id(exp_name), y_test.numpy(), test_pred_arr)

    print(f"[bica_v2] Val  → RMSE={val_m['rmse']:.4f}  Pearson={val_m['pearson_r']:.4f}")
    print(f"[bica_v2] Test → RMSE={test_m['rmse']:.4f}  Pearson={test_m['pearson_r']:.4f}")

    checkpoint_dir = Path("cache/models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_dir / f"{_exp_id(exp_name)}.pt")
    print(f"[bica_v2] Saved checkpoint: cache/models/{_exp_id(exp_name)}.pt")

    log_result(
        experiment_id   = _exp_id(exp_name),
        model_name      = _exp_id(exp_name),
        model_family    = "bica_v2",
        ligand_repr     = (f"chemberta_tokens_{cb_model_name.split('/')[-1]}"
                           if lig_repr == "chemberta_tokens" else f"mol_atom{max_atoms}"),
        protein_repr    = f"esm2_{prot_model_size}_perresidue",
        fusion_strategy = f"bica_v2_{variant}",
        n_params        = n_params,
        epochs_trained  = epoch,
        batch_size      = bs,
        learning_rate   = LEARNING_RATE,
        split_type      = _split_tag(),
        n_train         = len(y_train),
        n_val           = len(y_val),
        n_test          = len(y_test),
        val_metrics     = val_m,
        test_metrics    = test_m,
        train_time_sec  = train_time,
        notes           = cfg.get("notes", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run a single experiment
# ─────────────────────────────────────────────────────────────────────────────

def run(exp_name: str):
    from harness.config import SPLIT_SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC

    if exp_name not in EXPERIMENTS:
        print(f"[ERROR] Unknown experiment: {exp_name}")
        print("Available:", list(EXPERIMENTS.keys()))
        return

    cfg = EXPERIMENTS[exp_name]

    # Route to specialised runners based on model family
    if cfg["model_family"] in ("lstm", "transformer_seq", "mamba"):
        run_seq(exp_name)
        return
    if cfg["model_family"] == "distmat_cnn":
        run_distmat(exp_name)
        return
    if cfg["model_family"] in ("gcn", "gat"):
        run_gnn(exp_name)
        return
    if cfg["model_family"] == "bica":
        run_bica(exp_name)
        return
    if cfg["model_family"] == "graphormer":
        run_graphormer(exp_name)
        return
    if cfg["model_family"] == "gli":
        run_gli(exp_name)
        return
    if cfg["model_family"] == "bica_v2":
        run_bica_v2(exp_name)
        return
    if cfg["model_family"] == "finetune_mlp":
        run_finetune_mlp(exp_name)
        return
    if cfg["model_family"] == "finetune_bica_v2":
        run_finetune_bica_v2(exp_name)
        return

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  Model: {cfg['model_family']}  Ligand: {cfg['ligand_repr']}  Protein: {cfg['protein_repr']}")
    print(f"{'='*60}")

    # Load data
    train_df, val_df, test_df = _get_splits()

    # Build features
    (X_train, X_val, X_test,
     y_train, y_val,  y_test,
     L_train, L_val,  L_test,
     P_train, P_val,  P_test) = build_features(cfg, train_df, val_df, test_df)

    print(f"[run] Feature dim: {X_train.shape[1]}  train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}")

    mf = cfg["model_family"]
    n_params    = "N/A"
    epochs_done = "N/A"

    if mf not in ("linear", "tree", "gp"):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
        # Scale separate ligand/protein tensors if they exist (for cross‑attention)
        if P_train.shape[1] > 0:
            scaler_lig = StandardScaler().fit(L_train)
            L_train = scaler_lig.transform(L_train)
            L_val   = scaler_lig.transform(L_val)
            L_test  = scaler_lig.transform(L_test)
            scaler_prot = StandardScaler().fit(P_train)
            P_train = scaler_prot.transform(P_train)
            P_val   = scaler_prot.transform(P_val)
            P_test  = scaler_prot.transform(P_test)




    if mf in ("linear", "tree", "gp"):
        # Sklearn path
        model = build_sklearn_model(cfg, exp_name)
        val_m, test_m, train_time, _, test_pred = train_sklearn(
            model, X_train, y_train, X_val, y_val, X_test, y_test
        )
        save_predictions(_exp_id(exp_name), y_test, test_pred)

    elif mf == "transformer" and cfg.get("transformer_arch") == "cross_attn":
        # Cross-attention: needs separate lig/prot tensors
        import torch
        from harness.trainer import _get_device
        from models.transformer import CrossAttentionFusion

        lig_dim  = L_train.shape[1]
        prot_dim = P_train.shape[1]
        model    = CrossAttentionFusion(lig_dim, prot_dim)
        n_params = count_parameters(model)
        device   = _get_device()
        model    = model.to(device)

        # Wrap forward to accept concatenated X but route separately
        # We'll train with a custom loop here
        import torch.nn as nn
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        L_tr = torch.tensor(L_train, dtype=torch.float32)
        P_tr = torch.tensor(P_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        L_vl = torch.tensor(L_val,   dtype=torch.float32).to(device)
        P_vl = torch.tensor(P_val,   dtype=torch.float32).to(device)
        y_vl = y_val

        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(L_tr, P_tr, y_tr)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        from harness.config import MAX_EPOCHS, PATIENCE
        best_rmse, best_state, patience_ctr = float("inf"), None, 0
        t0 = time.time()

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for lb, pb, yb in loader:
                lb, pb, yb = lb.to(device), pb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model.forward_separate(lb, pb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model.forward_separate(L_vl, P_vl).cpu().numpy().ravel()
            rm = compute_metrics(y_vl, pred)
            if rm["rmse"] < best_rmse:
                best_rmse, patience_ctr = rm["rmse"], 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
            if epoch == 1 or epoch % 5 == 0:
                print(f"  epoch {epoch}  val_rmse={rm['rmse']:.4f}")
            if patience_ctr >= PATIENCE:
                break

        model.load_state_dict(best_state)
        model.eval()
        L_te = torch.tensor(L_test, dtype=torch.float32).to(device)
        P_te = torch.tensor(P_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            val_pred  = model.forward_separate(L_vl, P_vl).cpu().numpy().ravel()
            test_pred = model.forward_separate(L_te, P_te).cpu().numpy().ravel()
        val_m   = compute_metrics(y_val,  val_pred)
        test_m  = compute_metrics(y_test, test_pred)
        train_time  = time.time() - t0
        epochs_done = epoch
        save_predictions(_exp_id(exp_name), y_test, test_pred)

    else:
        # Generic PyTorch path (mlp, cnn, transformer self-attn)
        model    = build_torch_model(cfg, X_train.shape[1], L_train.shape[1], P_train.shape[1])
        n_params = count_parameters(model)
        if mf == "cnn":
            tr_in, vl_in, te_in = L_train, L_val, L_test
        else:
            tr_in, vl_in, te_in = X_train, X_val, X_test

        aux_loss_fn = None
        if cfg.get("use_ranking_loss"):
            from harness.losses import PairwiseRankingLoss
            aux_loss_fn = PairwiseRankingLoss(margin=0.5, n_pairs=32, lambda_rank=0.1)

        dsm_aux = None
        if cfg.get("use_dsm"):
            from models.dsm import DSMAuxHead, dsm_loss as _dsm_loss_fn
            from functools import partial
            from harness.trainer import _get_device

            # Infer embed dim from MLP arch (last hidden layer)
            _arch_to_dims = {"shallow": [256], "medium": [512, 256, 128],
                             "deep": [1024, 512, 256, 128, 64], "wide": [2048, 1024]}
            _embed_dim = _arch_to_dims[cfg.get("mlp_arch", "medium")][-1]
            _dsm_head  = DSMAuxHead(embed_dim=_embed_dim).to(_get_device())
            dsm_aux    = partial(_dsm_loss_fn, _dsm_head)
            print(f"[run] DualBind DSM auxiliary loss enabled (embed_dim={_embed_dim})")

        val_m, test_m, train_time, epochs_done, test_pred = train_torch(
            model,
            tr_in, y_train,
            vl_in, y_val,
            te_in, y_test,
            aux_loss=aux_loss_fn,
            dsm_aux=dsm_aux,
        )
        save_predictions(_exp_id(exp_name), y_test, test_pred)

    print(f"[run] Val  → {format_metrics(val_m)}")
    print(f"[run] Test → {format_metrics(test_m)}")

    log_result(
        experiment_id  = _exp_id(exp_name),
        model_name     = _exp_id(exp_name),
        model_family   = cfg["model_family"],
        ligand_repr    = cfg["ligand_repr"],
        protein_repr   = cfg["protein_repr"],
        fusion_strategy= cfg.get("fusion", "concat"),
        n_params       = n_params,
        epochs_trained = epochs_done,
        batch_size     = BATCH_SIZE if mf not in ("linear", "tree") else "N/A",
        learning_rate  = LEARNING_RATE if mf not in ("linear", "tree") else "N/A",
        split_type     = _split_tag(),
        n_train        = len(y_train),
        n_val          = len(y_val),
        n_test         = len(y_test),
        val_metrics    = val_m,
        test_metrics   = test_m,
        train_time_sec = train_time,
        notes          = cfg.get("notes", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full grid registration (all new embedding combinations)
# ─────────────────────────────────────────────────────────────────────────────

def register_full_grid():
    """Register all 246 experiments from the full grid."""
    # Vector ligand representations (new large ones + best old)
    lig_vectors = [
        "chemberta_600", "chemberta_5M", "chemberta_77M", "chemberta_100M",
         "esmc_300M"
    ]
    # Protein vector representations
    prot_vectors = [
        "esm2_8M_320", "esm2_35M_480", "esm2_150M", "esm2_650M",
        "esmc_300M", "prot_electra_256"
    ]
    # Model families that use vector concat
    vector_families = ["linear", "tree", "mlp", "transformer", "bica"]
    prefix = {
        "linear": "ridge",
        "tree": "xgb",
        "mlp": "mlp",
        "transformer": "transformer",
        "bica": "bica",
    }
    # For each family, register all ligand × protein combos
    for fam in vector_families:
        for lr in lig_vectors:
            for pr in prot_vectors:
                name = f"{prefix[fam]}_{lr}_{pr}"
                if name in EXPERIMENTS:
                    continue
                register(
                    name,
                    group="full_grid",
                    model_family=fam,
                    ligand_repr=lr,
                    protein_repr=pr,
                    fusion="concat" if fam != "bica" else "bidirectional_cross_attn",
                    notes="Full grid - new embeddings"
                )
    # CNN (fixed ligand = smiles_onehot)
    for pr in prot_vectors:
        name = f"cnn_smiles_onehot_{pr}"
        if name not in EXPERIMENTS:
            register(name, group="full_grid", model_family="cnn",
                     ligand_repr="smiles_onehot", protein_repr=pr, fusion="concat")
    # DistmatCNN (fixed ligand = distmat_100)
    for pr in prot_vectors:
        name = f"distmat_cnn_distmat_100_{pr}"
        if name not in EXPERIMENTS:
            register(name, group="full_grid", model_family="distmat_cnn",
                     ligand_repr="distmat_100", protein_repr=pr, fusion="distmat")
    # GNNs (GCN, GAT) fixed ligand = mol_graph
    for fam in ["gcn", "gat"]:
        for pr in prot_vectors:
            name = f"{fam}_mol_graph_{pr}"
            if name not in EXPERIMENTS:
                register(name, group="full_grid", model_family=fam,
                         ligand_repr="mol_graph", protein_repr=pr, fusion="graph")
    # Sequence models: use top tokenizer combos from earlier
    lig_tokens = ["smiles_atom", "smiles_bpe_1000"]
    prot_tokens = ["protein_char", "protein_bpe_1000"]
    seq_families = ["lstm", "transformer_seq", "mamba"]
    for fam in seq_families:
        for lt in lig_tokens:
            for pt in prot_tokens:
                name = f"{fam}_{lt}_{pt}"
                if name not in EXPERIMENTS:
                    register(name, group="full_grid", model_family=fam,
                             lig_tok=lt, prot_tok=pt, fusion="dual_encoder")

# Register the full grid (this will add all experiments)
register_full_grid()

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning experiments (reviewer response)
# ─────────────────────────────────────────────────────────────────────────────

register("mlp_ecfp4_esm2_8M_ft_k3",
    group="finetune",
    model_family="finetune_mlp",
    ligand_repr="ecfp4_1024", protein_repr="esm2_8M_online",
    fusion="concat", esm2_size="8M", finetune_layers=3,
    notes="Reviewer: MLP + fine-tuned ESM-2 8M (top 3 layers unfrozen)",
)
register("mlp_ecfp4_esm2_8M_ft_k6",
    group="finetune",
    model_family="finetune_mlp",
    ligand_repr="ecfp4_1024", protein_repr="esm2_8M_online",
    fusion="concat", esm2_size="8M", finetune_layers=6,
    notes="Reviewer: MLP + fine-tuned ESM-2 8M (all 6 layers unfrozen)",
)
register("mlp_ecfp4_esm2_35M_ft_k3",
    group="finetune",
    model_family="finetune_mlp",
    ligand_repr="ecfp4_1024", protein_repr="esm2_35M_online",
    fusion="concat", esm2_size="35M", finetune_layers=3,
    notes="Reviewer: MLP + fine-tuned ESM-2 35M (top 3 layers unfrozen)",
)
register("bica_v2_cb77M_esm2_35M_ft_k3",
    group="finetune",
    model_family="finetune_bica_v2",
    ligand_repr="chemberta_online_77M", protein_repr="esm2_35M_online",
    fusion="bidirectional_cross_attn", esm2_size="35M", finetune_layers=3,
    notes="Reviewer: BiCA v2 + fine-tuned ChemBERTa-77M + ESM-2 35M (top 3 layers)",
)
register("bica_v2_cb77M_esm2_35M_ft_k6",
    group="finetune",
    model_family="finetune_bica_v2",
    ligand_repr="chemberta_online_77M", protein_repr="esm2_35M_online",
    fusion="bidirectional_cross_attn", esm2_size="35M", finetune_layers=6,
    notes="Reviewer: BiCA v2 + fine-tuned ChemBERTa-77M + ESM-2 35M (top 6 layers)",
)

# ── Fine-tuning runners ───────────────────────────────────────────────────

def run_finetune_mlp(exp_name: str):
    """MLP with online fine-tuned ESM-2 encoder.

    ponytail: loads ESM-2 in-graph, freezes bottom, unfreezes top k layers,
    forward-propagates protein sequences through the encoder every batch.
    ECFP4 ligand features stay pre-computed (fingerprint, not a neural encoder).
    """
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import EsmModel, AutoTokenizer
    from harness.trainer import _get_device, count_parameters
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL
    from harness.config import BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE
    from harness.metrics import compute_metrics, format_metrics
    from harness.diary import log_result, save_predictions
    from harness.featurizers import ecfp
    import time, numpy as np

    cfg = EXPERIMENTS[exp_name]
    esm_size       = cfg.get("esm2_size", "8M")
    finetune_k     = cfg.get("finetune_layers", 3)
    esm_map = {
        "8M":  ("facebook/esm2_t6_8M_UR50D",   320, 6),
        "35M": ("facebook/esm2_t12_35M_UR50D", 480, 12),
    }
    model_name, esm_dim, total_layers = esm_map[esm_size]

    print(f"\n{'='*60}\n  Experiment: {exp_name}  [MLP + ESM-2 FT k={finetune_k}]\n{'='*60}")

    train_df, val_df, test_df = _get_splits()
    device = _get_device()

    # ── Ligand: pre-computed ECFP4 ──────────────────────────────────────
    print("[ft_mlp] Computing ECFP4 ligand features …")
    L_train = ecfp(train_df[SMILES_COL].tolist(), radius=2, nbits=1024)
    L_val   = ecfp(val_df[SMILES_COL].tolist(),   radius=2, nbits=1024)
    L_test  = ecfp(test_df[SMILES_COL].tolist(),  radius=2, nbits=1024)

    # ── Load ESM-2, freeze all, unfreeze top k ──────────────────────────
    print(f"[ft_mlp] Loading ESM-2 {esm_size} ({total_layers} layers, {esm_dim}d) …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    esm = EsmModel.from_pretrained(model_name, local_files_only=True).to(device)

    # Freeze all parameters
    for p in esm.parameters():
        p.requires_grad = False

    # Unfreeze top k transformer layers + final layer norm + pooler
    if finetune_k > 0:
        layers = list(esm.encoder.layer)
        for layer in layers[-finetune_k:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in esm.encoder.emb_layer_norm_after.parameters():
            p.requires_grad = True
        n_trainable = sum(p.numel() for p in esm.parameters() if p.requires_grad)
        print(f"[ft_mlp] Unfrozen params: {n_trainable:,} / "
              f"{sum(p.numel() for p in esm.parameters()):,}")
    else:
        esm.eval()  # fully frozen

    # ── Tokenize protein sequences ──────────────────────────────────────
    def tokenize(seqs):
        return tokenizer(seqs, return_tensors="pt", padding=True,
                         truncation=True, max_length=512)

    tok_train = tokenize(train_df[PROTEIN_COL].tolist())
    tok_val   = tokenize(val_df[PROTEIN_COL].tolist())
    tok_test  = tokenize(test_df[PROTEIN_COL].tolist())

    # ── Build MLP head ──────────────────────────────────────────────────
    # Input: ECFP4 (1024) + ESM-2 pooled (esm_dim)
    hidden_dim = 256
    mlp_head = nn.Sequential(
        nn.Linear(1024 + esm_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim // 2, 1),
    ).to(device)

    n_params = count_parameters(mlp_head) + sum(p.numel() for p in esm.parameters() if p.requires_grad)

    # ── Optimizer: separate LR for encoder vs head ──────────────────────
    encoder_params = [p for p in esm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": LEARNING_RATE * 0.1},  # lower LR for pretrained
        {"params": mlp_head.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=1e-4)

    criterion = nn.MSELoss()
    y_tr = torch.tensor(train_df[LABEL_COL].values, dtype=torch.float32)
    y_vl = val_df[LABEL_COL].values.astype(np.float32)
    y_te = test_df[LABEL_COL].values.astype(np.float32)

    L_tr = torch.tensor(L_train, dtype=torch.float32)
    L_vl = torch.tensor(L_val, dtype=torch.float32).to(device)
    L_te = torch.tensor(L_test, dtype=torch.float32).to(device)

    # ── Create DataLoader ───────────────────────────────────────────────
    ds = TensorDataset(
        tok_train["input_ids"], tok_train["attention_mask"], L_tr, y_tr.unsqueeze(1))
    loader = DataLoader(ds, batch_size=min(32, BATCH_SIZE), shuffle=True)

    # ── Helper: batched ESM inference ──────────────────────────────────
    def esm_encode(tok_dict, batch_size=64):
        """Encode proteins in mini-batches to avoid OOM."""
        n = len(tok_dict["input_ids"])
        outputs = []
        esm.eval()
        with torch.no_grad():
            for i in range(0, n, batch_size):
                ids = tok_dict["input_ids"][i:i+batch_size].to(device)
                mask = tok_dict["attention_mask"][i:i+batch_size].to(device)
                out = esm(input_ids=ids, attention_mask=mask)
                outputs.append(out.pooler_output.cpu())
        return torch.cat(outputs, dim=0)

    # ── Training loop ───────────────────────────────────────────────────
    # ponytail: manual loop — simpler than modifying train_torch for online encoding
    best_rmse, best_state, patience_ctr = float("inf"), None, 0
    t0 = time.time()
    epochs_done = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        esm.train() if finetune_k > 0 else esm.eval()
        mlp_head.train()

        for p_ids, p_mask, l_batch, y_batch in loader:
            p_ids, p_mask = p_ids.to(device), p_mask.to(device)
            l_batch, y_batch = l_batch.to(device), y_batch.to(device)

            with torch.set_grad_enabled(finetune_k > 0):
                esm_out = esm(input_ids=p_ids, attention_mask=p_mask)
            prot_emb = esm_out.pooler_output  # (B, esm_dim)

            x = torch.cat([l_batch, prot_emb], dim=1)
            pred = mlp_head(x)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mlp_head.parameters(), 1.0)
            if finetune_k > 0:
                nn.utils.clip_grad_norm_(encoder_params, 1.0)
            optimizer.step()

        # ── Validation ──────────────────────────────────────────────
        esm.eval()
        mlp_head.eval()
        with torch.no_grad():
            val_emb = esm_encode(tok_val)
            val_pred = mlp_head(torch.cat([L_vl, val_emb.to(device)], dim=1))
            val_pred = val_pred.cpu().numpy().ravel()

        vm = compute_metrics(y_vl, val_pred)
        if vm["rmse"] < best_rmse:
            best_rmse = vm["rmse"]
            patience_ctr = 0
            best_state = {
                "esm": {k: v.cpu().clone() for k, v in esm.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in mlp_head.state_dict().items()},
            }
        else:
            patience_ctr += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  val_rmse={vm['rmse']:.4f}  "
                  f"val_pearson={vm['pearson_r']:.4f}")
        if patience_ctr >= PATIENCE:
            epochs_done = epoch
            break
        epochs_done = epoch

    # ── Test ────────────────────────────────────────────────────────────
    esm.load_state_dict(best_state["esm"])
    mlp_head.load_state_dict(best_state["head"])
    esm.eval()
    mlp_head.eval()

    with torch.no_grad():
        val_emb2 = esm_encode(tok_val)
        val_pred2 = mlp_head(torch.cat([L_vl, val_emb2.to(device)], dim=1))
        val_pred2 = val_pred2.cpu().numpy().ravel()

        test_emb = esm_encode(tok_test)
        test_pred = mlp_head(torch.cat([L_te, test_emb.to(device)], dim=1))
        test_pred = test_pred.cpu().numpy().ravel()

    val_m  = compute_metrics(y_vl, val_pred2)
    test_m = compute_metrics(y_te, test_pred)
    train_time = time.time() - t0

    print(f"[ft_mlp] Val  → {format_metrics(val_m)}")
    print(f"[ft_mlp] Test → {format_metrics(test_m)}")

    save_predictions(_exp_id(exp_name), y_te, test_pred)
    log_result(
        experiment_id=_exp_id(exp_name), model_name=_exp_id(exp_name),
        model_family="finetune_mlp", ligand_repr=cfg["ligand_repr"],
        protein_repr="esm2_online", fusion_strategy="concat",
        n_params=n_params, epochs_trained=epochs_done,
        batch_size=min(32, BATCH_SIZE), learning_rate=LEARNING_RATE,
        split_type=_split_tag(), n_train=len(y_tr), n_val=len(y_vl),
        n_test=len(y_te), val_metrics=val_m, test_metrics=test_m,
        train_time_sec=train_time,
        notes=f"ESM-2 {esm_size} top {finetune_k} layers unfrozen. {cfg.get('notes','')}",
    )


def run_finetune_bica_v2(exp_name: str):
    """BiCA v2 with fine-tuned ESM-2 + ChemBERTa encoders.

    ponytail: loads both encoders in-graph, unfreezes top k layers,
    forward-propagates sequences through both every batch.
    """
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel
    from harness.trainer import _get_device, count_parameters
    from harness.config import LABEL_COL, SMILES_COL, PROTEIN_COL
    from harness.config import BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE
    from harness.metrics import compute_metrics, format_metrics
    from harness.diary import log_result, save_predictions
    import time, numpy as np

    cfg = EXPERIMENTS[exp_name]
    esm_size       = cfg.get("esm2_size", "35M")
    finetune_k     = cfg.get("finetune_layers", 3)
    esm_map = {
        "35M": ("facebook/esm2_t12_35M_UR50D", 480, 12),
    }
    esm_model_name, esm_dim, esm_layers = esm_map[esm_size]
    cb_model_name = "DeepChem/ChemBERTa-77M-MTR"

    print(f"\n{'='*60}\n  Experiment: {exp_name}  [BiCA v2 FT k={finetune_k}]\n{'='*60}")

    train_df, val_df, test_df = _get_splits()
    device = _get_device()

    # ── Load ESM-2 ──────────────────────────────────────────────────────
    print(f"[ft_bica] Loading ESM-2 {esm_size} via esm library …")
    import esm as esm_lib
    esm_loaders = {
        "35M": esm_lib.pretrained.esm2_t12_35M_UR50D,
    }
    esm, esm_alphabet = esm_loaders[esm_size]()
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm = esm.to(device)
    for p in esm.parameters():
        p.requires_grad = False
    if finetune_k > 0:
        for layer in list(esm.layers)[-finetune_k:]:
            for p in layer.parameters():
                p.requires_grad = True
        # Unfreeze emb_layer_norm_after equivalent
        for p in esm.emb_layer_norm_after.parameters():
            p.requires_grad = True

    # ── Load ChemBERTa ──────────────────────────────────────────────────
    print(f"[ft_bica] Loading ChemBERTa {cb_model_name} …")
    cb_tok = AutoTokenizer.from_pretrained(cb_model_name, local_files_only=True)
    cb = AutoModel.from_pretrained(cb_model_name, local_files_only=True).to(device)
    for p in cb.parameters():
        p.requires_grad = False
    if finetune_k > 0:
        cb_layers = list(cb.encoder.layer) if hasattr(cb, 'encoder') else []
        for layer in cb_layers[-finetune_k:]:
            for p in layer.parameters():
                p.requires_grad = True

    # ── Build BiCA v2 model ─────────────────────────────────────────────
    # ponytail: reuse existing bica_v2 builder — lig_dim from ChemBERTa, prot_dim from ESM-2
    cb_dim = cb.config.hidden_size  # typically 600 for ChemBERTa-zinc-base
    from models.bica_v2 import build_bica_v2
    bica = build_bica_v2(protein_dim=esm_dim, ligand_dim=cb_dim,
                         hidden_dim=128, num_heads=4, dropout=0.1).to(device)
    n_params = (count_parameters(bica) +
                sum(p.numel() for p in esm.parameters() if p.requires_grad) +
                sum(p.numel() for p in cb.parameters() if p.requires_grad))

    # ── Optimizer ───────────────────────────────────────────────────────
    ft_params = ([p for p in esm.parameters() if p.requires_grad] +
                 [p for p in cb.parameters() if p.requires_grad])
    optimizer = torch.optim.AdamW([
        {"params": ft_params, "lr": LEARNING_RATE * 0.1},
        {"params": bica.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=1e-4)
    criterion = nn.MSELoss()

    # ── Tokenize ────────────────────────────────────────────────────────
    def tokenize_prot(seqs):
        data = [(f"p{i}", s[:512]) for i, s in enumerate(seqs)]
        _, _, tokens = esm_batch_converter(data)
        mask = (tokens != esm_alphabet.padding_idx).long()
        return {"input_ids": tokens, "attention_mask": mask}
    def tokenize_lig(smiles_list):
        return cb_tok(smiles_list, return_tensors="pt", padding=True,
                      truncation=True, max_length=256)

    tok_p_tr = tokenize_prot(train_df[PROTEIN_COL].tolist())
    tok_p_vl = tokenize_prot(val_df[PROTEIN_COL].tolist())
    tok_p_te = tokenize_prot(test_df[PROTEIN_COL].tolist())
    tok_l_tr = tokenize_lig(train_df[SMILES_COL].tolist())
    tok_l_vl = tokenize_lig(val_df[SMILES_COL].tolist())
    tok_l_te = tokenize_lig(test_df[SMILES_COL].tolist())

    y_tr = train_df[LABEL_COL].values.astype(np.float32)
    y_vl = val_df[LABEL_COL].values.astype(np.float32)
    y_te = test_df[LABEL_COL].values.astype(np.float32)

    # ── DataLoader ──────────────────────────────────────────────────────
    ds = TensorDataset(
        tok_p_tr["input_ids"], tok_p_tr["attention_mask"],
        tok_l_tr["input_ids"], tok_l_tr["attention_mask"],
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(ds, batch_size=min(16, BATCH_SIZE // 2), shuffle=True)

    best_rmse, best_state, patience_ctr = float("inf"), None, 0
    t0 = time.time()
    epochs_done = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        esm.train() if finetune_k > 0 else esm.eval()
        cb.train() if finetune_k > 0 else cb.eval()
        bica.train()

        for pid, pmask, lid, lmask, yb in loader:
            pid, pmask = pid.to(device), pmask.to(device)
            lid, lmask = lid.to(device), lmask.to(device)
            yb = yb.to(device)

            with torch.set_grad_enabled(finetune_k > 0):
                prot_out = esm(pid, repr_layers=[esm.num_layers])
                prot_out = prot_out["representations"][esm.num_layers]
                lig_out  = cb(input_ids=lid, attention_mask=lmask).last_hidden_state

            # Create masks: 1 = real, 0 = pad (matching bica_v2 convention)
            prot_mask = pmask.bool()
            lig_mask  = lmask.bool()

            pred = bica(prot_out, lig_out, prot_mask, lig_mask)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(bica.parameters(), 1.0)
            if ft_params:
                nn.utils.clip_grad_norm_(ft_params, 1.0)
            optimizer.step()

        # ── Validation ──────────────────────────────────────────────
        esm.eval(); cb.eval(); bica.eval()
        with torch.no_grad():
            # Batched protein encoding to avoid OOM
            MAX_PROT_PER_BATCH = 64
            all_p_out = []
            all_l_out = []
            tok_p_vl_ids = tok_p_vl["input_ids"].to(device)
            tok_p_vl_mask = tok_p_vl["attention_mask"].bool().to(device)
            for p_start in range(0, len(tok_p_vl_ids), MAX_PROT_PER_BATCH):
                p_end = min(p_start + MAX_PROT_PER_BATCH, len(tok_p_vl_ids))
                p_batch = esm(tok_p_vl_ids[p_start:p_end],
                              repr_layers=[esm.num_layers])
                all_p_out.append(p_batch["representations"][esm.num_layers].cpu())
            p_vl = torch.cat(all_p_out, dim=0).to(device)
            l_vl = cb(input_ids=tok_l_vl["input_ids"].to(device),
                      attention_mask=tok_l_vl["attention_mask"].to(device))
            val_pred = bica(p_vl, l_vl.last_hidden_state,
                           tok_p_vl_mask,
                           tok_l_vl["attention_mask"].bool().to(device))
            val_pred = val_pred.cpu().numpy().ravel()

        vm = compute_metrics(y_vl, val_pred)
        if vm["rmse"] < best_rmse:
            best_rmse = vm["rmse"]; patience_ctr = 0
            best_state = {
                "esm": {k: v.cpu().clone() for k, v in esm.state_dict().items()},
                "cb": {k: v.cpu().clone() for k, v in cb.state_dict().items()},
                "bica": {k: v.cpu().clone() for k, v in bica.state_dict().items()},
            }
        else:
            patience_ctr += 1

        if epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  val_rmse={vm['rmse']:.4f}  "
                  f"val_pearson={vm['pearson_r']:.4f}")
        if patience_ctr >= PATIENCE:
            epochs_done = epoch; break
        epochs_done = epoch

    # ── Test ────────────────────────────────────────────────────────────
    esm.load_state_dict(best_state["esm"])
    cb.load_state_dict(best_state["cb"])
    bica.load_state_dict(best_state["bica"])
    esm.eval(); cb.eval(); bica.eval()

    with torch.no_grad():
        # Validation (batched)
        all_p_vl2 = []
        tok_p_vl_ids2 = tok_p_vl["input_ids"].to(device)
        for p_start in range(0, len(tok_p_vl_ids2), MAX_PROT_PER_BATCH):
            p_end = min(p_start + MAX_PROT_PER_BATCH, len(tok_p_vl_ids2))
            pb = esm(tok_p_vl_ids2[p_start:p_end], repr_layers=[esm.num_layers])
            all_p_vl2.append(pb["representations"][esm.num_layers].cpu())
        p_vl2 = torch.cat(all_p_vl2, dim=0).to(device)
        l_vl2 = cb(input_ids=tok_l_vl["input_ids"].to(device),
                   attention_mask=tok_l_vl["attention_mask"].to(device))
        val_pred2 = bica(p_vl2, l_vl2.last_hidden_state,
                        tok_p_vl["attention_mask"].bool().to(device),
                        tok_l_vl["attention_mask"].bool().to(device))
        val_pred2 = val_pred2.cpu().numpy().ravel()

        # Test (batched)
        all_p_te = []
        tok_p_te_ids = tok_p_te["input_ids"].to(device)
        for p_start in range(0, len(tok_p_te_ids), MAX_PROT_PER_BATCH):
            p_end = min(p_start + MAX_PROT_PER_BATCH, len(tok_p_te_ids))
            pb = esm(tok_p_te_ids[p_start:p_end], repr_layers=[esm.num_layers])
            all_p_te.append(pb["representations"][esm.num_layers].cpu())
        p_te = torch.cat(all_p_te, dim=0).to(device)
        l_te = cb(input_ids=tok_l_te["input_ids"].to(device),
                  attention_mask=tok_l_te["attention_mask"].to(device))
        test_pred = bica(p_te, l_te.last_hidden_state,
                        tok_p_te["attention_mask"].bool().to(device),
                        tok_l_te["attention_mask"].bool().to(device))
        test_pred = test_pred.cpu().numpy().ravel()

    val_m  = compute_metrics(y_vl, val_pred2)
    test_m = compute_metrics(y_te, test_pred)
    train_time = time.time() - t0

    print(f"[ft_bica] Val  → {format_metrics(val_m)}")
    print(f"[ft_bica] Test → {format_metrics(test_m)}")

    save_predictions(_exp_id(exp_name), y_te, test_pred)
    log_result(
        experiment_id=_exp_id(exp_name), model_name=_exp_id(exp_name),
        model_family="finetune_bica_v2", ligand_repr="chemberta_online_77M",
        protein_repr="esm2_online", fusion_strategy="bidirectional_cross_attn",
        n_params=n_params, epochs_trained=epochs_done,
        batch_size=min(16, BATCH_SIZE // 2), learning_rate=LEARNING_RATE,
        split_type=_split_tag(), n_train=len(y_tr), n_val=len(y_vl),
        n_test=len(y_te), val_metrics=val_m, test_metrics=test_m,
        train_time_sec=train_time,
        notes=f"ESM-2 {esm_size} + ChemBERTa-77M, top {finetune_k} layers unfrozen. {cfg.get('notes','')}",
    )

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Binding affinity benchmark runner")
    parser.add_argument("--exp",     type=str, help="Run a single experiment by name")
    parser.add_argument("--group",   type=str, help="Run all experiments in a group")
    parser.add_argument("--list",    action="store_true", help="List all experiments")
    parser.add_argument("--leaderboard", action="store_true", help="Print leaderboard")
    parser.add_argument("--dataset", type=str, default="bindingdb",
                        choices=["bindingdb", "leakypdb"],
                        help="Dataset to use (default: bindingdb)")
    parser.add_argument("--seed",    type=int, default=SPLIT_SEED,
                        help=f"Scaffold split seed (default: {SPLIT_SEED}, bindingdb only)")
    args = parser.parse_args()

    # Set active dataset/seed globals before any run() call
    global _ACTIVE_DATASET, _ACTIVE_SEED
    _ACTIVE_DATASET = args.dataset
    _ACTIVE_SEED    = args.seed

    if args.list:
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name:45s}  group={cfg['group']}")
        return

    if args.leaderboard:
        print_leaderboard()
        return

    if args.exp:
        run(args.exp)
        return

    if args.group:
        targets = (EXPERIMENTS.keys() if args.group == "all"
                   else [n for n, c in EXPERIMENTS.items() if c["group"] == args.group])
        for name in list(targets):
            try:
                run(name)
            except Exception as e:
                print(f"[ERROR] {name} failed: {e}")
        print_leaderboard()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
