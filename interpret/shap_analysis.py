"""
SHAP-based feature importance for XGBoost, Random Forest, and MLP models.

Reference: Lundberg & Lee, NeurIPS 2017 (SHAP).

Provides:
  shap_for_tree(model, X, feature_names)  — TreeExplainer for XGB/RF/LGBM
  shap_for_mlp(model, X_bg, X_test, ...)  — DeepExplainer / KernelExplainer for MLP
  plot_shap_summary(shap_values, feature_names, top_k, save_path)
  ecfp_shap_to_substructure(shap_vals, smiles_list, ...)  — maps ECFP bit → RDKit substructure

Usage:
    from interpret.shap_analysis import shap_for_tree, plot_shap_summary
    shap_vals = shap_for_tree(xgb_model, X_test, feature_names)
    plot_shap_summary(shap_vals, feature_names, top_k=30, save_path="shap_summary.png")
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def shap_for_tree(
    model,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """
    Compute SHAP values for a tree-based model (XGBoost, RF, LightGBM).

    Args:
        model:         Fitted sklearn-compatible tree estimator.
        X:             (N, D) feature array.
        feature_names: Optional list of D feature names.

    Returns:
        shap_values: (N, D) SHAP values array.
    """
    import shap
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)
    # For multi-output or binary classifiers shap_values may be a list
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    return np.array(shap_vals)


def shap_for_mlp(
    model,
    X_background: np.ndarray,
    X_test: np.ndarray,
    n_background: int = 100,
    method: str = "deep",
) -> np.ndarray:
    """
    Compute SHAP values for a PyTorch MLP.

    Args:
        model:         Fitted MLP (nn.Module), already on CPU or GPU.
        X_background:  Background dataset for integration (all training data OK,
                       but n_background samples are sampled for speed).
        X_test:        (N, D) test samples to explain.
        n_background:  Number of background samples to keep.
        method:        "deep"   — DeepExplainer (faster, gradient-based)
                       "kernel" — KernelExplainer (model-agnostic, slower)

    Returns:
        shap_values: (N, D) SHAP values array.
    """
    import shap
    import torch

    model.eval()
    device = next(model.parameters()).device

    idx = np.random.choice(len(X_background),
                           size=min(n_background, len(X_background)),
                           replace=False)
    bg = torch.tensor(X_background[idx], dtype=torch.float32).to(device)

    if method == "deep":
        explainer = shap.DeepExplainer(model, bg)
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        shap_vals = explainer.shap_values(X_t)
    else:
        # KernelExplainer: needs a numpy predict function
        def _predict(x):
            with torch.no_grad():
                t = torch.tensor(x, dtype=torch.float32).to(device)
                return model(t).cpu().numpy().ravel()

        bg_np     = X_background[idx]
        explainer = shap.KernelExplainer(_predict, bg_np)
        shap_vals = explainer.shap_values(X_test)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    return np.array(shap_vals)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 30,
    save_path: str | Path | None = None,
    title: str = "SHAP feature importance",
) -> None:
    """
    Beeswarm summary plot of SHAP values.

    Args:
        shap_values:   (N, D) SHAP values.
        X:             (N, D) feature matrix (for colour mapping).
        feature_names: Optional list of D feature names.
        top_k:         Number of top features to show.
        save_path:     If provided, save figure here (PNG).
        title:         Plot title.
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.25)))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        max_display=top_k,
        show=False,
        plot_type="dot",
    )
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[shap] Saved summary plot → {save_path}")
    plt.close(fig)


def ecfp_shap_to_substructure(
    shap_values: np.ndarray,
    smiles_list: list[str],
    radius: int = 3,
    nbits: int = 1024,
    top_k: int = 10,
) -> dict[int, dict]:
    """
    Map the top-k ECFP bits by mean |SHAP| back to RDKit substructure info.

    Uses GetBitInfo (RDKit) to retrieve which atom × radius produced each bit.

    Args:
        shap_values: (N, nbits) SHAP values on ECFP features.
        smiles_list: (N,) SMILES used to compute ECFPs.
        radius:      ECFP radius (must match the ECFP used for features).
        nbits:       Number of ECFP bits.
        top_k:       Number of most-important bits to decode.

    Returns:
        Dict mapping bit_index → {"mean_abs_shap": float,
                                   "smiles_count": int,
                                   "example_atom_env": str or None}
        "example_atom_env" is a SMARTS string from the first molecule that sets the bit.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D

    mean_abs = np.abs(shap_values).mean(axis=0)          # (nbits,)
    top_bits  = np.argsort(mean_abs)[::-1][:top_k]

    results = {}
    for bit in top_bits.tolist():
        info_entry = {
            "mean_abs_shap": float(mean_abs[bit]),
            "smiles_count":  0,
            "example_atom_env": None,
        }
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            bit_info = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=nbits, bitInfo=bit_info)
            if bit in bit_info:
                info_entry["smiles_count"] += 1
                if info_entry["example_atom_env"] is None:
                    # Extract atom environment as SMARTS
                    atom_idx, rad = bit_info[bit][0]
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
                    amap = {}
                    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                    try:
                        info_entry["example_atom_env"] = Chem.MolToSmarts(submol)
                    except Exception:
                        info_entry["example_atom_env"] = "?"
        results[bit] = info_entry
    return results
