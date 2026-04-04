"""
Quantitative fidelity evaluation for all interpretability methods.

Fidelity measures how much the model's prediction changes when the
most-important features are masked out (Deletion) vs the least-important
ones (Insertion). High fidelity = masking important features hurts more.

Metrics:
  AUC-Deletion:   area under the prediction-drop curve as top-k% features
                  are replaced by baseline (lower is better for the explanation)
  AUC-Insertion:  area under the prediction-recovery curve as top-k% features
                  are added back from baseline (higher is better)
  Sufficiency:    prediction with only the important features kept vs. full
  Comprehensiveness: prediction drop when important features removed

Reference: Petsiuk et al., BMVC 2018 (RISE); Samek et al., IEEE TNNLS 2017.

Usage:
    from interpret.fidelity_eval import deletion_insertion_auc, fidelity_report

    scores = fidelity_report(
        model, X_test, feature_importances,
        model_type="mlp",
        steps=10, top_fractions=[0.1, 0.2, 0.5],
    )
    print(scores)
"""

from __future__ import annotations
import numpy as np
from typing import Callable


def _masked_predict_flat(
    model,
    X: np.ndarray,
    importance: np.ndarray,
    frac: float,
    mask_value: float = 0.0,
    mode: str = "deletion",
    device_str: str = "cpu",
) -> np.ndarray:
    """
    Mask top-frac fraction of features (by |importance|) and return predictions.

    Args:
        model:       Fitted model with a predict or forward method.
        X:           (N, D) feature array.
        importance:  (D,) feature importance scores.
        frac:        Fraction of features to mask (0 < frac ≤ 1).
        mask_value:  Value to use for masked features (default 0.0 = baseline).
        mode:        "deletion" → mask top features;
                     "insertion" → keep only top features, zero the rest.
        device_str:  Device for PyTorch models.

    Returns:
        preds: (N,) predictions on masked inputs.
    """
    import torch

    D = X.shape[1]
    k = max(1, int(D * frac))
    top_idx = np.argsort(np.abs(importance))[::-1][:k]

    X_masked = X.copy()
    if mode == "deletion":
        X_masked[:, top_idx] = mask_value
    else:  # insertion — keep only top features
        mask = np.zeros(D, dtype=bool)
        mask[top_idx] = True
        X_masked[:, ~mask] = mask_value

    # Predict
    if hasattr(model, "predict"):
        return model.predict(X_masked).ravel()
    else:
        # PyTorch model
        model.eval()
        device = torch.device(device_str)
        model  = model.to(device)
        with torch.no_grad():
            t = torch.tensor(X_masked, dtype=torch.float32).to(device)
            return model(t).cpu().numpy().ravel()


def deletion_insertion_auc(
    model,
    X: np.ndarray,
    importance: np.ndarray,
    steps: int = 10,
    mask_value: float = 0.0,
    device_str: str = "cpu",
) -> dict[str, float]:
    """
    Compute AUC-Deletion and AUC-Insertion curves.

    Args:
        model:       Fitted model.
        X:           (N, D) test feature array.
        importance:  (D,) per-feature importance (e.g., mean |SHAP|).
        steps:       Number of fraction steps (evenly spaced 1/steps to 1).
        mask_value:  Baseline replacement value.
        device_str:  Device string for PyTorch models.

    Returns:
        {
          "auc_deletion":  float  (lower = explanation captures more signal)
          "auc_insertion": float  (higher = explanation recovers prediction faster)
          "deletion_curve":  list of (frac, mean_pred) tuples
          "insertion_curve": list of (frac, mean_pred) tuples
        }
    """
    fractions = np.linspace(1 / steps, 1.0, steps)

    del_curve  = []
    ins_curve  = []

    # Baseline predictions (unmasked)
    if hasattr(model, "predict"):
        base_preds = model.predict(X).ravel()
    else:
        import torch
        model.eval()
        device = torch.device(device_str)
        model  = model.to(device)
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(device)
            base_preds = model(t).cpu().numpy().ravel()

    for frac in fractions:
        del_preds = _masked_predict_flat(
            model, X, importance, frac, mask_value, "deletion", device_str)
        ins_preds = _masked_predict_flat(
            model, X, importance, frac, mask_value, "insertion", device_str)

        del_curve.append((float(frac), float(np.mean(np.abs(del_preds - base_preds)))))
        ins_curve.append((float(frac), float(np.mean(np.abs(ins_preds - base_preds)))))

    auc_del = float(np.trapz([v for _, v in del_curve],
                             [f for f, _ in del_curve]))
    auc_ins = float(np.trapz([v for _, v in ins_curve],
                             [f for f, _ in ins_curve]))

    return {
        "auc_deletion":   auc_del,
        "auc_insertion":  auc_ins,
        "deletion_curve": del_curve,
        "insertion_curve": ins_curve,
    }


def sufficiency_comprehensiveness(
    model,
    X: np.ndarray,
    importance: np.ndarray,
    top_frac: float = 0.2,
    mask_value: float = 0.0,
    device_str: str = "cpu",
) -> dict[str, float]:
    """
    Compute Sufficiency and Comprehensiveness at a fixed fraction.

    Sufficiency:       mean |pred(top_features_only) - pred(full)|
    Comprehensiveness: mean |pred(full) - pred(top_features_removed)|

    Lower sufficiency = top features alone nearly recover full prediction.
    Higher comprehensiveness = removing top features hurts prediction more.

    Args:
        model:       Fitted model.
        X:           (N, D) test features.
        importance:  (D,) per-feature importance scores.
        top_frac:    Fraction of features considered "top" (default 20%).
        mask_value:  Baseline value for masked features.
        device_str:  PyTorch device string.

    Returns:
        {"sufficiency": float, "comprehensiveness": float}
    """
    import torch

    if hasattr(model, "predict"):
        full_preds = model.predict(X).ravel()
    else:
        model.eval()
        device = torch.device(device_str)
        model  = model.to(device)
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(device)
            full_preds = model(t).cpu().numpy().ravel()

    ins_preds = _masked_predict_flat(
        model, X, importance, top_frac, mask_value, "insertion", device_str)
    del_preds = _masked_predict_flat(
        model, X, importance, top_frac, mask_value, "deletion",  device_str)

    sufficiency      = float(np.mean(np.abs(ins_preds - full_preds)))
    comprehensiveness = float(np.mean(np.abs(full_preds - del_preds)))

    return {
        "sufficiency":      sufficiency,
        "comprehensiveness": comprehensiveness,
    }


def fidelity_report(
    model,
    X: np.ndarray,
    importance: np.ndarray,
    model_type: str = "mlp",      # informational only (for logging)
    experiment_id: str = "",
    steps: int = 10,
    top_frac: float = 0.2,
    mask_value: float = 0.0,
    device_str: str = "cpu",
    save_path: str | Path | None = None,
) -> dict:
    """
    Full fidelity evaluation: AUC-Deletion, AUC-Insertion, Sufficiency,
    Comprehensiveness.

    Args:
        model:         Fitted model.
        X:             (N, D) test features.
        importance:    (D,) feature importance vector.
        model_type:    String label for logging (e.g. "mlp", "xgb").
        experiment_id: Experiment name for the report.
        steps:         Steps for AUC curve.
        top_frac:      Fraction for sufficiency/comprehensiveness.
        mask_value:    Baseline replacement value.
        device_str:    PyTorch device.
        save_path:     If provided, save JSON report here.

    Returns:
        Dict of all fidelity metrics.
    """
    from pathlib import Path
    import json

    auc_scores = deletion_insertion_auc(
        model, X, importance, steps=steps,
        mask_value=mask_value, device_str=device_str)
    suf_comp = sufficiency_comprehensiveness(
        model, X, importance, top_frac=top_frac,
        mask_value=mask_value, device_str=device_str)

    report = {
        "experiment_id":    experiment_id,
        "model_type":       model_type,
        "n_samples":        len(X),
        "n_features":       X.shape[1],
        "top_frac":         top_frac,
        "auc_deletion":     auc_scores["auc_deletion"],
        "auc_insertion":    auc_scores["auc_insertion"],
        "sufficiency":      suf_comp["sufficiency"],
        "comprehensiveness": suf_comp["comprehensiveness"],
        "deletion_curve":   auc_scores["deletion_curve"],
        "insertion_curve":  auc_scores["insertion_curve"],
    }

    print(f"[fidelity] {experiment_id} ({model_type})")
    print(f"  AUC-Deletion:     {report['auc_deletion']:.4f}  (lower = better explanation)")
    print(f"  AUC-Insertion:    {report['auc_insertion']:.4f}  (higher = better explanation)")
    print(f"  Sufficiency:      {report['sufficiency']:.4f}  (lower = top features sufficient)")
    print(f"  Comprehensiveness:{report['comprehensiveness']:.4f}  (higher = top features critical)")

    if save_path:
        Path(save_path).write_text(json.dumps(report, indent=2))
        print(f"[fidelity] Saved report → {save_path}")

    return report
