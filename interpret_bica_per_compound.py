"""
BiCA v2 — Per-Compound Interpretability Analysis
=================================================
Takes the test protein with the most compounds, runs every compound through
the model and produces residue-level × token-level attribution maps using:

  1. Raw cross-attention S1 (p2l) and S3 (AttentionPool protein) — baselines
  2. Value-weighted attention:  A[i,j] × ‖V[j]‖₂
     Removes "sink" tokens whose attention weight is high but Value vector
     carries near-zero information.
  3. Integrated Gradients (captum) on the projected protein embedding
     → per-residue L2 norm of gradient × input

Outputs (diary/figures/per_compound/):
  top_compounds_heatmap.png      — 3×3 grid of p2l heatmaps, top 9 compounds
  protein_consensus.png          — mean protein attribution across all compounds
  value_weighted_comparison.png  — raw vs value-weighted attention for 4 compounds
  ig_vs_attention.png            — IG attribution vs S3 AttentionPool side-by-side
  token_importance.png           — mean ligand-token importance across all compounds
  compound_scatter.png           — per-compound pKd vs top-protein-position entropy
"""

import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm

from harness.config import PROTEIN_COL, SMILES_COL, LABEL_COL
from harness.data import get_splits_for_seed
from models.bica_v2 import build_bica_v2

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT    = Path("cache/models/bica_v2_chemberta77M_tokens.pt")
FEAT_CACHE    = Path("cache/features")
OUT_DIR       = Path("diary/figures/per_compound")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED          = 42
PROT_SIZE     = "35M"
PROT_DIM      = 480
LIG_DIM       = 384       # ChemBERTa-77M-MTR hidden dim
HIDDEN_DIM    = 256
NUM_HEADS     = 8
NUM_LAYERS    = 2
MAX_PROT_LEN  = 512
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Max compounds to process (all of them — but limit heatmap grids)
MAX_COMPOUNDS = 200
HEATMAP_GRID  = 9   # top N compounds shown in the 3×3 grid

print(f"[per_compound] Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data, pick top protein
# ─────────────────────────────────────────────────────────────────────────────
_, _, test_df = get_splits_for_seed(SEED)
test_df = test_df.reset_index(drop=True)

counts = test_df[PROTEIN_COL].value_counts()
top_protein_seq = counts.index[0]
n_compounds     = counts.iloc[0]
prot_indices    = test_df.index[test_df[PROTEIN_COL] == top_protein_seq].tolist()

# Limit for speed
prot_indices = prot_indices[:MAX_COMPOUNDS]
smiles_list  = test_df.loc[prot_indices, SMILES_COL].tolist()
pkd_list     = test_df.loc[prot_indices, LABEL_COL].tolist()

print(f"[per_compound] Protein length  : {len(top_protein_seq)} aa")
print(f"[per_compound] Compounds found : {n_compounds} (using {len(prot_indices)})")
print(f"[per_compound] pKd range       : {min(pkd_list):.2f} – {max(pkd_list):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load ChemBERTa tokenizer for token → SMILES substring decoding
# ─────────────────────────────────────────────────────────────────────────────
print(f"[per_compound] Loading ChemBERTa tokenizer: {CHEMBERTA_MODEL}")
from transformers import AutoTokenizer
cb_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)

def decode_tokens(smiles: str):
    """
    Tokenize SMILES and return list of decoded token strings, stripping
    [CLS] (index 0) and [SEP]/padding — matching the featurizer logic.
    """
    enc = cb_tokenizer(smiles, return_tensors="pt", truncation=True, max_length=130)
    ids = enc["input_ids"][0].tolist()          # includes [CLS], [SEP]
    # strip leading [CLS] and trailing [SEP]
    ids = [i for i in ids if i not in (cb_tokenizer.cls_token_id,
                                        cb_tokenizer.sep_token_id,
                                        cb_tokenizer.pad_token_id)]
    tokens = cb_tokenizer.convert_ids_to_tokens(ids)
    # ChemBERTa uses "Ġ" prefix for space; clean it up
    tokens = [t.replace("Ġ", " ").strip() for t in tokens]
    return tokens

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load cached features for these specific compounds
# ─────────────────────────────────────────────────────────────────────────────
prot_tag = f"prot_esm2_{PROT_SIZE}_L{MAX_PROT_LEN}"
P_test_all  = torch.load(FEAT_CACHE / f"{prot_tag}_seqemb_test.pt").float()
Pm_test_all = torch.load(FEAT_CACHE / f"{prot_tag}_seqmask_test.pt")
L_test_all  = torch.load(FEAT_CACHE / f"lig_cb_ChemBERTa-77M-MTR_seqemb_test.pt").float()
Lm_test_all = torch.load(FEAT_CACHE / f"lig_cb_ChemBERTa-77M-MTR_seqmask_test.pt")

# Truncate protein dim if needed
if P_test_all.shape[1] > MAX_PROT_LEN:
    P_test_all  = P_test_all[:, :MAX_PROT_LEN, :]
    Pm_test_all = Pm_test_all[:, :MAX_PROT_LEN]

idx_t     = torch.tensor(prot_indices)
P_sel     = P_test_all[idx_t]       # (N, MAX_PROT_LEN, 480)
Pm_sel    = Pm_test_all[idx_t]      # (N, MAX_PROT_LEN)
L_sel     = L_test_all[idx_t]       # (N, MAX_LIG_LEN, 384)
Lm_sel    = Lm_test_all[idx_t]      # (N, MAX_LIG_LEN)

# True protein length (shared across all compounds for this protein)
p_len = int(Pm_sel[0].sum().item())
MAX_LIG_LEN = L_sel.shape[1]
print(f"[per_compound] P shape {P_sel.shape}, L shape {L_sel.shape}")
print(f"[per_compound] Protein actual length: {p_len}")

N = len(prot_indices)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build and load model
# ─────────────────────────────────────────────────────────────────────────────
model = build_bica_v2(
    protein_dim=PROT_DIM, ligand_dim=LIG_DIM,
    hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
    dropout=0.3, drop_path=0.1,
)
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
state = ckpt["model_state"] if "model_state" in ckpt else ckpt
model.load_state_dict(state)
model = model.to(DEVICE).eval()
print(f"[per_compound] Loaded {sum(p.numel() for p in model.parameters()):,} param model")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Hooks for AttentionPool and Value vectors
# ─────────────────────────────────────────────────────────────────────────────
_pool_weights = {}   # "prot" / "lig"  → (B, L)
_value_norms  = {}   # "p2l_l{layer}" / "l2p_l{layer}"  → (B, L_key)

def _make_pool_hook(name):
    def hook(module, inputs, output):
        x    = inputs[0]
        mask = inputs[1] if len(inputs) > 1 else None
        with torch.no_grad():
            scores = module.score(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask.bool(), float("-inf"))
            w = F.softmax(scores, dim=-1)
        _pool_weights[name] = w.detach().cpu()
    return hook

def _make_value_hook(layer_module, mha_key, li):
    """
    Hook on CrossAttentionBlock to capture ‖V[j]‖₂ for the p2l direction.

    In CrossAttentionBlock.forward the p2l call is:
        self.p2l(query=norm_p1(protein_h), key=norm_l1(ligand_h),
                 value=norm_l1(ligand_h), ...)
    So the value source is norm_l1(ligand_h).

    We capture ligand_h (inputs[1]) and apply norm_l1 + W_v manually.
    """
    def hook(module, inputs, output):
        # CrossAttentionBlock.forward(protein_h, ligand_h, p_key_mask, l_key_mask, ...)
        # inputs tuple: (protein_h, ligand_h, p_key_mask, l_key_mask, return_weights)
        # But positional vs kwargs varies — use both defensively
        if len(inputs) >= 2:
            ligand_h = inputs[1]
        else:
            return   # can't compute — skip silently

        mha = layer_module.p2l          # nn.MultiheadAttention for p2l direction
        norm = layer_module.norm_l1     # LayerNorm applied to ligand before p2l

        H  = mha.embed_dim
        Wv = mha.in_proj_weight[2*H:, :]   # (H, embed_dim)
        bv = mha.in_proj_bias[2*H:] if mha.in_proj_bias is not None \
             else torch.zeros(H, device=ligand_h.device)

        with torch.no_grad():
            val_src = norm(ligand_h)          # (B, L_lig, H)
            V  = val_src @ Wv.T + bv          # (B, L_lig, H)
            nv = V.norm(dim=-1)               # (B, L_lig)
        _value_norms[mha_key] = nv.detach().cpu()
    return hook

# Register pool hooks
h_prot_pool = model.prot_pool.register_forward_hook(_make_pool_hook("prot"))
h_lig_pool  = model.lig_pool.register_forward_hook(_make_pool_hook("lig"))

# Register Value-norm hooks on CrossAttentionBlock (not MHA directly)
_mha_hooks = []
for li, layer in enumerate(model.layers):
    _mha_hooks.append(
        layer.register_forward_hook(_make_value_hook(layer, f"p2l_l{li}", li))
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6. Forward pass — one compound at a time (need individual IG)
# ─────────────────────────────────────────────────────────────────────────────

# Storage (use last cross-attention layer — most refined representations)
LAST_LAYER = NUM_LAYERS - 1

# Per compound arrays
preds           = []
s3_prot_all     = np.zeros((N, p_len), np.float32)       # AttentionPool protein
s4_lig_all      = []                                     # variable length
p2l_last_all    = np.zeros((N, NUM_HEADS, p_len, MAX_LIG_LEN), np.float32)  # S1 last layer
vw_p2l_all      = np.zeros_like(p2l_last_all)            # value-weighted S1
ig_prot_all     = np.zeros((N, p_len), np.float32)       # IG protein attribution
l_lens          = []                                     # real ligand token lengths

print("[per_compound] Running forward pass + hooks …")

with torch.no_grad():
    for ci in tqdm(range(N)):
        p_b  = P_sel[ci:ci+1].to(DEVICE)      # (1, MAX_PROT_LEN, 480)
        pm_b = Pm_sel[ci:ci+1].to(DEVICE)
        l_b  = L_sel[ci:ci+1].to(DEVICE)
        lm_b = Lm_sel[ci:ci+1].to(DEVICE)

        pred, attn = model(p_b, l_b,
                           protein_mask=pm_b, ligand_mask=lm_b,
                           return_attention=True)
        preds.append(float(pred.item()))

        # S3 AttentionPool protein (already in hook)
        s3 = _pool_weights["prot"][0, :p_len].numpy()       # (p_len,)
        s3_prot_all[ci] = s3

        # S4 ligand
        l_len = int(lm_b[0].sum().item())
        l_lens.append(l_len)
        s4 = _pool_weights["lig"][0, :l_len].numpy()
        s4_lig_all.append(s4)

        # S1 last layer — cross-attention p2l: (1, H, L_prot, L_lig)
        p2l_w = attn["protein_to_ligand"][LAST_LAYER][0].cpu().numpy()  # (H, Lp, Ll)
        p2l_last_all[ci, :, :p_len, :l_len] = p2l_w[:, :p_len, :l_len]

        # Value-weighted S1 — scale each ligand column j by ‖V_lig[j]‖₂
        v_lig_norms = _value_norms[f"p2l_l{LAST_LAYER}"][0, :l_len].numpy()  # (l_len,)
        # Normalise so weights still sum to ~1 (for comparability)
        vw = p2l_w[:, :p_len, :l_len] * v_lig_norms[None, None, :]   # (H, p_len, l_len)
        row_sums = vw.sum(axis=-1, keepdims=True).clip(min=1e-12)
        vw = vw / row_sums
        vw_p2l_all[ci, :, :p_len, :l_len] = vw

print("[per_compound] Hook-based pass complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Integrated Gradients — protein embedding attribution
#    Target: predicted pKd (scalar).
#    Input:  projected protein embedding ph = norm_p_in(protein_proj(P))
#            We propagate gradients back to this (post-projection) representation.
#    Baseline: zero tensor (padding representation).
#    Attribution magnitude: L2 norm across the hidden dimension per residue.
# ─────────────────────────────────────────────────────────────────────────────
print("[per_compound] Computing Integrated Gradients …")

try:
    from captum.attr import IntegratedGradients

    # We need a wrapper that takes the protein projected embedding as input
    # (bypassing protein_proj) and runs the rest of the model.
    class _ProtEmbWrapper(nn.Module):
        def __init__(self, base_model, l_b, lm_b, pm_b):
            super().__init__()
            self.base  = base_model
            self.l_b   = l_b    # (1, L_lig, lig_dim)
            self.lm_b  = lm_b  # (1, L_lig)
            self.pm_b  = pm_b  # (1, L_prot)

        def forward(self, ph):
            # ph: (B, L_p, H) — B can be > 1 when captum interpolates steps
            B = ph.shape[0]

            # Pre-compute ligand embedding once, then expand to match B
            with torch.no_grad():
                lh_base = self.base.norm_l_in(self.base.ligand_proj(self.l_b))  # (1,Ll,H)
            lh = lh_base.expand(B, -1, -1).contiguous()   # (B, L_lig, H)

            # Expand masks to match B
            pm_b_exp = self.pm_b.expand(B, -1).contiguous() if self.pm_b is not None else None
            lm_b_exp = self.lm_b.expand(B, -1).contiguous() if self.lm_b is not None else None

            p_kpm = (~pm_b_exp.bool()) if pm_b_exp is not None else None
            l_kpm = (~lm_b_exp.bool()) if lm_b_exp is not None else None

            for layer in self.base.layers:
                ph, lh, _, _ = layer(ph, lh, p_kpm, l_kpm, return_weights=False)

            pv = self.base._pool(ph, pm_b_exp, self.base.prot_pool)
            lv = self.base._pool(lh, lm_b_exp, self.base.lig_pool)

            pv = self.base.prot_proj2(pv)
            lv = self.base.lig_proj2(lv)

            combined = torch.cat([pv, lv], dim=-1)
            return self.base.predictor(combined).squeeze(-1)

    for ci in tqdm(range(N)):
        p_b  = P_sel[ci:ci+1].to(DEVICE)
        pm_b = Pm_sel[ci:ci+1].to(DEVICE)
        l_b  = L_sel[ci:ci+1].to(DEVICE)
        lm_b = Lm_sel[ci:ci+1].to(DEVICE)

        # Pre-compute the projected protein embedding
        with torch.no_grad():
            ph_actual = model.norm_p_in(model.protein_proj(p_b))  # (1, L_p, H)

        wrapper = _ProtEmbWrapper(model, l_b, lm_b, pm_b).to(DEVICE)
        wrapper.eval()

        ig = IntegratedGradients(wrapper)
        baseline = torch.zeros_like(ph_actual)

        ph_actual.requires_grad_(True)
        attrs, delta = ig.attribute(
            ph_actual,
            baselines=baseline,
            n_steps=50,
            return_convergence_delta=True,
        )
        # attrs: (1, L_p, H)  →  L2 norm per residue → (L_p,)
        ig_prot = attrs[0, :p_len, :].norm(dim=-1).detach().cpu().numpy()
        ig_prot_all[ci] = ig_prot / (ig_prot.sum() + 1e-12)   # normalise to sum=1

    print("[per_compound] IG complete.")
    HAS_IG = True

except Exception as e:
    print(f"[per_compound] IG failed ({e}) — skipping IG figures.")
    HAS_IG = False

# Remove hooks
h_prot_pool.remove()
h_lig_pool.remove()
for h in _mha_hooks:
    h.remove()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Consensus attribution = IG × S3 (protein)
# ─────────────────────────────────────────────────────────────────────────────
if HAS_IG:
    consensus_prot = ig_prot_all * s3_prot_all      # (N, p_len) element-wise
    # normalise per compound
    row_sums = consensus_prot.sum(axis=1, keepdims=True).clip(min=1e-12)
    consensus_prot = consensus_prot / row_sums

# ─────────────────────────────────────────────────────────────────────────────
# 9. Decode ChemBERTa tokens for each compound
# ─────────────────────────────────────────────────────────────────────────────
print("[per_compound] Decoding ChemBERTa tokens …")
token_labels = [decode_tokens(s) for s in smiles_list]

# Pad/trim token label lists to match the ligand length from cache
for ci in range(N):
    l_len_ci = l_lens[ci]
    tl = token_labels[ci]
    if len(tl) < l_len_ci:
        token_labels[ci] = tl + ["…"] * (l_len_ci - len(tl))
    else:
        token_labels[ci] = tl[:l_len_ci]

# ─────────────────────────────────────────────────────────────────────────────
# 10. Figure helpers
# ─────────────────────────────────────────────────────────────────────────────
RES_POSITIONS = np.arange(p_len)

def plot_attn_heatmap(ax, matrix, token_lbl, p_len,
                      title="", xlabel="Ligand token", ylabel="Residue"):
    """
    matrix: (p_len, l_len)  — e.g., mean over heads
    """
    l_len = matrix.shape[1]
    im = ax.imshow(matrix, aspect="auto", cmap="hot", origin="upper",
                   interpolation="nearest",
                   vmin=0, vmax=matrix.max())
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)

    # X-axis: token labels (skip if too many)
    if l_len <= 30:
        ax.set_xticks(np.arange(l_len))
        ax.set_xticklabels(token_lbl[:l_len], rotation=90, fontsize=5)
    else:
        step = max(1, l_len // 15)
        ax.set_xticks(np.arange(0, l_len, step))
        ax.set_xticklabels(token_lbl[::step], rotation=90, fontsize=5)

    ax.tick_params(axis="y", labelsize=5)
    return im

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: 3×3 grid of p2l attention heatmaps for 9 compounds spanning pKd range
# ─────────────────────────────────────────────────────────────────────────────
print("[per_compound] Plotting heatmap grid …")

# Pick 9 compounds spanning the pKd range (low, mid, high)
pkd_arr = np.array(pkd_list)
grid_idx = np.round(np.linspace(0, N - 1, HEATMAP_GRID)).astype(int)
sorted_by_pkd = np.argsort(pkd_arr)
grid_idx = sorted_by_pkd[grid_idx]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for gi, ci in enumerate(grid_idx):
    l_len = l_lens[ci]
    # Mean over heads for last layer
    p2l_mean = p2l_last_all[ci, :, :p_len, :l_len].mean(axis=0)   # (p_len, l_len)
    ax = axes[gi]
    im = plot_attn_heatmap(ax, p2l_mean, token_labels[ci], p_len,
                           title=f"pKd={pkd_arr[ci]:.2f}  pred={preds[ci]:.2f}\n"
                                 f"SMILES[:{min(30,len(smiles_list[ci]))}]: "
                                 f"{smiles_list[ci][:30]}…")

fig.suptitle(f"BiCA v2 Cross-Attention (p2l, last layer, head-avg)\n"
             f"Protein: {len(top_protein_seq)} aa  |  9 compounds spanning pKd range",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "top_compounds_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUT_DIR}/top_compounds_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Mean protein attribution profile across all compounds
# ─────────────────────────────────────────────────────────────────────────────
print("[per_compound] Plotting protein consensus profile …")

mean_s3   = s3_prot_all.mean(axis=0)                                 # (p_len,)
mean_p2l  = p2l_last_all[:, :, :p_len, :].mean(axis=(0, 1, 3))     # (p_len,)
mean_vwp2l= vw_p2l_all[:, :, :p_len, :].mean(axis=(0, 1, 3))       # (p_len,)
mean_ig   = ig_prot_all.mean(axis=0) if HAS_IG else None            # (p_len,)

# Find top-10 positions for annotation
top10_s3 = np.argsort(mean_s3)[-10:]

fig, axes = plt.subplots(3 if HAS_IG else 2, 1, figsize=(14, 9), sharex=True)

axes[0].bar(RES_POSITIONS, mean_s3, width=1.0, color="steelblue", alpha=0.8)
axes[0].set_ylabel("S3 AttentionPool\nweight", fontsize=9)
axes[0].set_title(f"Mean protein attribution profile ({N} compounds)", fontsize=11)
for pos in top10_s3:
    axes[0].axvline(pos, color="crimson", alpha=0.5, linewidth=0.8)
    axes[0].text(pos, axes[0].get_ylim()[1] * 0.85, str(pos),
                 fontsize=5, ha="center", color="crimson")

axes[1].bar(RES_POSITIONS, mean_vwp2l, width=1.0, color="darkorange", alpha=0.8,
            label="Value-weighted")
axes[1].bar(RES_POSITIONS, mean_p2l,   width=1.0, color="royalblue",  alpha=0.4,
            label="Raw attention")
axes[1].set_ylabel("p2l attention\n(last layer, head-avg)", fontsize=9)
axes[1].legend(fontsize=8)

if HAS_IG:
    axes[2].bar(RES_POSITIONS, mean_ig, width=1.0, color="mediumseagreen", alpha=0.8)
    axes[2].set_ylabel("Integrated\nGradients", fontsize=9)
    axes[2].set_xlabel("Protein residue position", fontsize=10)
else:
    axes[1].set_xlabel("Protein residue position", fontsize=10)

for ax in axes:
    ax.set_xlim(-1, p_len)
    ax.tick_params(labelsize=8)

uniform_s3 = 1.0 / p_len
axes[0].axhline(uniform_s3, color="gray", linestyle="--", linewidth=0.8, label="Uniform")
axes[0].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "protein_consensus.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUT_DIR}/protein_consensus.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Raw vs Value-weighted attention for 4 compounds side-by-side
# ─────────────────────────────────────────────────────────────────────────────
print("[per_compound] Plotting value-weighted comparison …")

cmp_idx = sorted_by_pkd[[0, N//3, 2*N//3, N-1]]   # low / mid-low / mid-high / high pKd

fig, axes = plt.subplots(2, 4, figsize=(18, 7))

for col, ci in enumerate(cmp_idx):
    l_len = l_lens[ci]
    raw_p2l = p2l_last_all[ci, :, :p_len, :l_len].mean(axis=0)   # (p_len, l_len)
    vw_p2l  = vw_p2l_all[ci, :, :p_len, :l_len].mean(axis=0)

    ax_raw = axes[0, col]
    ax_vw  = axes[1, col]

    im1 = plot_attn_heatmap(ax_raw, raw_p2l, token_labels[ci], p_len,
                            title=f"Raw  pKd={pkd_arr[ci]:.2f}")
    im2 = plot_attn_heatmap(ax_vw,  vw_p2l,  token_labels[ci], p_len,
                            title=f"Value-weighted  pKd={pkd_arr[ci]:.2f}")

axes[0, 0].set_ylabel("Residue (Raw p2l)", fontsize=9)
axes[1, 0].set_ylabel("Residue (Value-weighted)", fontsize=9)

fig.suptitle("Raw vs Value-Weighted Cross-Attention (p2l, last layer, head-avg)\n"
             "Value weighting suppresses 'sink' tokens with near-zero information",
             fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "value_weighted_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUT_DIR}/value_weighted_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: IG vs AttentionPool comparison
# ─────────────────────────────────────────────────────────────────────────────
if HAS_IG:
    print("[per_compound] Plotting IG vs AttentionPool …")

    fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharex=True)

    for col, ci in enumerate(cmp_idx):
        ax_ig = axes[0, col]
        ax_s3 = axes[1, col]

        ax_ig.bar(RES_POSITIONS, ig_prot_all[ci], width=1.0,
                  color="mediumseagreen", alpha=0.8)
        ax_ig.set_title(f"pKd={pkd_arr[ci]:.2f}", fontsize=8)
        ax_ig.tick_params(labelsize=6)
        ax_ig.set_xlim(-1, p_len)

        ax_s3.bar(RES_POSITIONS, s3_prot_all[ci], width=1.0,
                  color="steelblue", alpha=0.8)
        ax_s3.axhline(1.0/p_len, color="gray", linestyle="--",
                      linewidth=0.7, label="Uniform")
        ax_s3.tick_params(labelsize=6)
        ax_s3.set_xlim(-1, p_len)
        ax_s3.set_xlabel("Residue position", fontsize=7)

    axes[0, 0].set_ylabel("Integrated Gradients\n(normalised)", fontsize=8)
    axes[1, 0].set_ylabel("S3 AttentionPool\nweight", fontsize=8)

    fig.suptitle("Integrated Gradients vs AttentionPool — 4 compounds spanning pKd range",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ig_vs_attention.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUT_DIR}/ig_vs_attention.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Mean ligand-token importance (S4 AttentionPool)
#         Show mean weight per token position, coloured by decoded token type
# ─────────────────────────────────────────────────────────────────────────────
print("[per_compound] Plotting ligand token importance …")

# Align all s4 vectors to a common length (pad with 0 for shorter compounds)
max_l = max(l_lens)
s4_mat = np.zeros((N, max_l), np.float32)
for ci in range(N):
    s4_mat[ci, :l_lens[ci]] = s4_lig_all[ci]

# Count of compounds reaching each position
pos_count = np.array([(s4_mat[:, j] > 0).sum() for j in range(max_l)])

mean_s4    = s4_mat.sum(axis=0) / np.clip(pos_count, 1, None)
std_s4     = np.sqrt(
    ((s4_mat - mean_s4[None, :]) ** 2 * (s4_mat > 0)).sum(axis=0) /
    np.clip(pos_count, 1, None)
)

fig, ax = plt.subplots(figsize=(14, 4))
xs = np.arange(max_l)
ax.bar(xs, mean_s4, width=0.9, color="mediumpurple", alpha=0.7,
       label="Mean S4 weight")
ax.fill_between(xs, np.clip(mean_s4 - std_s4, 0, None),
                mean_s4 + std_s4, alpha=0.3, color="mediumpurple")
ax.axhline(0.0, color="k", linewidth=0.5)

# Annotate positions that appear in most compounds with the token label
# (use modal token at each position)
for j in range(min(max_l, 50)):
    tokens_j = [token_labels[ci][j] for ci in range(N) if j < l_lens[ci]]
    if tokens_j:
        from collections import Counter
        modal_tok = Counter(tokens_j).most_common(1)[0][0]
        ax.text(j, mean_s4[j] + 0.001, modal_tok, ha="center",
                va="bottom", fontsize=5, rotation=90)

ax.axhline(0, color="k", linewidth=0.3)
ax.set_xlabel("Ligand token position (ChemBERTa)", fontsize=10)
ax.set_ylabel("Mean S4 AttentionPool weight", fontsize=10)
ax.set_title(f"Ligand token importance (S4) — {N} compounds\n"
             f"Higher weight = more influential token position in final prediction",
             fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "token_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUT_DIR}/token_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Per-compound pKd vs protein-attention entropy (scatter)
# ─────────────────────────────────────────────────────────────────────────────
from scipy.stats import entropy as scipy_entropy

print("[per_compound] Plotting pKd–entropy scatter …")

s3_entropy = np.array([
    float(scipy_entropy(s3_prot_all[ci]) / np.log(p_len))   # normalised [0,1]
    for ci in range(N)
])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(pkd_arr, s3_entropy, c=pkd_arr, cmap="plasma",
                s=20, alpha=0.7, edgecolors="none")
axes[0].set_xlabel("True pKd", fontsize=10)
axes[0].set_ylabel("S3 entropy (normalised)", fontsize=10)
axes[0].set_title("pKd vs protein attention entropy\n"
                  "Low entropy = focused attention; High = diffuse", fontsize=9)

from scipy.stats import pearsonr
r, p = pearsonr(pkd_arr, s3_entropy)
axes[0].text(0.05, 0.95, f"r = {r:.3f}  p = {p:.3f}",
             transform=axes[0].transAxes, fontsize=9, va="top")

# Also: predicted pKd error vs entropy
errors = np.abs(np.array(preds) - pkd_arr)
axes[1].scatter(s3_entropy, errors, c=pkd_arr, cmap="plasma",
                s=20, alpha=0.7, edgecolors="none")
axes[1].set_xlabel("S3 entropy (normalised)", fontsize=10)
axes[1].set_ylabel("|pred − true| pKd", fontsize=10)
axes[1].set_title("Prediction error vs protein attention entropy\n"
                  "Does diffuse attention predict harder?", fontsize=9)
r2, p2 = pearsonr(s3_entropy, errors)
axes[1].text(0.05, 0.95, f"r = {r2:.3f}  p = {p2:.3f}",
             transform=axes[1].transAxes, fontsize=9, va="top")

plt.tight_layout()
plt.savefig(OUT_DIR / "compound_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {OUT_DIR}/compound_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 (bonus): Consensus attribution heatmap if IG succeeded
# ─────────────────────────────────────────────────────────────────────────────
if HAS_IG:
    print("[per_compound] Plotting consensus attribution heatmap …")

    # Consensus = IG × S3 (normalised per compound)
    # Show as (N, p_len) matrix ordered by pKd
    order = np.argsort(pkd_arr)
    con_mat = consensus_prot[order]                 # (N, p_len)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(con_mat, aspect="auto", cmap="magma",
                   origin="upper", interpolation="nearest")
    ax.set_xlabel("Protein residue position", fontsize=10)
    ax.set_ylabel(f"Compound (sorted by pKd: {pkd_arr.min():.1f}→{pkd_arr.max():.1f})",
                  fontsize=9)
    ax.set_title("Consensus Attribution (IG × AttentionPool) per compound\n"
                 "Persistent hot spots = structurally important binding residues",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    # Annotate top-5 persistent residue positions
    mean_con = con_mat.mean(axis=0)
    top5 = np.argsort(mean_con)[-5:]
    for pos in top5:
        ax.axvline(pos, color="cyan", alpha=0.6, linewidth=1.0)
        ax.text(pos, -1.5, str(pos), fontsize=6, ha="center", color="cyan")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "consensus_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {OUT_DIR}/consensus_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics printout
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PER-COMPOUND ANALYSIS SUMMARY")
print("="*60)
print(f"Protein            : {len(top_protein_seq)} aa  |  {top_protein_seq[:60]}…")
print(f"Compounds analysed : {N}  (pKd {pkd_arr.min():.2f}–{pkd_arr.max():.2f})")
print(f"Model RMSE         : {np.sqrt(np.mean((np.array(preds)-pkd_arr)**2)):.4f}")
print(f"\nProtein S3 AttentionPool (mean over all compounds):")
top10_s3_sorted = top10_s3[np.argsort(mean_s3[top10_s3])[::-1]]
for pos in top10_s3_sorted[:10]:
    print(f"  Residue {pos:4d} : mean weight = {mean_s3[pos]:.5f}  "
          f"(uniform = {1/p_len:.5f}, ratio = {mean_s3[pos]*p_len:.1f}×)")
print(f"\nEntropy stats:")
print(f"  Mean normalised entropy : {s3_entropy.mean():.4f}")
print(f"  Std                     : {s3_entropy.std():.4f}")
print(f"  Min (most focused)      : {s3_entropy.min():.4f}  "
      f"[pKd = {pkd_arr[s3_entropy.argmin()]:.2f}]")
print(f"  Max (most diffuse)      : {s3_entropy.max():.4f}  "
      f"[pKd = {pkd_arr[s3_entropy.argmax()]:.2f}]")
if HAS_IG:
    print(f"\nConsensus top-5 persistent residues : {sorted(top5.tolist())}")
    print(f"  (persistent = high in both IG and AttentionPool across compounds)")
print(f"\nFigures written to: {OUT_DIR.resolve()}")
print("="*60)
