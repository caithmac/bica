"""
BALM-style training: ChemBERTa + ESM-2 with PEFT + cosine similarity loss,
evaluated per-target with Fisher z-transformed Pearson.
"""
import os
import math
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import LoKrConfig, LoHaConfig, get_peft_model
from scipy.stats import pearsonr

OUT_DIR = Path("results/balm_fisher")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────
ESM_MODEL = "facebook/esm2_t12_35M_UR50D"   # 35M params, 480-dim (fits A4000)
ESM_DIM = 480
CB_MODEL = "DeepChem/ChemBERTa-77M-MTR"      # 384-dim hidden
CB_DIM = 384
PROJECTED_DIM = 256
BATCH_SIZE = 8
MAX_EPOCHS = 50
PATIENCE = 10
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Fisher transform helpers ──────────────────────────────────────────────
def fisher_z(r):
    """Fisher z-transform: r → z"""
    r = max(min(r, 0.9999), -0.9999)
    return 0.5 * math.log((1 + r) / (1 - r))


def inv_fisher_z(z):
    """Inverse Fisher: z → r"""
    return (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)


def per_target_fisher_pearson(y_true, y_pred, target_ids):
    """
    Compute per-target Pearson r, Fisher z-transform each,
    average, then back-transform. Each target gets equal weight.
    """
    df = pd.DataFrame({"y": y_true, "pred": y_pred, "tid": target_ids})
    zs = []
    ns = []
    for tid, g in df.groupby("tid"):
        if len(g) < 3:  # need at least 3 points for meaningful correlation
            continue
        r, _ = pearsonr(g["y"], g["pred"])
        zs.append(fisher_z(r))
        ns.append(len(g))
    if not zs:
        return float("nan"), float("nan"), []
    mean_z = np.mean(zs)
    mean_r = inv_fisher_z(mean_z)
    return mean_z, mean_r, ns


# ── Data loading ──────────────────────────────────────────────────────────
print("\n[1] Loading data...")
ds_bdb = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered", split="train")
df_bdb = ds_bdb.to_pandas()
print(f"    BindingDB: {len(df_bdb):,} rows, {df_bdb['Target_ID'].nunique()} targets")

# Train/val/test split by random 70/10/20
np.random.seed(42)
n = len(df_bdb)
idx = np.random.permutation(n)
n_tr = int(n * 0.70)
n_vl = int(n * 0.10)
train_idx = idx[:n_tr]
val_idx = idx[n_tr:n_tr + n_vl]
test_idx = idx[n_tr + n_vl:]

train_df = df_bdb.iloc[train_idx].reset_index(drop=True)
val_df = df_bdb.iloc[val_idx].reset_index(drop=True)
test_df = df_bdb.iloc[test_idx].reset_index(drop=True)
print(f"    Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")


# ── Tokenizers ────────────────────────────────────────────────────────────
print("\n[2] Loading tokenizers & models...")
esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
cb_tokenizer = AutoTokenizer.from_pretrained(CB_MODEL)

# ── PEFT configs ──────────────────────────────────────────────────────────
esm_peft_config = LoKrConfig(
    r=16, alpha=32, rank_dropout=0.0, module_dropout=0.0,
    target_modules=["query", "key", "value"],
)
cb_peft_config = LoHaConfig(
    r=16, alpha=32, rank_dropout=0.0, module_dropout=0.0,
    target_modules=["query", "key", "value"],
)


# ── Model: projection + cosine similarity ─────────────────────────────────
class BALMModel(nn.Module):
    def __init__(self, esm, cb, esm_dim, cb_dim, proj_dim):
        super().__init__()
        self.esm = esm
        self.cb = cb
        self.proj_protein = nn.Linear(esm_dim, proj_dim)
        self.proj_ligand = nn.Linear(cb_dim, proj_dim)

    def encode(self, prot_ids, prot_mask, lig_ids, lig_mask):
        # Protein
        prot_out = self.esm(input_ids=prot_ids, attention_mask=prot_mask)
        prot_emb = prot_out.last_hidden_state * prot_mask.unsqueeze(-1).float()
        prot_emb = prot_emb.sum(1) / prot_mask.sum(1, keepdim=True).float().clamp(min=1)
        prot_proj = nn.functional.normalize(self.proj_protein(prot_emb), dim=-1)

        # Ligand
        lig_out = self.cb(input_ids=lig_ids, attention_mask=lig_mask)
        lig_emb = lig_out.last_hidden_state * lig_mask.unsqueeze(-1).float()
        lig_emb = lig_emb.sum(1) / lig_mask.sum(1, keepdim=True).float().clamp(min=1)
        lig_proj = nn.functional.normalize(self.proj_ligand(lig_emb), dim=-1)

        return prot_proj, lig_proj

    def forward(self, prot_ids, prot_mask, lig_ids, lig_mask):
        prot_proj, lig_proj = self.encode(prot_ids, prot_mask, lig_ids, lig_mask)
        return nn.functional.cosine_similarity(prot_proj, lig_proj, dim=-1)


# ── Load & wrap models ────────────────────────────────────────────────────
print("    Loading ESM-2 150M...")
esm_base = AutoModel.from_pretrained(ESM_MODEL)
esm = get_peft_model(esm_base, esm_peft_config)
print(f"    ESM trainable: {sum(p.numel() for p in esm.parameters() if p.requires_grad):,}")

print("    Loading ChemBERTa-77M...")
cb_base = AutoModel.from_pretrained(CB_MODEL)
cb = get_peft_model(cb_base, cb_peft_config)
print(f"    CB trainable: {sum(p.numel() for p in cb.parameters() if p.requires_grad):,}")

model = BALMModel(esm, cb, ESM_DIM, CB_DIM, PROJECTED_DIM).to(DEVICE)
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    Total trainable params: {total_trainable:,}")


# ── Tokenize dataset ──────────────────────────────────────────────────────
class BindingDBDataset(Dataset):
    def __init__(self, df, esm_tok, cb_tok, max_prot=1024, max_lig=512):
        self.df = df.reset_index(drop=True)
        self.esm_tok = esm_tok
        self.cb_tok = cb_tok
        self.max_prot = max_prot
        self.max_lig = max_lig

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prot = row["Target"][:self.max_prot]
        lig = str(row["Drug"])[:self.max_lig]
        y = float(row["Y"])
        tid = row["Target_ID"]

        # Tokenize individually (batching done by collator)
        prot_enc = self.esm_tok(prot, return_tensors="pt", padding=False,
                                truncation=True, max_length=self.max_prot)
        lig_enc = self.cb_tok(lig, return_tensors="pt", padding=False,
                              truncation=True, max_length=self.max_lig)
        return {
            "prot_ids": prot_enc["input_ids"].squeeze(0),
            "prot_mask": prot_enc["attention_mask"].squeeze(0),
            "lig_ids": lig_enc["input_ids"].squeeze(0),
            "lig_mask": lig_enc["attention_mask"].squeeze(0),
            "y": torch.tensor(y, dtype=torch.float32),
            "tid": tid,
        }


def collate_fn(batch):
    prot_ids = nn.utils.rnn.pad_sequence([b["prot_ids"] for b in batch], batch_first=True, padding_value=1)
    prot_mask = nn.utils.rnn.pad_sequence([b["prot_mask"] for b in batch], batch_first=True, padding_value=0)
    lig_ids = nn.utils.rnn.pad_sequence([b["lig_ids"] for b in batch], batch_first=True, padding_value=1)
    lig_mask = nn.utils.rnn.pad_sequence([b["lig_mask"] for b in batch], batch_first=True, padding_value=0)
    y = torch.stack([b["y"] for b in batch])
    tids = [b["tid"] for b in batch]
    return prot_ids, prot_mask, lig_ids, lig_mask, y, tids


# ── Training ──────────────────────────────────────────────────────────────
print("\n[3] Training...")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.MSELoss()

train_ds = BindingDBDataset(train_df, esm_tokenizer, cb_tokenizer)
val_ds = BindingDBDataset(val_df, esm_tokenizer, cb_tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate_fn)

best_fisher_r = -1.0
best_state = None
patience_ctr = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch_idx, (prot_ids, prot_mask, lig_ids, lig_mask, yb, _) in enumerate(train_loader):
        if batch_idx == 0 or (batch_idx + 1) % 500 == 0:
            print(f"    epoch {epoch} batch {batch_idx+1}/{len(train_loader)}", flush=True)
        prot_ids, prot_mask = prot_ids.to(DEVICE), prot_mask.to(DEVICE)
        lig_ids, lig_mask = lig_ids.to(DEVICE), lig_mask.to(DEVICE)
        yb = yb.to(DEVICE)

        pred = model(prot_ids, prot_mask, lig_ids, lig_mask)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

        if batch_i == 0:
            dt = time.time() - t0
            print(f"    batch 0/{len(train_loader)} done in {dt:.1f}s", flush=True)

    # ── Validation (per-target Fisher) ────────────────────────────────
    model.eval()
    all_y, all_pred, all_tids = [], [], []
    with torch.no_grad():
        for prot_ids, prot_mask, lig_ids, lig_mask, yb, tids in val_loader:
            prot_ids, prot_mask = prot_ids.to(DEVICE), prot_mask.to(DEVICE)
            lig_ids, lig_mask = lig_ids.to(DEVICE), lig_mask.to(DEVICE)
            pred = model(prot_ids, prot_mask, lig_ids, lig_mask)
            all_y.extend(yb.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_tids.extend(tids)

    y_arr = np.array(all_y)
    p_arr = np.array(all_pred)
    global_r, _ = pearsonr(y_arr, p_arr)
    _, fisher_r, ns = per_target_fisher_pearson(y_arr, p_arr, all_tids)
    rmse = np.sqrt(np.mean((y_arr - p_arr) ** 2))

    if epoch <= 3 or epoch % 5 == 0:
        print(f"  epoch {epoch:3d} | loss={train_loss/len(train_loader):.4f} | "
              f"val_rmse={rmse:.4f} | global_r={global_r:.4f} | "
              f"fisher_r={fisher_r:.4f} (n_targets={len(ns)})")

    if fisher_r > best_fisher_r:
        best_fisher_r = fisher_r
        patience_ctr = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

# ── Load best & test ──────────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()

print(f"\n[4] Final evaluation (best fisher_r={best_fisher_r:.4f})")

# Test on BindingDB (per-target Fisher)
test_ds = BindingDBDataset(test_df, esm_tokenizer, cb_tokenizer)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate_fn)
all_y, all_pred, all_tids = [], [], []
with torch.no_grad():
    for prot_ids, prot_mask, lig_ids, lig_mask, yb, tids in test_loader:
        prot_ids, prot_mask = prot_ids.to(DEVICE), prot_mask.to(DEVICE)
        lig_ids, lig_mask = lig_ids.to(DEVICE), lig_mask.to(DEVICE)
        pred = model(prot_ids, prot_mask, lig_ids, lig_mask)
        all_y.extend(yb.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_tids.extend(tids)

y_arr = np.array(all_y)
p_arr = np.array(all_pred)
test_rmse = np.sqrt(np.mean((y_arr - p_arr) ** 2))
test_global_r, _ = pearsonr(y_arr, p_arr)
_, test_fisher_r, test_ns = per_target_fisher_pearson(y_arr, p_arr, all_tids)

print(f"  BindingDB test:")
print(f"    RMSE:     {test_rmse:.4f}")
print(f"    Global r: {test_global_r:.4f}")
print(f"    Fisher r: {test_fisher_r:.4f} (n_targets={len(test_ns)})")

# ── Zero-shot: USP7 & MPro ────────────────────────────────────────────────
print(f"\n[5] Zero-shot evaluation...")
results = {"bindingdb_test": {"rmse": test_rmse, "global_r": test_global_r,
                                "fisher_r": test_fisher_r, "n_targets": len(test_ns)}}

for zs_name in ["USP7", "Mpro"]:
    ds_zs = load_dataset("BALM/BALM-benchmark", zs_name, split="train")
    df_zs = ds_zs.to_pandas()
    ds_zs_ds = BindingDBDataset(df_zs, esm_tokenizer, cb_tokenizer)
    zs_loader = DataLoader(ds_zs_ds, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate_fn)

    all_y, all_pred, all_tids = [], [], []
    with torch.no_grad():
        for prot_ids, prot_mask, lig_ids, lig_mask, yb, tids in zs_loader:
            prot_ids, prot_mask = prot_ids.to(DEVICE), prot_mask.to(DEVICE)
            lig_ids, lig_mask = lig_ids.to(DEVICE), lig_mask.to(DEVICE)
            pred = model(prot_ids, prot_mask, lig_ids, lig_mask)
            all_y.extend(yb.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_tids.extend(tids)

    y_arr = np.array(all_y)
    p_arr = np.array(all_pred)
    zs_rmse = np.sqrt(np.mean((y_arr - p_arr) ** 2))
    zs_global_r, _ = pearsonr(y_arr, p_arr)
    _, zs_fisher_r, zs_ns = per_target_fisher_pearson(y_arr, p_arr, all_tids)

    print(f"  {zs_name} (zero-shot):")
    print(f"    RMSE:     {zs_rmse:.4f}")
    print(f"    Global r: {zs_global_r:.4f}")
    print(f"    Fisher r: {zs_fisher_r:.4f} (n_targets={len(zs_ns)})")

    results[zs_name] = {"rmse": zs_rmse, "global_r": zs_global_r,
                         "fisher_r": zs_fisher_r, "n_targets": len(zs_ns)}

# ── Save ──────────────────────────────────────────────────────────────────
with open(OUT_DIR / "results.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"\n[6] Done. Results saved to {OUT_DIR}/results.pkl")
