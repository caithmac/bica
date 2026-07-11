#!/usr/bin/env python3
"""
E202-BALM: MLP head sweep on BALM BindingDB_filtered with cold-drug split.
Replicates E202 methodology (5 configs x 3 seeds = 15 fits) on the BALM benchmark.
Also runs RF baseline on the exact same split for direct comparison.

Features: ESM-2 8M + ChemBERTa-77M-MTR (frozen, mean-pooled)
Split: BALM cold-drug (unique Drug_IDs → test, no overlap)
Preprocessing: BALM aggregation (groupby drug-target, max Y)
"""
import sys, os, time, logging, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
RDLogger.logger().setLevel(RDLogger.ERROR)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXP_DIR = Path("E:/Drug Discovery/experiments/runs/E202_BALM")
EXP_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(EXP_DIR / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("E202_BALM")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ESM-2 and ChemBERTa model names (frozen encoders only)
ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"

# ---------------------------------------------------------------------------
# 1. Load + Preprocess BALM BindingDB_filtered
# ---------------------------------------------------------------------------
log.info("Loading BALM/BALM-benchmark BindingDB_filtered...")
ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered", split="train")
df = ds.to_pandas()
log.info(f"  Raw: {len(df)} rows")

# BALM preprocessing: group by drug-target, take max Y
df = df.groupby(['Drug_ID', 'Drug', 'Target_ID', 'Target'])['Y'].agg('max').reset_index()
log.info(f"  After BALM aggregation: {len(df)} rows")
log.info(f"  Unique drugs: {df['Drug'].nunique()}, targets: {df['Target'].nunique()}")

# Drop any rows with missing SMILES or targets
df = df.dropna(subset=['Drug', 'Target']).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Cold-drug split (BALM method)
# ---------------------------------------------------------------------------
def cold_drug_split(df, seed, test_frac=0.2):
    """BALM cold-drug: sample unique Drug values → test. No drug overlap."""
    rng = np.random.RandomState(seed)
    unique_drugs = df['Drug'].drop_duplicates().values
    n_test = int(len(unique_drugs) * test_frac)
    test_drugs = set(rng.choice(unique_drugs, size=n_test, replace=False))
    test_idx = df[df['Drug'].isin(test_drugs)].index.tolist()
    train_idx = df[~df['Drug'].isin(test_drugs)].index.tolist()
    return train_idx, test_idx

# We'll use seed=42 for the main split, then vary training seeds
TRAIN_IDX, TEST_IDX = cold_drug_split(df, seed=42)
log.info(f"Cold-drug split (seed=42): {len(TRAIN_IDX)} train, {len(TEST_IDX)} test")
train_drugs = set(df.iloc[TRAIN_IDX]['Drug'])
test_drugs = set(df.iloc[TEST_IDX]['Drug'])
assert len(train_drugs & test_drugs) == 0, "Drug leakage detected!"
log.info("  √ No drug overlap")

# ---------------------------------------------------------------------------
# 3. Compute ESM-2 embeddings for proteins
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_esm_embeddings(sequences, model, tokenizer, batch_size=64):
    """Mean-pooled ESM-2 embeddings."""
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        out = model(**tokens)
        # Mean pool over sequence length (excluding padding)
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

@torch.no_grad()
def compute_chemberta_embeddings(smiles_list, model, tokenizer, batch_size=64):
    """Mean-pooled ChemBERTa embeddings."""
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        out = model(**tokens)
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

# Load models
log.info(f"Loading ESM-2: {ESM_MODEL}")
esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
esm_model = AutoModel.from_pretrained(ESM_MODEL, use_safetensors=False).to(DEVICE).eval()

log.info(f"Loading ChemBERTa: {CHEMBERTA_MODEL}")
cb_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
cb_model = AutoModel.from_pretrained(CHEMBERTA_MODEL).to(DEVICE).eval()

# Get all sequences and SMILES
all_targets = df['Target'].tolist()
all_drugs = df['Drug'].tolist()

# Compute embeddings
log.info("Computing ESM-2 protein embeddings...")
P_all = compute_esm_embeddings(all_targets, esm_model, esm_tokenizer)
log.info(f"  ESM-2 shape: {P_all.shape}")

log.info("Computing ChemBERTa drug embeddings...")
L_all = compute_chemberta_embeddings(all_drugs, cb_model, cb_tokenizer)
log.info(f"  ChemBERTa shape: {L_all.shape}")

# Concatenate features
X_all = np.concatenate([L_all, P_all], axis=1)
y_all = df['Y'].values.astype(np.float32)
log.info(f"  Combined features: {X_all.shape}")

# Split into train/test
X_train_full = X_all[TRAIN_IDX]
y_train_full = y_all[TRAIN_IDX]
X_test = X_all[TEST_IDX]
y_test = y_all[TEST_IDX]

# Save features for reuse
np.save(EXP_DIR / "X_train.npy", X_train_full)
np.save(EXP_DIR / "y_train.npy", y_train_full)
np.save(EXP_DIR / "X_test.npy", X_test)
np.save(EXP_DIR / "y_test.npy", y_test)
log.info(f"Features saved to {EXP_DIR}/")

# ---------------------------------------------------------------------------
# 4. RF Baseline (on the exact same split)
# ---------------------------------------------------------------------------
log.info("\n=== RF Baseline ===")
# ECFP4 fingerprints for RF (since RF can't use DL embeddings directly)
def compute_ecfp4(smiles_list, radius=2, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32))
    return np.array(fps)

# AAC for RF
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
def compute_aac(sequence):
    if not isinstance(sequence, str) or len(sequence) == 0:
        return [0]*20
    counts = [sequence.count(aa) for aa in AMINO_ACIDS]
    total = sum(counts)
    return [c/total for c in counts] if total > 0 else [0]*20

train_smiles = df.iloc[TRAIN_IDX]['Drug'].tolist()
test_smiles = df.iloc[TEST_IDX]['Drug'].tolist()
train_targets = df.iloc[TRAIN_IDX]['Target'].tolist()
test_targets = df.iloc[TEST_IDX]['Target'].tolist()

ecfp4_train = compute_ecfp4(train_smiles)
ecfp4_test = compute_ecfp4(test_smiles)
aac_train = np.array([compute_aac(s) for s in train_targets])
aac_test = np.array([compute_aac(s) for s in test_targets])

X_rf_train = np.concatenate([ecfp4_train, aac_train], axis=1)
X_rf_test = np.concatenate([ecfp4_test, aac_test], axis=1)

rf_seeds = [42, 123, 456]
rf_results = []
for s in rf_seeds:
    rf = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=s, n_jobs=-1)
    rf.fit(X_rf_train, y_train_full)
    yp = rf.predict(X_rf_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, yp)))
    r_val, _ = pearsonr(y_test, yp)
    rf_results.append({'seed': s, 'rmse': rmse, 'pearson_r': r_val})
    log.info(f"  RF seed={s}: RMSE={rmse:.4f}, R={r_val:.4f}")

rf_rmse_mean = np.mean([r['rmse'] for r in rf_results])
rf_rmse_std = np.std([r['rmse'] for r in rf_results], ddof=1)
rf_r_mean = np.mean([r['pearson_r'] for r in rf_results])
log.info(f"  RF baseline: RMSE={rf_rmse_mean:.4f}±{rf_rmse_std:.4f}, R={rf_r_mean:.4f}")

# ---------------------------------------------------------------------------
# 5. MLP sweep (same configs as E202)
# ---------------------------------------------------------------------------
log.info("\n=== MLP Sweep ===")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(model, X_train, y_train, X_test, y_test, batch_size=128, lr=5e-4, max_epochs=100, patience=20):
    """Train with early stopping. Uses small validation split from training data."""
    # Split 10% of train for validation
    n_val = max(100, int(len(X_train) * 0.1))
    idx = np.random.RandomState(42).permutation(len(X_train))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    
    X_tr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train[tr_idx], dtype=torch.float32)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)
    
    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0
    epochs_done = 0
    
    for epoch in range(max_epochs):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(DEVICE)).cpu().numpy()
            val_rmse = float(np.sqrt(mean_squared_error(y_val.numpy(), val_pred)))
        
        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        epochs_done = epoch + 1
        if patience_counter >= patience:
            break
    
    # Restore best model and predict
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_te.to(DEVICE)).cpu().numpy()
    
    rmse = float(np.sqrt(mean_squared_error(y_te.numpy(), test_pred)))
    r_val = float(np.corrcoef(test_pred, y_te.numpy())[0, 1]) if len(np.unique(test_pred)) > 1 else 0.0
    return rmse, r_val, epochs_done

# Configs from E202
CONFIGS = [
    {"name": "shallow", "hidden": [256], "dropout": 0.2, "lr": 5e-4},
    {"name": "medium", "hidden": [512, 256], "dropout": 0.2, "lr": 5e-4},
    {"name": "deep", "hidden": [512, 256, 128], "dropout": 0.2, "lr": 5e-4},
    {"name": "deep_hidrop", "hidden": [512, 256, 128], "dropout": 0.5, "lr": 5e-4},
    {"name": "deep_lowlr", "hidden": [512, 256, 128], "dropout": 0.2, "lr": 1e-4},
]
SEEDS = [42, 123, 456]

results = []
for cfg in CONFIGS:
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = MLP(input_dim=X_train_full.shape[1], hidden_dims=cfg['hidden'], dropout=cfg['dropout'])
        rmse, r_val, epochs = train_mlp(
            model, X_train_full, y_train_full, X_test, y_test,
            batch_size=128, lr=cfg['lr'], max_epochs=100, patience=20
        )
        
        elapsed = time.time() - t0
        results.append({
            **cfg, "seed": seed, "rmse": rmse, "pearson_r": r_val,
            "epochs": epochs, "time_s": elapsed
        })
        log.info(f"  {cfg['name']:15s} seed={seed} RMSE={rmse:.4f} R={r_val:.4f} ep={epochs} t={elapsed:.0f}s")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
df_res = pd.DataFrame(results)
df_res.to_csv(EXP_DIR / "results.csv", index=False)

log.info("\n" + "="*80)
log.info("SUMMARY — MLP head sweep on BALM cold-drug split")
log.info("="*80)
for name in df_res['name'].unique():
    sub = df_res[df_res['name'] == name]
    log.info(f"  {name:15s}: RMSE={sub['rmse'].mean():.4f}±{sub['rmse'].std():.4f}  "
             f"best={sub['rmse'].min():.4f}  R={sub['pearson_r'].mean():.4f}")

log.info(f"\n  RF baseline: {rf_rmse_mean:.4f}±{rf_rmse_std:.4f}")
log.info(f"  Best DL:     {df_res['rmse'].min():.4f} (gap: {df_res['rmse'].min()-rf_rmse_mean:+.4f})")

# Save summary JSON
summary = {
    "dataset": "BALM BindingDB_filtered (cold-drug split)",
    "n_train": len(TRAIN_IDX),
    "n_test": len(TEST_IDX),
    "n_unique_drugs": int(df['Drug'].nunique()),
    "rf_baseline": {"rmse_mean": rf_rmse_mean, "rmse_std": rf_rmse_std, "pearson_r_mean": rf_r_mean},
    "dl_configs": []
}
for name in df_res['name'].unique():
    sub = df_res[df_res['name'] == name]
    summary['dl_configs'].append({
        "name": name,
        "rmse_mean": float(sub['rmse'].mean()),
        "rmse_std": float(sub['rmse'].std()),
        "rmse_best": float(sub['rmse'].min()),
        "pearson_r_mean": float(sub['pearson_r'].mean()),
    })

with open(EXP_DIR / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

log.info(f"\nResults saved to {EXP_DIR}/")
log.info("Done.")
