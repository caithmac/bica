"""GAP-3 — Overfitting Evidence. Re-train top DL models with per-epoch logging."""
import sys, os, time, logging, json
from pathlib import Path
import numpy as np, pandas as pd

EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("E:/Drug Discovery")
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(EXP_DIR / "run.log", mode="w"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("GAP3")

from harness.config import LABEL_COL, BATCH_SIZE
from harness.data import get_splits_for_seed
import harness.featurizers as F
from harness.trainer import _get_device, _make_loader, _evaluate_loader
from models.mlp import MLP
import torch, torch.nn as nn

train_df, val_df, test_df = get_splits_for_seed(42)
FEAT_CACHE = Path("E:/Drug Discovery/cache/features")
X_train = F.concat(np.load(FEAT_CACHE / "cb600_train.npy"), np.load(FEAT_CACHE / "esm2_8M_train.npy"))
X_test  = F.concat(np.load(FEAT_CACHE / "cb600_test.npy"), np.load(FEAT_CACHE / "esm2_8M_test.npy"))
y_train = train_df[LABEL_COL].values.astype(np.float32)
y_test  = test_df[LABEL_COL].values.astype(np.float32)
X_val, y_val = X_train[:500], y_train[:500]

MODELS = ["mlp_shallow", "mlp_medium", "mlp_deep"]

all_curves = []
for name, hidden in [("shallow", [256]), ("medium", [512, 256]), ("deep", [512, 256, 128])]:
    log.info(f"\n{name} MLP: hidden={hidden}")
    device = _get_device()
    model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    train_loader = _make_loader(X_train, y_train, shuffle=True, batch_size=128)
    val_loader = _make_loader(X_val, y_val, shuffle=False, batch_size=512)
    test_loader = _make_loader(X_test, y_test, shuffle=False, batch_size=512)
    
    best_val, best_state, patience_ctr = float("inf"), None, 0
    curves = []
    
    for epoch in range(1, 101):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Per-epoch metrics
        train_m = _evaluate_loader(model, train_loader, device)
        val_m = _evaluate_loader(model, val_loader, device)
        scheduler.step(val_m["rmse"])
        
        curves.append({"epoch": epoch, "model": name,
                       "train_rmse": float(train_m["rmse"]), "val_rmse": float(val_m["rmse"])})
        
        if val_m["rmse"] < best_val:
            best_val = val_m["rmse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        
        if epoch % 20 == 0:
            log.info(f"  ep {epoch:3d}: train={train_m['rmse']:.4f} val={val_m['rmse']:.4f} "
                     f"gap={val_m['rmse']-train_m['rmse']:+.4f}")
    
    # Final test
    model.load_state_dict(best_state)
    test_m = _evaluate_loader(model, test_loader, device)
    curves_df = pd.DataFrame(curves)
    curves_df.to_csv(EXP_DIR / f"curves_{name}.csv", index=False)
    all_curves.append(curves_df)
    log.info(f"  Final test RMSE={test_m['rmse']:.4f}  overfit gap={curves_df['val_rmse'].iloc[-1]-curves_df['train_rmse'].iloc[-1]:+.4f}")

# Combine all curves
pd.concat(all_curves).to_csv(EXP_DIR / "all_curves.csv", index=False)

# Plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ARCH_LABELS = {"shallow": ("[256]", 1), "medium": ("[512, 256]", 2), "deep": ("[512, 256, 128]", 3)}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name in zip(axes, ["shallow", "medium", "deep"]):
    df = pd.read_csv(EXP_DIR / f"curves_{name}.csv")
    ax.plot(df["epoch"], df["train_rmse"], label="Train", color="#1b7837", linewidth=2)
    ax.plot(df["epoch"], df["val_rmse"], label="Val", color="#762a83", linewidth=2)
    ax.fill_between(df["epoch"], df["train_rmse"], df["val_rmse"], alpha=0.15, color="gray")
    arch_str, n_layers = ARCH_LABELS[name]
    ax.set_title(f"MLP {name} {arch_str}, {n_layers} layer{'s' if n_layers > 1 else ''}", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE")
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(EXP_DIR / "overfitting.pdf", dpi=150, bbox_inches="tight")
fig.savefig(EXP_DIR / "overfitting.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("\nSaved overfitting.pdf")
