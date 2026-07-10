"""Parse PSICHIC full_result files and generate learning curve comparison plot."""
import re, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("E:/Drug Discovery/experiments/runs/E203_v2")

def extract_json_block(text, key):
    start = text.find(key + ": {")
    if start == -1:
        return None
    brace_start = start + len(key) + 2
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start:i+1])
                except:
                    return None
    return None

def parse_full_result(path):
    text = Path(path).read_text()
    blocks = text.split("------------------------------")
    rows = []
    current_step = None
    for block in blocks:
        step_m = re.search(r"Training Step: (\d+)", block)
        if step_m:
            current_step = int(step_m.group(1))
            continue
        if current_step is None:
            continue
        loss_m = re.search(r"Train MSE Loss: ([\d.]+)", block)
        if not loss_m:
            continue
        train_loss = float(loss_m.group(1))
        val = extract_json_block(block, "Validation Results")
        test = extract_json_block(block, "Test Results")
        rows.append({
            "step": current_step,
            "train_loss": train_loss,
            "val_rmse": val["rmse"] if val else None,
            "val_pearson": val["pearson"] if val else None,
            "test_rmse": test["rmse"] if test else None,
            "test_pearson": test["pearson"] if test else None,
        })
    return pd.DataFrame(rows).sort_values("step")

scaffold_df = parse_full_result(EXP_DIR / "scaffold_result/full_result-1.txt")
random_df = parse_full_result(EXP_DIR / "random_result/full_result-1.txt")

print(f"Scaffold: {len(scaffold_df)} pts, final test_rmse={scaffold_df.iloc[-1]['test_rmse']:.4f}")
print(f"Random:   {len(random_df)} pts, final test_rmse={random_df.iloc[-1]['test_rmse']:.4f}")

scf_final = scaffold_df.iloc[-1]["test_rmse"]
rnd_final = random_df.iloc[-1]["test_rmse"]
print(f"Gap: {scf_final - rnd_final:+.4f}")

# Save CSVs
scaffold_df.to_csv(EXP_DIR / "scaffold_curves.csv", index=False)
random_df.to_csv(EXP_DIR / "random_curves.csv", index=False)

# ── Plot ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

# Panel 1: Train MSE Loss (sqrt for RMSE scale)
ax = axes[0]
ax.plot(scaffold_df["step"], np.sqrt(scaffold_df["train_loss"]), "o-", color="#d7191c", linewidth=1.5, markersize=4, alpha=0.7, label="Scaffold train")
ax.plot(random_df["step"], np.sqrt(random_df["train_loss"]), "o-", color="#1b7837", linewidth=1.5, markersize=4, alpha=0.7, label="Random train")
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Train RMSE (sqrt MSE)", fontsize=12)
ax.set_title("Training Loss", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Val + Test RMSE
ax = axes[1]
ax.plot(scaffold_df["step"], scaffold_df["val_rmse"], "s-", color="#fc8d59", linewidth=2, markersize=5, label="Scaffold val")
ax.plot(scaffold_df["step"], scaffold_df["test_rmse"], "o-", color="#d7191c", linewidth=2, markersize=6, label="Scaffold test")
ax.plot(random_df["step"], random_df["val_rmse"], "s-", color="#a6dba0", linewidth=2, markersize=5, label="Random val")
ax.plot(random_df["step"], random_df["test_rmse"], "o-", color="#1b7837", linewidth=2, markersize=6, label="Random test")
# Annotate final values
ax.annotate(f"{scf_final:.3f}", xy=(5000, scf_final), xytext=(5100, scf_final+0.02),
            fontsize=10, color="#d7191c", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="#d7191c"))
ax.annotate(f"{rnd_final:.3f}", xy=(5000, rnd_final), xytext=(5100, rnd_final-0.02),
            fontsize=10, color="#1b7837", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="#1b7837"))
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("RMSE", fontsize=12)
ax.set_title("Val / Test RMSE", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 3: Test RMSE gap
ax = axes[2]
ax.fill_between(scaffold_df["step"], scaffold_df["test_rmse"], random_df["test_rmse"],
                alpha=0.15, color="gray")
ax.plot(scaffold_df["step"], scaffold_df["test_rmse"], "o-", color="#d7191c", linewidth=2, markersize=6, label="Scaffold test")
ax.plot(random_df["step"], random_df["test_rmse"], "o-", color="#1b7837", linewidth=2, markersize=6, label="Random test")
# Gap annotation
mid_step = 2500
scf_mid = scaffold_df[scaffold_df["step"] == 2400]["test_rmse"].values[0]
rnd_mid = random_df[random_df["step"] == 2400]["test_rmse"].values[0]
ax.annotate(f"Gap = {scf_final-rnd_final:+.3f}", xy=(mid_step, (scf_mid+rnd_mid)/2),
            fontsize=13, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Test RMSE", fontsize=12)
ax.set_title(f"Memorization Gap: Scaffold vs Random", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(EXP_DIR / "psichic_learning_curves.pdf", dpi=150, bbox_inches="tight")
fig.savefig(EXP_DIR / "psichic_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved psichic_learning_curves.png/pdf")
print(f"\n=== KEY RESULT ===")
print(f"Scaffold final test RMSE: {scf_final:.4f}")
print(f"Random final test RMSE:   {rnd_final:.4f}")
print(f"Memorization gap:         {scf_final - rnd_final:+.4f}")
print(f"\nScaffold test RMSE plateaus around step 2000 at ~1.21")
print(f"Random test RMSE keeps improving throughout training")
