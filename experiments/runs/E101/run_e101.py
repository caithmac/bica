"""
E101 — BindingDB Re-Cleaning Pipeline
======================================
Reproducible cleaning of the BALM BindingDB_filtered dataset.
Documents every drop decision with curation log.

Source: BALM/BALM-benchmark (BindingDB_filtered) on HuggingFace
  - Already: Kd-only filtering, pKd conversion by BALM authors
  - We add: RDKit canonical validation, dedup, conflict flagging,
            protein sequence QC, per-target minimum compound filtering

Run: python experiments/runs/E101/run_e101.py
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from datasets import load_dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
EXP_DIR = Path(__file__).parent  # experiments/runs/E101/
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = EXP_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("E101")

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "experiment": "E101",
    "description": "BindingDB re-cleaning pipeline",
    "source": "BALM/BALM-benchmark (BindingDB_filtered)",
    "steps": [
        "1_load_raw",        # Load BALM HF dataset
        "2_canonical_smiles", # RDKit canonical SMILES, drop invalid
        "3_dedup",            # Drop exact (canonical SMILES, protein seq) duplicates
        "4_conflict_flag",    # Flag pairs with |ΔpKd| > 1.0 per (smiles, protein)
        "5_protein_qc",       # Standard 20 AA check, drop >5% non-standard
        "6_min_compounds",    # Drop targets with <10 compounds
    ],
    "thresholds": {
        "conflict_max_delta_pkd": 1.0,    # |ΔpKd| threshold for conflict flagging
        "protein_max_nonstandard_pct": 5.0, # max % non-standard AA residues
        "min_compounds_per_target": 10,    # minimum compounds per protein target
    },
    "standard_aa": "ACDEFGHIKLMNPQRSTVWY",
    "timestamp": datetime.now().isoformat(),
}
CONFIG["standard_aa_set"] = set(CONFIG["standard_aa"])

with open(EXP_DIR / "config.yaml", "w") as f:
    yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)

# ── Curation log ───────────────────────────────────────────────────────────────
curation_steps = []  # list of dicts: step, rows_in, rows_out, n_dropped, drop_reason

def log_step(step_name, df_before, df_after, reason=""):
    n_dropped = len(df_before) - len(df_after)
    curation_steps.append({
        "step": step_name,
        "rows_in": len(df_before),
        "rows_out": len(df_after),
        "n_dropped": n_dropped,
        "drop_pct": round(100 * n_dropped / len(df_before), 2) if len(df_before) > 0 else 0.0,
        "drop_reason": reason,
    })
    log.info(f"  {step_name}: {len(df_before):,} → {len(df_after):,} "
             f"({n_dropped:,} dropped, {reason})")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load raw data from HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 1: Loading BALM/BALM-benchmark (BindingDB_filtered)")
log.info("=" * 70)

t0 = time.time()
ds = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered")
if "train" in ds:
    df = ds["train"].to_pandas()
else:
    split_key = list(ds.keys())[0]
    df = ds[split_key].to_pandas()

# Keep only the columns we need
df = df[["Drug", "Target", "Y"]].copy()
df.columns = ["smiles_raw", "protein_seq", "pkd"]
df = df.dropna().reset_index(drop=True)
df["pkd"] = df["pkd"].astype(float)

n_raw = len(df)
log.info(f"Loaded {n_raw:,} rows in {time.time()-t0:.1f}s")
log.info(f"  pKd range: [{df['pkd'].min():.2f}, {df['pkd'].max():.2f}]")
log.info(f"  Unique raw SMILES: {df['smiles_raw'].nunique():,}")
log.info(f"  Unique proteins (seq): {df['protein_seq'].nunique():,}")

df_before = df.copy()
log_step("1_load_raw", df_before, df_before, "Initial load from BALM HF")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: RDKit canonical SMILES
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 2: RDKit canonical SMILES + validation")
log.info("=" * 70)

dropped_invalid = []
dropped_nostructure = []
canonical_smiles = []
valid_mask = []

for i, smi in enumerate(df["smiles_raw"]):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            dropped_invalid.append(i)
            valid_mask.append(False)
        elif mol.GetNumAtoms() == 0:
            dropped_nostructure.append(i)
            valid_mask.append(False)
        else:
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            valid_mask.append(True)
    except Exception:
        dropped_invalid.append(i)
        valid_mask.append(False)

df = df[valid_mask].reset_index(drop=True)
df["smiles_canonical"] = canonical_smiles

reason = f"RDKit parse failed ({len(dropped_invalid)}); empty mol ({len(dropped_nostructure)})"
log_step("2_canonical_smiles", df_before, df, reason)

# Save dropped invalid SMILES for supplementary
dropped_invalid_df = df_before.iloc[dropped_invalid][["smiles_raw", "protein_seq", "pkd"]]
dropped_invalid_df["drop_reason"] = "RDKit parse failure"
dropped_invalid_df.to_csv(EXP_DIR / "dropped_invalid_smiles.csv", index=False)
log.info(f"  Saved {len(dropped_invalid_df)} dropped invalid SMILES to dropped_invalid_smiles.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Drop exact duplicates
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 3: Drop exact duplicates (canonical SMILES + protein sequence)")
log.info("=" * 70)

df_before = df.copy()

# For duplicate (smiles, protein) pairs, keep the one with highest pKd
# (strongest binding, most conservative for false positives)
df["dup_key"] = df["smiles_canonical"] + "|||" + df["protein_seq"]
dup_counts = df["dup_key"].value_counts()
n_dup_pairs = (dup_counts > 1).sum()
log.info(f"  Duplicate groups (>1 row per unique SMILES+protein): {n_dup_pairs}")

# Sort by pKd descending within each group so we keep the highest
df = df.sort_values("pkd", ascending=False)
df = df.drop_duplicates(subset="dup_key", keep="first")
df = df.drop(columns=["dup_key"]).reset_index(drop=True)

log_step("3_dedup", df_before, df, f"{n_dup_pairs} duplicate groups resolved")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Flag conflict groups
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 4: Flag conflict groups (|ΔpKd| > 1.0 for same SMILES + protein)")
log.info("=" * 70)

# Now that we've deduped, check if there are still SMILES+protein pairs
# with different pKd values (shouldn't happen after dedup, but just in case)
# Actually, this step is about identifying conflicts in the ORIGINAL data
# before dedup. We already saved the highest-pKd row. Now we need to go back
# to the raw data to count how many had conflicts.

# Re-check on df_before (pre-dedup)
conflict_groups = []
conflict_df_before = df_before.copy()
conflict_df_before["dup_key"] = conflict_df_before["smiles_canonical"] + "|||" + conflict_df_before["protein_seq"]
for key, group in conflict_df_before.groupby("dup_key"):
    if len(group) > 1:
        pkd_range = group["pkd"].max() - group["pkd"].min()
        if pkd_range > CONFIG["thresholds"]["conflict_max_delta_pkd"]:
            conflict_groups.append({
                "smiles": group["smiles_canonical"].iloc[0],
                "protein_seq": group["protein_seq"].iloc[0][:50],  # truncated for readability
                "n_measurements": len(group),
                "pkd_min": group["pkd"].min(),
                "pkd_max": group["pkd"].max(),
                "pkd_range": pkd_range,
            })

log.info(f"  Conflict groups (|ΔpKd| > {CONFIG['thresholds']['conflict_max_delta_pkd']}): {len(conflict_groups)}")

if conflict_groups:
    conflict_df = pd.DataFrame(conflict_groups).sort_values("pkd_range", ascending=False)
    conflict_df.to_csv(EXP_DIR / "conflict_groups.csv", index=False)
    log.info(f"  Saved {len(conflict_groups)} conflict groups to conflict_groups.csv")
else:
    log.info("  No conflict groups found after dedup")

df_before = df.copy()
log_step("4_conflict_flag", df_before, df, f"{len(conflict_groups)} conflict groups flagged")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Protein sequence quality control
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 5: Protein sequence QC (standard 20 AA only)")
log.info("=" * 70)

df_before = df.copy()

standard_aa = CONFIG["standard_aa_set"]
max_nonstd_pct = CONFIG["thresholds"]["protein_max_nonstandard_pct"]

def validate_protein(seq):
    """Return (valid, nonstandard_pct)."""
    if not isinstance(seq, str) or len(seq) == 0:
        return False, 100.0
    seq_upper = seq.upper()
    n_nonstd = sum(1 for aa in seq_upper if aa not in standard_aa)
    pct = 100.0 * n_nonstd / len(seq_upper)
    return pct <= max_nonstd_pct, pct

valid_proteins = []
nonstandard_pcts = []
for seq in df["protein_seq"]:
    is_valid, pct = validate_protein(seq)
    valid_proteins.append(is_valid)
    nonstandard_pcts.append(pct)

n_dropped_protein = sum(1 for v in valid_proteins if not v)
df = df[valid_proteins].reset_index(drop=True)
log.info(f"  Dropped {n_dropped_protein} rows with >{max_nonstd_pct}% non-standard AA")

# Also clean: strip non-standard characters from sequences for the clean output
# (keep the original sequence but make a cleaned version)
df["protein_seq_clean"] = df["protein_seq"].apply(
    lambda s: "".join(aa for aa in s.upper() if aa in standard_aa)
)

log_step("5_protein_qc", df_before, df, 
         f"Sequences with >{max_nonstd_pct}% non-standard AA removed ({n_dropped_protein} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Drop targets with fewer than N compounds
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 6: Filter targets with <10 compounds")
log.info("=" * 70)

df_before = df.copy()

min_compounds = CONFIG["thresholds"]["min_compounds_per_target"]

target_counts = df["protein_seq_clean"].value_counts()
valid_targets = target_counts[target_counts >= min_compounds].index
n_valid_targets = len(valid_targets)
n_dropped_targets = len(target_counts) - n_valid_targets
df = df[df["protein_seq_clean"].isin(valid_targets)].reset_index(drop=True)

log.info(f"  Targets with ≥{min_compounds} compounds: {n_valid_targets}")
log.info(f"  Targets dropped (<{min_compounds} compounds): {n_dropped_targets}")

log_step("6_min_compounds", df_before, df,
         f"Targets with <{min_compounds} compounds removed ({n_dropped_targets} targets)")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL: Save clean dataset + summary
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("FINAL: Saving clean dataset")
log.info("=" * 70)

# Final dataframe with clean columns
df_clean = df[["smiles_canonical", "protein_seq_clean", "pkd"]].copy()
df_clean.columns = ["smiles", "protein_seq", "pkd"]

clean_path = EXP_DIR / "bindingdb_clean.csv"
df_clean.to_csv(clean_path, index=False)
log.info(f"  Saved {len(df_clean):,} clean rows to {clean_path}")

# Summary stats
log.info(f"\n{'='*50}")
log.info(f"CLEAN DATASET SUMMARY")
log.info(f"{'='*50}")
log.info(f"  Total compounds (unique SMILES): {df_clean['smiles'].nunique():,}")
log.info(f"  Total protein targets (unique seqs): {df_clean['protein_seq'].nunique():,}")
log.info(f"  Total interactions: {len(df_clean):,}")
log.info(f"  pKd: mean={df_clean['pkd'].mean():.2f} ± {df_clean['pkd'].std():.2f}")
log.info(f"  pKd range: [{df_clean['pkd'].min():.2f}, {df_clean['pkd'].max():.2f}]")
log.info(f"  Compounds per target: "
         f"median={df_clean.groupby('protein_seq').size().median():.0f}, "
         f"mean={df_clean.groupby('protein_seq').size().mean():.1f}")

# ── Save curation log ─────────────────────────────────────────────────────────
curation_df = pd.DataFrame(curation_steps)
curation_df.to_csv(EXP_DIR / "curation_log.csv", index=False)
log.info(f"\n  Curation log saved to curation_log.csv")

# ── Save curation summary as LaTeX table ───────────────────────────────────────
latex_lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{BindingDB curation pipeline summary.}",
    r"\label{tab:curation}",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"Step & Rows In & Rows Out & Dropped & Drop \% \\",
    r"\midrule",
]
for step in curation_steps:
    latex_lines.append(
        f"{step['step']} & {step['rows_in']:,} & {step['rows_out']:,} & "
        f"{step['n_dropped']:,} & {step['drop_pct']:.1f}\% \\\\"
    )
latex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])

with open(EXP_DIR / "curation_summary.tex", "w") as f:
    f.write("\n".join(latex_lines))
log.info(f"  Curation LaTeX table saved to curation_summary.tex")

# ── Overall timing ────────────────────────────────────────────────────────────
elapsed = time.time() - t0
log.info(f"\n  Total time: {elapsed/60:.1f} min ({elapsed:.0f}s)")

# ── Update PROGRESS.md ────────────────────────────────────────────────────────
progress_path = Path("E:/Drug Discovery/experiments/PROGRESS.md")
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
log.info(f"\n  Update {progress_path} to mark E101 as done")
print(f"\n[E101] DONE — Update PROGRESS.md with timestamp {timestamp}")
print(f"[E101] Clean dataset: {clean_path}")
print(f"[E101] Rows: {n_raw:,} → {len(df_clean):,}")
