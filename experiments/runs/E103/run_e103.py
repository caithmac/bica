"""
E103 — Three-Way Split Implementation
======================================
Generate three splits from E101 clean data:
  1. Scaffold split (Bemis-Murcko, seed 42)
  2. Sequence-identity split (MMseqs2 clustering at 40%)
  3. Random split (stratified, no grouping)

Run: python experiments/runs/E103/run_e103.py
"""

import sys
import os
import subprocess
import time
import logging
import pickle
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ── Paths ──────────────────────────────────────────────────────────────────────
EXP_DIR = Path(__file__).parent
EXP_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path("E:/Drug Discovery")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(EXP_DIR / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("E103")

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "experiment": "E103",
    "description": "Three-way split implementation",
    "data_source": "E101 clean (bindingdb_clean.csv)",
    "splits": {
        "scaffold": {"method": "Bemis-Murcko", "seed": 42, "ratios": "70/10/20"},
        "seqid": {"method": "MMseqs2 clustering", "identity": 0.40, "seed": 42, "ratios": "70/10/20"},
        "random": {"method": "Stratified random", "seed": 42, "ratios": "70/10/20"},
    },
    "train_frac": 0.70,
    "val_frac": 0.10,
    "test_frac": 0.20,
    "timestamp": datetime.now().isoformat(),
}

# ── Load E101 clean data ───────────────────────────────────────────────────────
clean_path = ROOT / "experiments/runs/E101/bindingdb_clean.csv"
df = pd.read_csv(clean_path)
log.info(f"Loaded E101 clean: {len(df):,} rows, {df['smiles'].nunique():,} compounds, "
         f"{df['protein_seq'].nunique():,} proteins")

# Add index column for tracking
df = df.reset_index(drop=True)
df["idx"] = df.index


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_scaffold(smiles):
    """Bemis-Murcko scaffold. Returns hashed fallback for invalids."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return f"__invalid__{hashlib.md5(str(smiles).encode()).hexdigest()}"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else f"__noscaffold__{hashlib.md5(str(smiles).encode()).hexdigest()}"
    except:
        return f"__invalid__{hashlib.md5(str(smiles).encode()).hexdigest()}"


def scaffold_split_df(df, train_frac, val_frac, seed):
    """
    Bemis-Murcko scaffold split.
    Groups by scaffold, shuffles groups, assigns greedily.
    Returns (train_idx, val_idx, test_idx).
    """
    from random import Random
    random = Random(seed)
    
    # Group by scaffold
    scaffolds = {}
    for i, smi in enumerate(df["smiles"].values):
        scf = get_scaffold(smi)
        scaffolds.setdefault(scf, []).append(i)
    
    groups = list(scaffolds.values())
    random.shuffle(groups)
    
    n = len(df)
    train_target = int(np.floor(train_frac * n))
    val_target = int(np.floor(val_frac * n))
    
    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) < train_target:
            train_idx.extend(group)
        elif len(val_idx) < val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
    
    return (np.array(train_idx, dtype=np.int64),
            np.array(val_idx, dtype=np.int64),
            np.array(test_idx, dtype=np.int64))


def random_split_df(df, train_frac, val_frac, seed):
    """
    Stratified random split (by protein target).
    Shuffles rows with seed, assigns in order.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    
    n = len(df)
    train_end = int(np.floor(train_frac * n))
    val_end = int(np.floor((train_frac + val_frac) * n))
    
    return (indices[:train_end], indices[train_end:val_end], indices[val_end:])


def protein_id_from_seq(seq):
    """Derive a stable protein ID from its sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def write_split(train_idx, val_idx, test_idx, df, split_dir):
    """Write train/val/test CSV files and diagnostics."""
    split_dir.mkdir(parents=True, exist_ok=True)
    
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        subset = df.iloc[idx][["smiles", "protein_seq", "pkd"]]
        subset.to_csv(split_dir / f"{name}.csv", index=False)
        log.info(f"    {name}: {len(subset):,} compounds, "
                 f"pKd {subset['pkd'].mean():.2f}±{subset['pkd'].std():.2f}")
    
    # Diagnostics
    scf_col = df["smiles"].apply(get_scaffold)
    n_scaffolds = scf_col.nunique()
    largest_scaffold_frac = scf_col.value_counts().max() / len(df)
    
    test_smiles = set(df.iloc[test_idx]["smiles"])
    train_smiles = set(df.iloc[train_idx]["smiles"])
    
    # Check if any test SMILES appear in train
    leaks = len(test_smiles & train_smiles)
    
    diag = {
        "n_total": len(df),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_scaffolds": n_scaffolds,
        "largest_scaffold_fraction": round(largest_scaffold_frac, 4),
        "test_smiles_in_train": leaks,
        "train_pkd_mean": round(df.iloc[train_idx]["pkd"].mean(), 3),
        "val_pkd_mean": round(df.iloc[val_idx]["pkd"].mean(), 3),
        "test_pkd_mean": round(df.iloc[test_idx]["pkd"].mean(), 3),
    }
    pd.DataFrame([diag]).to_csv(split_dir / "diagnostics.csv", index=False)
    return diag


# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT 1: Scaffold split
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("SPLIT 1: Bemis-Murcko Scaffold (seed 42)")
log.info("=" * 70)

t0 = time.time()
train_scf, val_scf, test_scf = scaffold_split_df(df, CONFIG["train_frac"], CONFIG["val_frac"], 42)
diag_scf = write_split(train_scf, val_scf, test_scf, df, EXP_DIR / "split_scaffold_42")
log.info(f"  Scaffold split done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT 2: Sequence-identity split (MMseqs2)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("SPLIT 2: Sequence-Identity (MMseqs2, 40%)")
log.info("=" * 70)

t0 = time.time()

# Write FASTA file of all protein sequences
fasta_path = EXP_DIR / "proteins.fasta"
prot_ids = []
with open(fasta_path, "w") as f:
    for i, seq in enumerate(df["protein_seq"].values):
        pid = protein_id_from_seq(seq)
        prot_ids.append(pid)
        f.write(f">{pid}\n{seq}\n")

log.info(f"  Wrote {len(prot_ids):,} sequences to {fasta_path}")

# Run MMseqs2 easy-cluster
mmseqs_db = EXP_DIR / "mmseqs_db"
cluster_tsv = EXP_DIR / "mmseqs_clusters.tsv"

# Check if MMseqs2 is available
mmseqs_bin = "mmseqs"
try:
    subprocess.run([mmseqs_bin, "version"], capture_output=True, timeout=5, check=True)
    mmseqs_available = True
except (FileNotFoundError, subprocess.TimeoutExpired):
    # Try conda env path
    import glob
    candidates = glob.glob("/c/Users/ss864/AppData/Local/miniconda3/envs/drug_discovery/bin/mmseqs*")
    if candidates:
        mmseqs_bin = candidates[0]
        mmseqs_available = True
    else:
        mmseqs_available = False

if mmseqs_available:
    log.info(f"  Running MMseqs2 easy-cluster (--min-seq-id 0.40)...")
    result = subprocess.run([
        mmseqs_bin, "easy-cluster",
        str(fasta_path),
        str(EXP_DIR / "mmseqs_result"),
        str(EXP_DIR / "mmseqs_tmp"),
        "--min-seq-id", "0.40",
        "-c", "0.8",  # coverage
        "--cov-mode", "0",  # bidirectional
    ], capture_output=True, text=True, timeout=600)
    
    if result.returncode == 0:
        # Parse clusters
        cluster_file = EXP_DIR / "mmseqs_result_cluster.tsv"
        clusters = {}
        with open(cluster_file) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                clusters.setdefault(rep, []).append(member)
        
        # Assign whole clusters to train/val/test
        cluster_list = list(clusters.values())
        rng = np.random.default_rng(42)
        rng.shuffle(cluster_list)
        
        n = len(df)
        train_target = int(np.floor(CONFIG["train_frac"] * n))
        val_target = int(np.floor(CONFIG["val_frac"] * n))
        
        train_seqid, val_seqid, test_seqid = [], [], []
        prot_to_idx = {pid: i for i, pid in enumerate(prot_ids)}
        
        # Create reverse mapping: protein_id -> all row indices
        pid_to_rows = {}
        for i, pid in enumerate(prot_ids):
            pid_to_rows.setdefault(pid, []).append(i)
        
        for cluster in cluster_list:
            # Gather all row indices for this cluster
            cluster_rows = []
            for pid in cluster:
                if pid in pid_to_rows:
                    cluster_rows.extend(pid_to_rows[pid])
            
            if len(train_seqid) < train_target:
                train_seqid.extend(cluster_rows)
            elif len(val_seqid) < val_target:
                val_seqid.extend(cluster_rows)
            else:
                test_seqid.extend(cluster_rows)
        
        train_seqid = np.array(train_seqid, dtype=np.int64)
        val_seqid = np.array(val_seqid, dtype=np.int64)
        test_seqid = np.array(test_seqid, dtype=np.int64)
        
        diag_seqid = write_split(train_seqid, val_seqid, test_seqid, df, 
                                 EXP_DIR / "split_seqid_42")
        diag_seqid["n_clusters"] = len(cluster_list)
        diag_seqid["max_cluster_size"] = max(len(c) for c in cluster_list)
        
        log.info(f"  MMseqs2 split done in {time.time()-t0:.1f}s")
        log.info(f"  {len(cluster_list):,} clusters at 40% identity")
    else:
        log.error(f"  MMseqs2 failed: {result.stderr[:500]}")
        mmseqs_available = False

if not mmseqs_available:
    log.warning("  MMseqs2 not available. Falling back to protein-level random split.")
    log.warning("  Install: conda install -c bioconda mmseqs2")
    
    # Fallback: group by protein sequence, randomly assign clusters
    pid_to_rows = {}
    for i, pid in enumerate(prot_ids):
        pid_to_rows.setdefault(pid, []).append(i)
    
    clusters = list(pid_to_rows.values())
    rng = np.random.default_rng(42)
    rng.shuffle(clusters)
    
    n = len(df)
    train_target = int(np.floor(CONFIG["train_frac"] * n))
    val_target = int(np.floor(CONFIG["val_frac"] * n))
    
    train_seqid, val_seqid, test_seqid = [], [], []
    for cluster in clusters:
        if len(train_seqid) < train_target:
            train_seqid.extend(cluster)
        elif len(val_seqid) < val_target:
            val_seqid.extend(cluster)
        else:
            test_seqid.extend(cluster)
    
    train_seqid = np.array(train_seqid, dtype=np.int64)
    val_seqid = np.array(val_seqid, dtype=np.int64)
    test_seqid = np.array(test_seqid, dtype=np.int64)
    
    diag_seqid = write_split(train_seqid, val_seqid, test_seqid, df, 
                             EXP_DIR / "split_seqid_42")
    diag_seqid["n_clusters"] = len(clusters)
    diag_seqid["note"] = "Protein-level clustering (MMseqs2 unavailable)"
    log.info(f"  Fallback split done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT 3: Random split
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("SPLIT 3: Random (seed 42)")
log.info("=" * 70)

t0 = time.time()
train_rnd, val_rnd, test_rnd = random_split_df(df, CONFIG["train_frac"], CONFIG["val_frac"], 42)
diag_rnd = write_split(train_rnd, val_rnd, test_rnd, df, EXP_DIR / "split_random_42")
log.info(f"  Random split done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("SPLIT COMPARISON")
log.info("=" * 70)

for name, diag in [("Scaffold", diag_scf), ("SeqID", diag_seqid), ("Random", diag_rnd)]:
    log.info(f"\n{name}:")
    log.info(f"  Train={diag['n_train']:,}  Val={diag['n_val']:,}  Test={diag['n_test']:,}")
    log.info(f"  Train pKd={diag['train_pkd_mean']:.2f}  "
             f"Val pKd={diag['val_pkd_mean']:.2f}  "
             f"Test pKd={diag['test_pkd_mean']:.2f}")
    if 'n_clusters' in diag:
        log.info(f"  Clusters={diag['n_clusters']:,}  Max cluster={diag.get('max_cluster_size', 'N/A')}")

# Save split comparison LaTeX table
latex = []
latex.append(r"\begin{table}[ht]")
latex.append(r"\centering")
latex.append(r"\caption{Three-way split diagnostics.}")
latex.append(r"\label{tab:splits}")
latex.append(r"\begin{tabular}{lrrrrr}")
latex.append(r"\toprule")
latex.append(r"Split & Train & Val & Test & Train pKd & Test pKd \\")
latex.append(r"\midrule")
for name, diag in [("Scaffold", diag_scf), ("Sequence-ID", diag_seqid), ("Random", diag_rnd)]:
    latex.append(f"{name} & {diag['n_train']:,} & {diag['n_val']:,} & {diag['n_test']:,} & "
                 f"{diag['train_pkd_mean']:.2f} & {diag['test_pkd_mean']:.2f} \\\\")
latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

with open(EXP_DIR / "split_comparison.tex", "w") as f:
    f.write("\n".join(latex))

log.info(f"\n  Split comparison saved to split_comparison.tex")
log.info(f"\nE103 COMPLETE")
