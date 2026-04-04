"""
Dataset loading and scaffold split.

Scaffold split strategy (Bemis-Murcko):
  - Extract scaffold for every ligand SMILES
  - Group by scaffold
  - Sort groups by size (largest first) for deterministic assignment
  - Walk groups and assign to train/val/test by cumulative fraction
  - Molecules with unparseable SMILES get a unique scaffold key = their index
"""

import hashlib
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from harness.config import (
    DATASET_NAME, DATASET_CONFIG,
    LEAKYPDB_NAME, LEAKYPDB_CONFIG,
    SMILES_COL, PROTEIN_COL, LABEL_COL,
    TRAIN_FRAC, VAL_FRAC, SPLIT_SEED,
    CACHE_DIR, SPLIT_DIR,
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_scaffold(smiles: str) -> str:
    """Return canonical Bemis-Murcko scaffold SMILES, or a fallback hash."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else "__no_scaffold__"
    except Exception:
        return f"__invalid__{hashlib.md5(smiles.encode()).hexdigest()}"


# def _scaffold_split(df: pd.DataFrame, train_frac: float, val_frac: float,
#                     seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Returns three arrays of integer indices into df: train, val, test.
#     """
#     rng = np.random.default_rng(seed)

#     scaffolds: dict[str, list[int]] = {}
#     for idx, smi in enumerate(df[SMILES_COL].tolist()):
#         s = _get_scaffold(smi)
#         scaffolds.setdefault(s, []).append(idx)

#     # Sort by scaffold size descending, shuffle ties with rng
#     groups = list(scaffolds.values())
#     rng.shuffle(groups)
#     groups.sort(key=lambda g: len(g), reverse=True)

#     n = len(df)
#     train_cutoff = int(np.floor(train_frac * n))
#     val_cutoff   = int(np.floor((train_frac + val_frac) * n))

#     train_idx, val_idx, test_idx = [], [], []
#     for group in groups:
#         if len(train_idx) < train_cutoff:
#             train_idx.extend(group)
#         elif len(train_idx) + len(val_idx) < val_cutoff:
#             val_idx.extend(group)
#         else:
#             test_idx.extend(group)

#     return (np.array(train_idx, dtype=np.int64),
#             np.array(val_idx,   dtype=np.int64),
#             np.array(test_idx,  dtype=np.int64))


def _scaffold_split(df: pd.DataFrame, train_frac: float, val_frac: float,
                    seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scaffold split as used in the paper:
    - Shuffle scaffold groups (not sorted by size)
    - Assign entire groups to train/val/test based on cumulative size
    - Skip invalid SMILES (drop those rows)
    """
    from random import Random
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    random = Random(seed)

    # Group indices by scaffold (skip invalid SMILES)
    scaffolds = {}
    for idx, smi in enumerate(df[SMILES_COL].tolist()):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds.setdefault(scaffold, []).append(idx)
        except:
            continue

    # Shuffle scaffold groups (random order, no size sorting)
    groups = list(scaffolds.values())
    random.shuffle(groups)

    n = len(df)
    train_target = int(np.floor(train_frac * n))
    val_target   = int(np.floor(val_frac * n))
    # Ensure targets don't exceed n
    if train_target + val_target > n:
        val_target = n - train_target

    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) < train_target:
            train_idx.extend(group)
        elif len(val_idx) < val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    # Safety: if test_idx is empty, move the last group from val to test
    if len(test_idx) == 0 and len(val_idx) > 0:
        # Move the smallest group from val to test (or the last added group)
        moved = val_idx[-len(groups[-1]):] if groups else val_idx
        test_idx.extend(moved)
        val_idx = val_idx[:-len(moved)] if len(moved) < len(val_idx) else []

    return (np.array(train_idx, dtype=np.int64),
            np.array(val_idx,   dtype=np.int64),
            np.array(test_idx,  dtype=np.int64))


# ── Public API ────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Download (or load from cache) and return the raw DataFrame."""
    cache_file = CACHE_DIR / "bindingdb_raw.pkl"
    if cache_file.exists():
        print(f"[data] Loading raw data from cache: {cache_file}")
        return pd.read_pickle(cache_file)

    print("[data] Downloading dataset from HuggingFace …")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    # The dataset may only have a 'train' split
    if "train" in ds:
        df = ds["train"].to_pandas()
    else:
        split_key = list(ds.keys())[0]
        df = ds[split_key].to_pandas()

    df = df[[SMILES_COL, PROTEIN_COL, LABEL_COL]].dropna().reset_index(drop=True)
    df.to_pickle(cache_file)
    print(f"[data] Cached {len(df):,} rows → {cache_file}")
    return df


def get_splits(df: pd.DataFrame | None = None):
    """
    Returns (train_df, val_df, test_df).
    Splits are cached so every model uses identical splits.
    """
    split_file = SPLIT_DIR / f"scaffold_seed{SPLIT_SEED}.pkl"

    if split_file.exists():
        print(f"[data] Loading cached splits from {split_file}")
        with open(split_file, "rb") as f:
            train_idx, val_idx, test_idx = pickle.load(f)
    else:
        if df is None:
            df = load_raw()
        print("[data] Computing scaffold splits …")
        train_idx, val_idx, test_idx = _scaffold_split(df, TRAIN_FRAC, VAL_FRAC, SPLIT_SEED)
        with open(split_file, "wb") as f:
            pickle.dump((train_idx, val_idx, test_idx), f)
        print(f"[data] Splits → train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    if df is None:
        df = load_raw()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx  ].reset_index(drop=True)
    test_df  = df.iloc[test_idx ].reset_index(drop=True)
    return train_df, val_df, test_df


def describe_splits():
    """Print a short summary of each split."""
    df = load_raw()
    train_df, val_df, test_df = get_splits(df)
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        y = split[LABEL_COL]
        print(f"{name:5s}  n={len(split):6,}  pKd: mean={y.mean():.2f} "
              f"std={y.std():.2f}  min={y.min():.2f}  max={y.max():.2f}")


# ── LeakyPDB (PDBBind-derived) ────────────────────────────────────────────────

def load_leakypdb_raw() -> pd.DataFrame:
    """
    Load the full LeakyPDB dataset (single 'train' HF split) as a DataFrame.
    The HF dataset has a 'new_split' column with values 'train'/'val'/'test'.
    Cached to cache/leakypdb_raw.pkl.
    """
    raw_cache = CACHE_DIR / "leakypdb_raw.pkl"

    if raw_cache.exists():
        print("[data] Loading LeakyPDB from cache …")
        return pd.read_pickle(raw_cache)

    print("[data] Downloading LeakyPDB from HuggingFace …")
    ds = load_dataset(LEAKYPDB_NAME, LEAKYPDB_CONFIG)
    df = ds["train"].to_pandas()
    # Keep only columns we need plus the split indicator
    df = df[["Drug", "Target", "Y", "new_split"]].dropna().reset_index(drop=True)
    # Rename to match harness column names
    df = df.rename(columns={"Drug": SMILES_COL, "Target": PROTEIN_COL, "Y": LABEL_COL})
    df.to_pickle(raw_cache)
    print(f"[data] LeakyPDB downloaded: {len(df):,} rows")
    return df


def get_leakypdb_splits() -> tuple:
    """
    Returns (train_df, val_df, test_df) for LeakyPDB using the
    dataset's built-in new_split column ('train'/'val'/'test').
    Cached to cache/splits/leakypdb_split.pkl.
    """
    split_file = SPLIT_DIR / "leakypdb_split.pkl"

    if split_file.exists():
        print(f"[data] Loading cached LeakyPDB splits from {split_file}")
        with open(split_file, "rb") as f:
            train_df, val_df, test_df = pickle.load(f)
        return train_df, val_df, test_df

    df = load_leakypdb_raw()

    train_df = df[df["new_split"] == "train"].drop(columns=["new_split"]).reset_index(drop=True)
    val_df   = df[df["new_split"] == "val"].drop(columns=["new_split"]).reset_index(drop=True)
    test_df  = df[df["new_split"] == "test"].drop(columns=["new_split"]).reset_index(drop=True)

    with open(split_file, "wb") as f:
        pickle.dump((train_df, val_df, test_df), f)
    print(f"[data] LeakyPDB splits → train={len(train_df):,}  "
          f"val={len(val_df):,}  test={len(test_df):,}")
    return train_df, val_df, test_df


# ── Multi-seed splits (BindingDB only) ───────────────────────────────────────

def get_splits_for_seed(seed: int, df: pd.DataFrame | None = None):
    """
    Like get_splits() but for an arbitrary seed.
    Cached to cache/splits/scaffold_seed{seed}.pkl.
    """
    split_file = SPLIT_DIR / f"scaffold_seed{seed}.pkl"

    if split_file.exists():
        with open(split_file, "rb") as f:
            train_idx, val_idx, test_idx = pickle.load(f)
    else:
        if df is None:
            df = load_raw()
        print(f"[data] Computing scaffold splits for seed={seed} …")
        train_idx, val_idx, test_idx = _scaffold_split(df, TRAIN_FRAC, VAL_FRAC, seed)
        with open(split_file, "wb") as f:
            pickle.dump((train_idx, val_idx, test_idx), f)

    if df is None:
        df = load_raw()

    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx  ].reset_index(drop=True),
            df.iloc[test_idx ].reset_index(drop=True))
