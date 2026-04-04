"""
Global configuration for the binding affinity benchmark.
All experiments must use these settings to ensure apples-to-apples comparison.
"""

# ── Datasets ──────────────────────────────────────────────────────────────────
# Dataset 1: BindingDB_filtered (original benchmark)
DATASET_NAME   = "BALM/BALM-benchmark"
DATASET_CONFIG = "BindingDB_filtered"
SMILES_COL  = "Drug"
PROTEIN_COL = "Target"
LABEL_COL   = "Y"

# Dataset 2: LeakyPDB (PDBBind-derived)
LEAKYPDB_NAME   = "BALM/BALM-benchmark"
LEAKYPDB_CONFIG = "LeakyPDB"
# LeakyPDB is a single HF 'train' split with a 'new_split' column (train/val/test)

# ── Multi-seed evaluation ─────────────────────────────────────────────────────
SPLIT_SEED  = 42                      # default / primary seed
MULTI_SEEDS = [42, 123, 456]          # used for stability analysis

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 128
MAX_EPOCHS    = 100
PATIENCE      = 20          # early stopping on val RMSE
LEARNING_RATE = 5e-4 
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
DEVICE        = "cuda"      # "cpu" fallback handled in trainer

# ── Sequence truncation ────────────────────────────────────────────────────────
MAX_SMILES_LEN  = 512
MAX_PROTEIN_LEN = 1200

# ── Paths ─────────────────────────────────────────────────────────────────────
import pathlib
ROOT        = pathlib.Path(__file__).parent.parent
CACHE_DIR   = ROOT / "cache"
SPLIT_DIR   = ROOT / "cache" / "splits"
RESULTS_DIR = ROOT / "results"
DIARY_PATH  = ROOT / "diary" / "results_diary.csv"

# ── Metrics reported ──────────────────────────────────────────────────────────
METRICS = ["rmse", "pearson_r", "spearman_r"]
