"""
Master runner — runs every planned experiment, then analysis and bootstrap CIs.

Resumable: already-logged experiment_ids are skipped (checks diary CSV).
All results append to the same diary/results_diary.csv.

Usage:
    python run_all.py               # everything
    python run_all.py --dry-run     # just print what would run
    python run_all.py --only bindingdb_multiseed   # one phase only
    python run_all.py --skip_slow   # skip distmat/GNN/seq (hours each)

Phases (run in order):
  1. bindingdb_base        — original 44 experiments (seed=42)
  2. bindingdb_ablations   — missing 2x2 ablations (xgb_ecfp4_esm2_8M etc.)
  3. bindingdb_multiseed   — fast experiments × seeds 123 and 456
  4. leakypdb_base         — core experiments on LeakyPDB dataset
  5. bootstrap_ci          — compute 95% CIs on all saved predictions
  6. analysis              — re-run analyze_results.py to update FINDINGS.md
"""

import os, sys, subprocess, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from pathlib import Path

DIARY  = Path("diary/results_diary.csv")
PYTHON = sys.executable   # same interpreter / conda env


# def already_done(exp_id: str) -> bool:
#     """True if this experiment_id is already in the diary."""
    
#     if not DIARY.exists():
#         return False
#     try:
#         df = pd.read_csv(DIARY, usecols=["experiment_id"])
#         return exp_id in df["experiment_id"].values
#     except Exception:
#         return False


from pathlib import Path

PRED_DIR = Path("cache/predictions")

def already_done(exp_id: str) -> bool:
    """True if this experiment_id is already in the diary AND has predictions."""
    
    if not DIARY.exists():
        return False

    try:
        df = pd.read_csv(DIARY, usecols=["experiment_id"])
        in_diary = exp_id in df["experiment_id"].values

        # 🔑 Check predictions
        pred_file = PRED_DIR / f"{exp_id}.npz"
        has_preds = pred_file.exists()

        return in_diary and has_preds

    except Exception:
        return False


def run_exp(exp_name: str, dataset: str = "bindingdb", seed: int = 42,
            dry_run: bool = False) -> bool:
    """
    Run a single experiment via run_experiment.py.
    Returns True on success, False on failure.
    Skips if already in diary.
    """
    # Compute the experiment_id that would be logged
    if dataset == "leakypdb":
        exp_id = f"{exp_name}__leakypdb"
    elif seed != 42:
        exp_id = f"{exp_name}__seed{seed}"
    else:
        exp_id = exp_name

    if already_done(exp_id):
        print(f"  [skip] {exp_id} — already in diary")
        return True

    cmd = [PYTHON, "run_experiment.py",
           "--exp", exp_name,
           "--dataset", dataset,
           "--seed", str(seed)]

    print(f"\n{'─'*60}")
    print(f"  Running: {exp_id}")
    print(f"  Command: {' '.join(cmd)}")
    if dry_run:
        return True

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [ERROR] {exp_id} failed with return code {result.returncode}")
        return False
    return True


def run_phase(phase_name: str, experiments: list, dry_run: bool = False):
    print(f"\n{'='*60}")
    print(f"  PHASE: {phase_name}  ({len(experiments)} experiments)")
    print(f"{'='*60}")
    n_ok, n_fail, n_skip = 0, 0, 0
    for kwargs in experiments:
        ok = run_exp(dry_run=dry_run, **kwargs)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
    print(f"\n  Phase done: {n_ok} ok  {n_fail} failed")
    return n_fail == 0


# ─────────────────────────────────────────────────────────────────────────────
# Experiment lists
# ─────────────────────────────────────────────────────────────────────────────

# All base experiments (already run; will be skipped if diary has them)
BASE_EXPERIMENTS = [
    # baselines
    "ridge_ecfp4_aac", "ridge_ecfp4_dipeptide", "ridge_maccs_aac", "ridge_rdkit_aac",
    # trees
    "rf_ecfp4_aac", "rf_ecfp4_dipeptide", "xgb_ecfp4_aac", "xgb_ecfp6_kmer3", "lgbm_ecfp4_aac",
    # mlp
    "mlp_shallow_ecfp4_aac", "mlp_medium_ecfp4_dipeptide", "mlp_deep_ecfp6_kmer3",
    "mlp_wide_maccs_dipeptide",
    # pretrained mlp
    "mlp_chemberta_aac", "mlp_chemberta_dipeptide", "mlp_ecfp4_esm2_8M",
    "mlp_chemberta_esm2_8M", "mlp_chemberta_esm2_35M", "xgb_chemberta_esm2_8M",
    # ablations (new)
    "xgb_ecfp4_esm2_8M", "xgb_chemberta_aac",
    # prot_electra
    "mlp_ecfp4_prot_electra", "mlp_chemberta_prot_electra", "xgb_chemberta_prot_electra",
    # transformer flat
    "transformer_ecfp4_aac", "transformer_chemberta_esm2_8M", "transformer_chemberta_esm2_35M",
    # cnn
    "cnn_smiles_onehot_aac",
    # seq models
    "lstm_smiles_char_protein_char", "lstm_smiles_atom_protein_char",
    "lstm_smiles_bpe512_protein_bpe512", "lstm_smiles_bpe1000_protein_bpe1000",
    "lstm_smiles_atom_protein_wordpiece512",
    "transformer_seq_smiles_char_protein_char", "transformer_seq_smiles_atom_protein_char",
    "transformer_seq_smiles_bpe512_protein_bpe512", "transformer_seq_smiles_bpe1000_protein_bpe1000",
    "transformer_seq_smiles_atom_protein_wordpiece512",
    # distmat
    "distmat_cnn_aac", "distmat_cnn_esm2_8M",
    # graph models
    "gcn_ecfp_aac", "gcn_ecfp_esm2_8M", "gat_ecfp_aac", "gat_ecfp_esm2_8M",
    # bica
    "bica_ecfp4_aac", "bica_chemberta_esm2_8M", "bica_chemberta_prot_electra",
]

# Fast experiments only — used for multi-seed stability analysis
# Excludes: seq models (>5min each), distmat (>10min), GNN (>1min/epoch × many epochs)
FAST_EXPERIMENTS = [
    "ridge_ecfp4_aac", "ridge_maccs_aac", "ridge_rdkit_aac",
    "rf_ecfp4_aac", "xgb_ecfp4_aac", "lgbm_ecfp4_aac", "xgb_chemberta_esm2_8M",
    "xgb_ecfp4_esm2_8M", "xgb_chemberta_aac", "xgb_chemberta_prot_electra",
    "mlp_shallow_ecfp4_aac", "mlp_medium_ecfp4_dipeptide", "mlp_deep_ecfp6_kmer3",
    "mlp_ecfp4_esm2_8M", "mlp_chemberta_esm2_8M", "mlp_chemberta_prot_electra",
    "bica_ecfp4_aac", "bica_chemberta_esm2_8M",
]

# Phase 1 new experiments — ranking loss and GraphMAE node reconstruction
PHASE1_EXPERIMENTS = [
    # Pairwise ranking loss variants (MBP paper)
    "mlp_chemberta_esm2_8M_ranked",
    "bica_chemberta_esm2_8M_ranked",
    "gat_ecfp_esm2_8M_ranked",
    # GraphMAE node reconstruction variants
    "gcn_ecfp_esm2_8M_recon",
    "gat_ecfp_esm2_8M_recon",
]

# Targeted experiments — best ligand × best protein combos (from generate_targeted_experiments.py)
# Top ligand reprs: ecfp6_1024, chemberta_600, smiles_char
# Top protein reprs: kmer3_8000, prot_electra_256, protein_char
TARGETED_EXPERIMENTS = [
    # Ridge
    "ridge_ecfp6_1024_kmer3_8000", "ridge_ecfp6_1024_prot_electra_256", "ridge_ecfp6_1024_protein_char",
    "ridge_chemberta_600_kmer3_8000", "ridge_chemberta_600_prot_electra_256", "ridge_chemberta_600_protein_char",
    "ridge_smiles_char_kmer3_8000", "ridge_smiles_char_prot_electra_256", "ridge_smiles_char_protein_char",
    # XGBoost
    "xgb_ecfp6_1024_kmer3_8000", "xgb_ecfp6_1024_prot_electra_256", "xgb_ecfp6_1024_protein_char",
    "xgb_chemberta_600_kmer3_8000", "xgb_chemberta_600_prot_electra_256", "xgb_chemberta_600_protein_char",
    "xgb_smiles_char_kmer3_8000", "xgb_smiles_char_prot_electra_256", "xgb_smiles_char_protein_char",
    # MLP
    "mlp_ecfp6_1024_kmer3_8000", "mlp_ecfp6_1024_prot_electra_256", "mlp_ecfp6_1024_protein_char",
    "mlp_chemberta_600_kmer3_8000", "mlp_chemberta_600_prot_electra_256", "mlp_chemberta_600_protein_char",
    "mlp_smiles_char_kmer3_8000", "mlp_smiles_char_prot_electra_256", "mlp_smiles_char_protein_char",
    # Transformer
    "transformer_ecfp6_1024_kmer3_8000", "transformer_ecfp6_1024_prot_electra_256", "transformer_ecfp6_1024_protein_char",
    "transformer_chemberta_600_kmer3_8000", "transformer_chemberta_600_prot_electra_256", "transformer_chemberta_600_protein_char",
    "transformer_smiles_char_kmer3_8000", "transformer_smiles_char_prot_electra_256", "transformer_smiles_char_protein_char",
    # BiCA
    "bica_ecfp6_1024_kmer3_8000", "bica_ecfp6_1024_prot_electra_256", "bica_ecfp6_1024_protein_char",
    "bica_chemberta_600_kmer3_8000", "bica_chemberta_600_prot_electra_256", "bica_chemberta_600_protein_char",
    "bica_smiles_char_kmer3_8000", "bica_smiles_char_prot_electra_256", "bica_smiles_char_protein_char",
    # CNN
    "cnn_smiles_onehot_kmer3_8000", "cnn_smiles_onehot_prot_electra_256", "cnn_smiles_onehot_protein_char",
    # DistmatCNN
    "distmat_cnn_kmer3_8000", "distmat_cnn_prot_electra_256", "distmat_cnn_protein_char",
    # GCN
    "gcn_mol_graph_kmer3_8000", "gcn_mol_graph_prot_electra_256", "gcn_mol_graph_protein_char",
    # GAT
    "gat_mol_graph_kmer3_8000", "gat_mol_graph_prot_electra_256", "gat_mol_graph_protein_char",
    # LSTM new tokenizer combos
    "lstm_smiles_char_protein_bpe_1000", "lstm_smiles_atom_protein_bpe_1000",
    "lstm_smiles_bpe_1000_protein_bpe_1000", "lstm_smiles_bpe_1000_protein_char",
    # TransformerSeq new tokenizer combos
    "transformer_seq_smiles_char_protein_bpe_1000", "transformer_seq_smiles_atom_protein_bpe_1000",
    "transformer_seq_smiles_bpe_1000_protein_bpe_1000", "transformer_seq_smiles_bpe_1000_protein_char",
]

# Phase 2 new architectures: Graphormer, GLI, DSM, Mamba
PHASE2_EXPERIMENTS = [
    # Graphormer graph transformer (Ying et al., NeurIPS 2021)
    # "graphormer_mol_aac",
    # "graphormer_mol_esm2_8M",
    # "graphormer_mol_prot_electra",
    # "graphormer_mol_esm2_35M",
    # GLI gated global-local interaction
    # "gli_mol_aac",
    # "gli_mol_esm2_8M",
    # "gli_mol_prot_electra",
    # "gli_mol_esm2_35M",
    # DualBind DSM auxiliary loss
    "bica_chemberta_esm2_8M_dsm",
    "bica_chemberta_prot_electra_dsm",
    "mlp_chemberta_esm2_8M_dsm",
    # Mamba SSM sequence encoder
    "mamba_smiles_char_protein_char",
    "mamba_smiles_atom_protein_char",
    "mamba_smiles_bpe512_protein_bpe512",
    "mamba_smiles_bpe1000_protein_bpe1000",
]

# BiCA v2 — true-sequence ablation suite
BICA_V2_EXPERIMENTS = [
    "bica_v2_full",
    "bica_v2_meanpool",
    "bica_v2_singlelayer",
    "bica_v2_noffn",
    "bica_v2_noresidual",
    "bica_v2_p2l",
    "bica_v2_l2p",
    "concat_baseline",
    # ChemBERTa per-token ligand variants (best so far)
    "bica_v2_chemberta_tokens",
    "bica_v2_chemberta77M_tokens",
    "bica_v2_chemberta_tokens_esm2_150M",
    "bica_v2_cb77M_mtr_esm2_150M",
]

# Experiments to run on LeakyPDB
LEAKYPDB_EXPERIMENTS = [
    # Fast baselines and best models — gives cross-dataset comparison
    "ridge_ecfp4_aac",
    "rf_ecfp4_aac", "xgb_ecfp4_aac", "lgbm_ecfp4_aac",
    "xgb_chemberta_esm2_8M", "xgb_ecfp4_esm2_8M", "xgb_chemberta_aac",
    "mlp_ecfp4_esm2_8M", "mlp_chemberta_esm2_8M",
    "mlp_ecfp4_prot_electra", "mlp_chemberta_prot_electra",
    "bica_chemberta_esm2_8M", "bica_chemberta_prot_electra",
    "gcn_ecfp_esm2_8M", "gat_ecfp_esm2_8M",
]


def build_phase_list(only: str = None, skip_slow: bool = False):
    """Return list of (phase_name, experiments_list) to run."""
    phases = []

    base = [{"exp_name": e, "dataset": "bindingdb", "seed": 42}
            for e in BASE_EXPERIMENTS]
    phases.append(("bindingdb_base", base))

    phase1 = [{"exp_name": e, "dataset": "bindingdb", "seed": 42}
              for e in PHASE1_EXPERIMENTS]
    phases.append(("phase1_new_objectives", phase1))

    grid_file = Path("diary/full_grid_experiments.txt")
    if grid_file.exists():
        with open(grid_file) as f:
            grid_names = [line.strip() for line in f if line.strip()]
        grid_phase = [{"exp_name": e, "dataset": "bindingdb", "seed": 42}
                      for e in grid_names]
        phases.append(("full_grid", grid_phase))

    targeted = [{"exp_name": e, "dataset": "bindingdb", "seed": 42}
                for e in TARGETED_EXPERIMENTS]
    phases.append(("targeted_repr_combos", targeted))

    phase2 = [{"exp_name": e, "dataset": "bindingdb", "seed": 42}
              for e in PHASE2_EXPERIMENTS]
    phases.append(("phase2_new_architectures", phase2))

    bica_v2 = [{"exp_name": e, "dataset": "bindingdb", "seed": 42}
               for e in BICA_V2_EXPERIMENTS]
    phases.append(("bica_v2_ablations", bica_v2))

    multiseed = [{"exp_name": e, "dataset": "bindingdb", "seed": s}
                 for s in [123, 456]
                 for e in (FAST_EXPERIMENTS if skip_slow else FAST_EXPERIMENTS)]
    phases.append(("bindingdb_multiseed", multiseed))

    leaky = [{"exp_name": e, "dataset": "leakypdb", "seed": 42}
             for e in LEAKYPDB_EXPERIMENTS]
    phases.append(("leakypdb_base", leaky))

    if only:
        phases = [(n, e) for n, e in phases if n == only]

    return phases


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--only",       type=str, default=None,
                        help="Run only this phase name")
    parser.add_argument("--skip_slow",  action="store_true",
                        help="Skip seq models, distmat, GNN (hours each)")
    args = parser.parse_args()

    phases = build_phase_list(only=args.only, skip_slow=args.skip_slow)

    total_planned = sum(len(e) for _, e in phases)
    print(f"\n{'='*60}")
    print(f"  BINDING AFFINITY BENCHMARK — FULL RUN")
    print(f"  Planned: {total_planned} experiment runs across {len(phases)} phases")
    print(f"  Already-done experiments will be skipped automatically.")
    if args.dry_run:
        print("  DRY RUN — nothing will be executed")
    print(f"{'='*60}")

    all_ok = True
    for phase_name, experiments in phases:
        ok = run_phase(phase_name, experiments, dry_run=args.dry_run)
        if not ok:
            all_ok = False
            print(f"\n  [WARNING] Phase {phase_name} had failures — continuing anyway")

    # Bootstrap CIs
    if not args.dry_run and (args.only is None or args.only == "bootstrap_ci"):
        print(f"\n{'='*60}")
        print("  Running bootstrap CI …")
        subprocess.run([PYTHON, "bootstrap_ci.py", "--n_bootstrap", "1000"])

    # Re-generate analysis
    if not args.dry_run and (args.only is None or args.only == "analysis"):
        print(f"\n{'='*60}")
        print("  Updating FINDINGS.md …")
        subprocess.run([PYTHON, "analyze_results.py"])

    print(f"\n{'='*60}")
    print(f"  All phases complete. {'All OK.' if all_ok else 'Some failures — check above.'}")
    print(f"  Results: diary/results_diary.csv")
    print(f"  CIs:     diary/bootstrap_ci.csv")
    print(f"  Report:  diary/FINDINGS.md")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
