#!/usr/bin/env bash
# Creates a fresh conda environment for the binding affinity benchmark.
# Run once: bash setup_env.sh

set -e

ENV_NAME="drug_discovery"
PYTHON_VERSION="3.11"

echo "=== Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION) ==="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "=== Activating environment ==="
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== Installing PyTorch (CUDA 12.1) ==="
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

echo "=== Installing RDKit ==="
conda install -y -c conda-forge rdkit

echo "=== Installing pip packages ==="
pip install \
    datasets \
    transformers \
    accelerate \
    tokenizers \
    scikit-learn \
    scipy \
    pandas \
    numpy \
    xgboost \
    lightgbm \
    tqdm

echo ""
echo "=== Done! ==="
echo "Activate with:  conda activate $ENV_NAME"
echo "Run experiments: python run_experiment.py --list"
echo "Run all baselines: python run_experiment.py --group baselines"
