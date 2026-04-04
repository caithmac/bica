@echo off
REM Creates a fresh conda environment for the binding affinity benchmark.
REM Run once from Anaconda Prompt: setup_env.bat

SET ENV_NAME=drug_discovery
SET PYTHON_VERSION=3.11

echo === Creating conda environment: %ENV_NAME% (Python %PYTHON_VERSION%) ===
call conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%

echo === Activating environment ===
call conda activate %ENV_NAME%

echo === Installing PyTorch (CUDA 12.1 for A4000) ===
call conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

echo === Installing RDKit ===
call conda install -y -c conda-forge rdkit

echo === Installing pip packages ===
pip install ^
    datasets ^
    transformers ^
    accelerate ^
    tokenizers ^
    scikit-learn ^
    scipy ^
    pandas ^
    numpy ^
    xgboost ^
    lightgbm ^
    tqdm

echo.
echo === Done! ===
echo Activate with:   conda activate %ENV_NAME%
echo Inspect data:    python inspect_data.py
echo List experiments: python run_experiment.py --list
echo Run baselines:   python run_experiment.py --group baselines
echo Run all:         python run_experiment.py --group all
pause
