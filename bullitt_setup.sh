#!/bin/bash
# Bullitt setup — run this ON bullitt (not locally)
# How to use:
#   1. scp bullitt_setup.sh rajeev@aicoe.snu.in:~/   (from another machine)
#   2. ssh rajeev@aicoe.snu.in
#   3. bash ~/bullitt_setup.sh
# Or just copy-paste each section into your SSH terminal.
set -e

echo "=== Setup bica project on bullitt ==="

# 1. Clone or copy project
# If you have git:
#   git clone <your-repo-url> ~/bica
# Otherwise, upload bica_upload.tar.gz from your local machine:
#   scp bica_upload.tar.gz rajeev@aicoe.snu.in:~/
#   tar xzf ~/bica_upload.tar.gz -C ~/bica/

cd ~/bica 2>/dev/null || { echo "ERROR: ~/bica not found — upload project first"; exit 1; }

# 2. Create venv
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch (CUDA 12.8)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install packages
echo "Installing packages..."
pip install transformers rdkit scikit-learn xgboost lightgbm captum fair-esm datasets scipy pandas

# 5. Install PyTorch Geometric
echo "Installing PyG..."
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu128.html
pip install torch-geometric

# 6. Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB')
import transformers, rdkit, datasets
print('All packages OK')
"

echo ""
echo "=== Setup complete ==="
echo "Run: source ~/bica/venv/bin/activate"
echo "Then: python run_protein_split.py"
echo "Then: python run_value_weight_ablation.py"
