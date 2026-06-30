"""Wrapper to run run_experiment.py with line-buffered file logging."""
import sys, os
os.chdir('E:/BICA')

sys.argv = ['run_experiment.py', '--exp', 'mlp_ecfp4_esm2_8M_ft_k3']

log = open('E:/BICA/logs/mlp_ft_k3.log', 'w', buffering=1)
sys.stdout = log
sys.stderr = log

import run_experiment
run_experiment.main()
