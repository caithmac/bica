"""Protein split experiments + value-weighting ablation — run via paramiko."""
import paramiko, time, sys, os

PW = '@@Gujrattt123'
HOST = 'aicoe.snu.in'
USER = 'rajeev'

def ssh(cmd, timeout=300, retries=3):
    for i in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, username=USER, password=PW, timeout=20)
            _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
            out = stdout.read().decode(errors='replace')
            err = stderr.read().decode(errors='replace')
            c.close()
            return out, err
        except Exception as e:
            print(f"  [retry {i+1}/{retries}: {e}]")
            time.sleep(5)
    return "", f"FAILED after {retries} retries"

# ── 1. Check what's already done ─────────────────────────────────────
out, _ = ssh("cd ~/bica && grep 'seed99' diary/results_diary.csv")
done = set()
for line in out.strip().split('\n'):
    if ',' in line:
        eid = line.split(',')[1]
        done.add(eid)
print(f"Already done: {done}")

# ── 2. Make sure protein split pickle exists ────────────────────────
out, err = ssh("cd ~/bica && source bica_env/bin/activate && python -c \""
    "import pickle,json; from harness.config import SPLIT_DIR; "
    "s=json.load(open(SPLIT_DIR/'protein_family_disjoint_seed42.json')); "
    "pickle.dump({'train_idx':s['train_idx'],'val_idx':s['val_idx'],'test_idx':s['test_idx']},"
    "open(SPLIT_DIR/'splits_seed99.pkl','wb')); print('Split ready')\"")
print(out.strip())

# ── 3. Run missing experiments ──────────────────────────────────────
TARGETS = [
    ('rf_ecfp4_aac',                     'rf_ecfp4_aac__seed99'),
    ('xgb_chemberta_5M_esm2_650M',       'xgb_chemberta_5M_esm2_650M__seed99'),
    ('bica_chemberta_5M_esmc_300M',      'bica_chemberta_5M_esmc_300M__seed99'),
    ('concat_baseline',                  'concat_baseline__seed99'),
    ('bica_v2_chemberta77M_tokens',      'bica_v2_chemberta77M_tokens__seed99'),
]

for exp, label in TARGETS:
    if label in done:
        print(f"SKIP {exp} (already done)")
        continue
    print(f"\n=== RUNNING {exp} --seed 99 ===")
    out, err = ssh(
        f"cd ~/bica && source bica_env/bin/activate && python run_experiment.py --exp {exp} --seed 99 2>&1",
        timeout=600
    )
    # Print last few lines
    for line in out.strip().split('\n'):
        if any(k in line.lower() for k in ['rmse=', 'test ', 'logged', 'error', 'traceback']):
            print(f"  {line.strip()}")
    if err:
        for line in err.strip().split('\n')[-4:]:
            print(f"  ERR: {line.strip()}")
    time.sleep(2)

# ── 4. Show final results ───────────────────────────────────────────
print("\n\n=== PROTEIN-FAMILY SPLIT RESULTS ===")
out, _ = ssh("cd ~/bica && grep 'seed99' diary/results_diary.csv | grep -v psichic")
for line in out.strip().split('\n'):
    parts = line.split(',')
    if len(parts) > 18:
        print(f"  {parts[1]:50s} RMSE={parts[17]}  R={parts[18]}  Spearman={parts[19]}")
print("\nDONE.")
