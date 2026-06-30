"""Queue missing key experiments on bullitt for the revised paper.
These fill the 2x2 factorial: architecture x protein_representation"""
import paramiko, time

PW='@@Gujrattt123'
def ssh(cmd, timeout=900):
    for attempt in range(5):
        try:
            c=paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect('aicoe.snu.in',username='rajeev',password=PW,timeout=20)
            _,stdout,stderr=c.exec_command(cmd,timeout=timeout)
            out=stdout.read().decode(errors='replace')
            err=stderr.read().decode(errors='replace')
            c.close()
            return out,err
        except Exception as e:
            print(f'  retry {attempt+1}/5: {e}')
            time.sleep(10)
    return '','ALL RETRIES FAILED'

EXPERIMENTS = [
    # 2x2: RF x protein_repr (ecfp4 + aac/esm2)
    'rf_ecfp4_aac',              # exists (1.007) — baseline
    'rf_ecfp4_esm2_8M',          # RF + ESM-2 8M (does ESM-2 help trees?)
    'rf_ecfp4_esm2_35M',         # RF + ESM-2 35M
    # 2x2: MLP x protein_repr
    'mlp_ecfp4_aac',             # exists (1.329) — baseline
    'mlp_ecfp4_esm2_8M',         # MLP + ESM-2
    'mlp_ecfp4_esm2_35M',        # MLP + ESM-2 35M
    # ChemCross on protein split (seed 99)
    'mlp_chemberta_esm2_8M',     # MLP + good reprs — competitive deep baseline
]

def check_existing():
    out,_ = ssh('cd ~/bica && grep -oP "^[^,]+" diary/results_diary.csv | sort -u')
    existing = set(out.strip().split('\n'))
    return existing

print('Checking existing experiments...')
existing = check_existing()
print(f'  {len(existing)} unique experiments in diary')

# Filter to missing
to_run = []
for exp in EXPERIMENTS:
    if exp not in existing:
        to_run.append(exp)
        print(f'  QUEUE: {exp}')
    else:
        print(f'  SKIP: {exp} (exists)')

if not to_run:
    print('\nAll experiments already exist — nothing to run.')
else:
    print(f'\nQueuing {len(to_run)} experiments...')
    for exp in to_run:
        print(f'\n=== {exp} ===')
        out,err = ssh(f'cd ~/bica && source bica_env/bin/activate && python run_experiment.py --exp {exp} 2>&1', timeout=900)
        for line in out.strip().split('\n'):
            if any(k in line.lower() for k in ['rmse=','test ','logged','error']):
                print(f'  {line.strip()[:120]}')
        if err:
            print(f'  ERR: {err.strip()[:200]}')
        time.sleep(3)

    # Also run RF on protein split
    print('\n=== rf_ecfp4_aac --seed 99 (protein split) ===')
    out,err = ssh('cd ~/bica && source bica_env/bin/activate && python run_experiment.py --exp rf_ecfp4_aac --seed 99 2>&1', timeout=900)
    for line in out.strip().split('\n'):
        if any(k in line.lower() for k in ['rmse=','test ','logged']):
            print(f'  {line.strip()[:120]}')

    # MLP on protein split
    print('\n=== mlp_ecfp4_aac --seed 99 (protein split) ===')
    out,err = ssh('cd ~/bica && source bica_env/bin/activate && python run_experiment.py --exp mlp_ecfp4_aac --seed 99 2>&1', timeout=900)
    for line in out.strip().split('\n'):
        if any(k in line.lower() for k in ['rmse=','test ','logged']):
            print(f'  {line.strip()[:120]}')

    print('\nDONE.')
