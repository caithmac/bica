import paramiko, time
PW='@@Gujrattt123'
c=paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('aicoe.snu.in',username='rajeev',password=PW,timeout=15)

def run(cmd, timeout=600):
    _,stdout,stderr=c.exec_command(cmd,timeout=timeout)
    return stdout.read().decode(errors='replace'),stderr.read().decode(errors='replace')

# Check what exists
out,_=run("cd ~/bica && grep -oP '^[^,]+' diary/results_diary.csv | sort -u")
existing=set(out.strip().split('\n'))
print(f'Existing: {len(existing)} experiments')

# Missing key 2x2 experiments
targets=[
    'rf_ecfp4_aac',        # RF baseline (should exist)
    'mlp_ecfp4_aac',       # MLP baseline
    'mlp_ecfp4_esm2_8M',   # MLP + ESM2
    'mlp_ecfp4_esm2_35M',  # MLP + larger ESM2
]
for exp in targets:
    if exp in existing:
        print(f'SKIP {exp} (exists)')
        continue
    print(f'RUN {exp}...')
    out,err=run(f'cd ~/bica && source bica_env/bin/activate && python run_experiment.py --exp {exp} 2>&1',timeout=600)
    for line in (out+err).strip().split('\n'):
        if 'logged' in line.lower() or 'error' in line.lower():
            print(f'  {line.strip()[:150]}')
    time.sleep(3)

# MLP + ChemBERTa combo (best deep representation)
for exp in ['mlp_chemberta_esm2_8M','mlp_chemberta_aac']:
    if exp in existing:
        print(f'SKIP {exp} (exists)')
        continue
    print(f'RUN {exp}...')
    out,err=run(f'cd ~/bica && source bica_env/bin/activate && python run_experiment.py --exp {exp} 2>&1',timeout=600)
    for line in (out+err).strip().split('\n'):
        if 'logged' in line.lower() or 'error' in line.lower():
            print(f'  {line.strip()[:150]}')
    time.sleep(3)

# Protein split experiments
for exp in ['rf_ecfp4_aac','mlp_ecfp4_aac','mlp_chemberta_esm2_8M']:
    lbl=f'{exp}__seed99'
    if lbl in existing:
        print(f'SKIP {lbl} (exists)')
        continue
    print(f'RUN {exp} --seed 99...')
    out,err=run(f'cd ~/bica && source bica_env/bin/activate && python run_experiment.py --exp {exp} --seed 99 2>&1',timeout=600)
    for line in (out+err).strip().split('\n'):
        if 'logged' in line.lower() or 'error' in line.lower():
            print(f'  {line.strip()[:150]}')
    time.sleep(3)

c.close()
print('\nDone.')
