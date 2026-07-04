"""Cross-dataset benchmark: RF + ECFP4 + AAC on all 7 BALM datasets.
Runs on bullitt via persistent SSH."""
import paramiko, time, json

PW='@@Gujrattt123'
DATASETS = ['BindingDB_filtered','HIF2A','LeakyPDB','MCL1','Mpro','SYK','USP7']

c=paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('aicoe.snu.in',username='rajeev',password=PW,timeout=15)

def run(cmd, timeout=600):
    _,stdout,stderr=c.exec_command(cmd,timeout=timeout)
    return stdout.read().decode(errors='replace'),stderr.read().decode(errors='replace')

results = {}
for ds in DATASETS:
    print(f'\n=== {ds} ===')
    out,err = run(f'cd ~/bica && source bica_env/bin/activate && python -c "'
        f'from datasets import load_dataset; '
        f'import pandas as pd; '
        f'import numpy as np; '
        f'from rdkit import Chem; '
        f'from rdkit.Chem.Scaffolds import MurckoScaffold; '
        f'import hashlib; '
        f'ds=load_dataset(\"BALM/BALM-benchmark\",\"{ds}\",split=\"train\"); '
        f'df=pd.DataFrame(ds); '
        f'print(f\"Rows: {{len(df)}}\"); '
        f'print(f\"Columns: {{list(df.columns)}}\"); '
        f'print(f\"Label range: {{df.iloc[:,-1].min():.2f}} - {{df.iloc[:,-1].max():.2f}}\"); '
        f'print(f\"Label mean: {{df.iloc[:,-1].mean():.2f}}\")" 2>&1', timeout=120)
    for line in (out+err).strip().split('\n')[-8:]:
        print(f'  {line.strip()[:150]}')

    # Quick train RF on scaffold split and report test RMSE
    out,err = run(f'cd ~/bica && source bica_env/bin/activate && python -c "'
        f'from datasets import load_dataset;'
        f'import pandas as pd;'
        f'import numpy as np;'
        f'from sklearn.ensemble import RandomForestRegressor;'
        f'from sklearn.metrics import mean_squared_error;'
        f'from rdkit import Chem;'
        f'from rdkit.Chem.Scaffolds import MurckoScaffold;'
        f'from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect;'
        f'import hashlib;'
        f'ds=load_dataset(\"BALM/BALM-benchmark\",\"{ds}\",split=\"train\");'
        f'df=pd.DataFrame(ds);'
        f'ligs=[Chem.MolFromSmiles(s) for s in df.iloc[:,0].tolist()];'
        f'X_lig=[list(GetMorganFingerprintAsBitVect(m,2,1024)) for m in ligs];'
        f'y=df.iloc[:,-1].values.astype(float);'
        f'# Scaffold split'
        f'scaffolds={{}};'
        f'for i,smi in enumerate(df.iloc[:,0].tolist()):'
        f'  try: mol=Chem.MolFromSmiles(smi); sc=MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else str(i)'
        f'  except: sc=str(i)'
        f'  scaffolds.setdefault(sc,[]).append(i)'
        f'sc_keys=list(scaffolds.keys()); np.random.RandomState(42).shuffle(sc_keys);'
        f'train_idx=[]; val_idx=[]; test_idx=[];'
        f'for sc in sc_keys:'
        f'  if len(train_idx)<0.7*len(df): train_idx.extend(scaffolds[sc])'
        f'  elif len(val_idx)<0.1*len(df): val_idx.extend(scaffolds[sc])'
        f'  else: test_idx.extend(scaffolds[sc])'
        f'rf=RandomForestRegressor(n_estimators=500,max_depth=20,random_state=42,n_jobs=-1);'
        f'rf.fit([X_lig[i] for i in train_idx],y[train_idx]);'
        f'pred=rf.predict([X_lig[i] for i in test_idx]);'
        f'rmse=np.sqrt(mean_squared_error(y[test_idx],pred));'
        f'pearson=np.corrcoef(pred,y[test_idx])[0,1];'
        f'print(f\"TEST_RMSE={rmse:.4f} PEARSON={pearson:.4f} N={len(test_idx)}\")" 2>&1', timeout=300)
    for line in (out+err).strip().split('\n'):
        if 'TEST_RMSE' in line or 'ERROR' in line.lower() or 'Traceback' in line:
            print(f'  {line.strip()[:200]}')
        if 'TEST_RMSE' in line:
            parts = line.strip().split()
            rmse_val = float([p.split('=')[1] for p in parts if 'TEST_RMSE' in p][0])
            pear_val = float([p.split('=')[1] for p in parts if 'PEARSON' in p][0])
            results[ds] = {'rmse': rmse_val, 'pearson': pear_val, 'n': len(test_idx)}
    time.sleep(3)

c.close()

print('\n\n=== CROSS-DATASET RESULTS ===')
print(f'{"Dataset":25s} {"RMSE":>8s} {"Pearson R":>10s} {"N test":>8s}')
print('-'*55)
for ds in DATASETS:
    if ds in results:
        r = results[ds]
        print(f'{ds:25s} {r["rmse"]:8.4f} {r["pearson"]:10.4f} {r["n"]:8d}')
    else:
        print(f'{ds:25s} {"FAILED":>8s}')

# Save for paper
with open('cross_dataset_results.json','w') as f:
    json.dump(results,f,indent=2)
print('\nSaved to cross_dataset_results.json')
