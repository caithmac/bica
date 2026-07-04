import csv
with open('diary/results_diary.csv') as f:
    rows = list(csv.DictReader(f))
best = {}
for r in rows:
    eid = r['experiment_id']
    if '__leakypdb' in eid or '__seed' in eid: continue
    try: rmse = float(r['test_rmse'])
    except: continue
    if eid not in best or rmse < float(best[eid]['test_rmse']): best[eid] = r
top = sorted(best.values(), key=lambda r: float(r['test_rmse']))[:20]
for i, r in enumerate(top):
    eid = r['experiment_id'].replace('ext', '').replace('_', '\\_')[:40]
    fam = r['model_family'].replace('_','\\_')
    lig = r['ligand_repr'].replace('_','\\_')[:18]
    prot = r['protein_repr'].replace('_','\\_')[:14]
    rmse = r['test_rmse']
    pear = r['test_pearson_r']
    spear = r['test_spearman_r']
    bs = chr(92) + chr(92)
    print(f'{i+1:2d} & {eid:40s} & {fam:14s} & {lig:18s} & {prot:14s} & {rmse} & {pear} & {spear} {bs}')
    if i < 19: print('\\midrule')
