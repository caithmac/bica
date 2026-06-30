from datasets import load_dataset
DATASETS = ['BindingDB_filtered','HIF2A','LeakyPDB','MCL1','Mpro','SYK','USP7']
for ds_name in DATASETS:
    try:
        ds = load_dataset("BALM/BALM-benchmark", ds_name, split="train")
        df = ds.to_pandas()
        print(f'{ds_name}: {len(df)} rows')
        print(f'  Cols: {list(df.columns)}')
        print(f'  Row 0: {dict(df.iloc[0])}')
        print()
    except Exception as e:
        print(f'{ds_name}: ERROR {e}')
