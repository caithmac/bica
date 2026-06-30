import paramiko
c=paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('aicoe.snu.in',username='rajeev',password='@@Gujrattt123',timeout=15)

# Kill all hung experiments
_,o,_=c.exec_command('kill -9 $(pgrep -f "run_experiment.py") 2>/dev/null; sleep 2; echo done')
print('Kill:', o.read().decode(errors='replace').strip())

# Check GPU
_,o,_=c.exec_command('cd ~/bica && source bica_env/bin/activate && python -c "import torch; print(f' + "'GPU: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}'" + ')"')
print('GPU:', o.read().decode(errors='replace').strip())

# Check protein feature cache status
_,o,_=c.exec_command('ls ~/bica/cache/features/prot* 2>/dev/null | head -10 || echo no_prot_cache')
print('Protein cache:', o.read().decode(errors='replace')[:500])

# Show what we have for protein split
_,o,_=c.exec_command('cd ~/bica && grep seed99 diary/results_diary.csv')
print('\nProtein split results:')
for l in o.read().decode(errors='replace').strip().split('\n'):
    p=l.split(',')
    print(f'  {p[1]:50s} RMSE={p[17]}')
c.close()
