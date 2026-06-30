import paramiko
c=paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('aicoe.snu.in',username='rajeev',password='@@Gujrattt123',timeout=15)

# Kill stuck proceses
_,o,_=c.exec_command('kill 3539114 2>/dev/null; kill 3539115 2>/dev/null; kill 3558392 2>/dev/null; echo done')
print('Kill:', o.read().decode(errors='replace').strip())

# Results
_,o,_=c.exec_command('cd ~/bica && grep seed99 diary/results_diary.csv')
results=o.read().decode(errors='replace').strip()
print('\n=== Protein split ===')
for l in results.split('\n'):
    p=l.split(',')
    print(f'  {p[1]:50s} RMSE={p[17]}  R={p[18]}  rho={p[19]}')

# What's still running
_,o,_=c.exec_command('ps aux | grep run_experiment | grep -v grep || echo none')
print('\nRunning:')
print(o.read().decode(errors='replace')[:500])

c.close()
