import paramiko, time
c=paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
for attempt in range(10):
    try:
        c.connect('aicoe.snu.in',username='rajeev',password='@@Gujrattt123',timeout=20)
        break
    except: time.sleep(10)
# kill all
c.exec_command('kill -9 $(pgrep -f run_experiment) 2>/dev/null; echo done')
c.close()
print('Cleaned up')
