#!/usr/bin/env python3
"""Upload evaluate_rf_balm.py to bullitt via SFTP, run it via SSH, download results."""
import paramiko
import os, json, sys

HOST = 'aicoe.snu.in'
USER = 'rajeev'
PASS = '@@Gujrattt123'
REMOTE_SCRIPT = '/home/rajeev/bica/evaluate_rf_balm.py'
LOCAL_SCRIPT = r'C:\Users\sps26\Desktop\bica\evaluate_rf_balm.py'
RESULTS_REMOTE = '/home/rajeev/bica/xdataset_aac_results.json'
RESULTS_LOCAL = r'C:\Users\sps26\Desktop\bica\xdataset_aac_results.json'

# ---- 1. SFTP upload ----
print("Connecting for SFTP upload...")
transport = paramiko.Transport((HOST, 22))
transport.connect(username=USER, password=PASS)
sftp = paramiko.SFTPClient.from_transport(transport)
sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
print(f"Uploaded {LOCAL_SCRIPT} -> {REMOTE_SCRIPT}")

# ---- 2. SSH execution ----
print("Connecting for SSH execution...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)

cmd = 'bash -c "source ~/bica/bica_env/bin/activate && cd ~/bica && python evaluate_rf_balm.py"'
print(f"Running: {cmd}")
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=600)
exit_code = stdout.channel.recv_exit_status()
output = stdout.read().decode('utf-8', errors='replace')
err_output = stderr.read().decode('utf-8', errors='replace')

print("\n--- STDOUT ---")
print(output)
if err_output.strip():
    print("\n--- STDERR ---")
    print(err_output)
print(f"\n--- Exit code: {exit_code} ---")

# ---- 3. Download results ----
if exit_code == 0:
    sftp.get(RESULTS_REMOTE, RESULTS_LOCAL)
    print(f"Downloaded {RESULTS_REMOTE} -> {RESULTS_LOCAL}")
    with open(RESULTS_LOCAL) as f:
        results = json.load(f)
    print("\nFinal results:")
    print(json.dumps(results, indent=2))
else:
    print("Script failed — no results to download.")

sftp.close()
transport.close()
ssh.close()
print("Done.")
