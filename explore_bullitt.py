#!/usr/bin/env python3
"""Explore bullitt: find dataset files and directory structure."""
import paramiko

HOST = 'aicoe.snu.in'
USER = 'rajeev'
PASS = '@@Gujrattt123'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)

# Check ~/bica structure
commands = [
    'ls -la /home/rajeev/bica/',
    'find /home/rajeev/bica -maxdepth 3 -name "*.csv" -o -name "*.parquet" -o -name "*.json" -o -name "*.tsv" 2>/dev/null',
    'find /home/rajeev -maxdepth 4 -name "BindingDB*" -o -name "Leaky*" -o -name "Mpro*" -o -name "USP7*" 2>/dev/null',
    'find /home/rajeev/bica -maxdepth 2 -type d 2>/dev/null',
    'ls -la /home/rajeev/bica/bica_env/bin/python*',
    'which python3',
    'ls -la /home/rajeev/',
]

for cmd in commands:
    print(f"\n$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out:
        print(out)
    if err:
        print(f"[STDERR] {err}")
    print(f"[exit={exit_code}]")

ssh.close()
