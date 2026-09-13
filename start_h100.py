"""Launch one GPU worker per dataset on a single H100."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu-index', type=int, default=0)
parser.add_argument('--skip-name-check', action='store_true')
args = parser.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; refusing CPU fallback')
if not 0 <= args.gpu_index < torch.cuda.device_count():
    raise ValueError('Invalid GPU index')
gpu_name = torch.cuda.get_device_name(args.gpu_index)
if 'H100' not in gpu_name.upper() and not args.skip_name_check:
    raise RuntimeError(f'Expected an H100, detected {gpu_name}; use --skip-name-check only after checking the allocation')

root = Path(__file__).resolve().parent
logs = root / 'logs'
logs.mkdir(exist_ok=True)
datasets = ['cs2', 'cx2', 'mit', 'oxford']
for dataset in datasets:
    lock = root / 'dataset_results' / dataset / 'DATASET_RUNNING.lock'
    if lock.exists():
        raise RuntimeError(f'{dataset} already has a dataset lock: {lock}')

children = []
try:
    for dataset in datasets:
        stdout = (logs / f'{dataset}.stdout.log').open('a', encoding='utf-8', buffering=1)
        stderr = (logs / f'{dataset}.stderr.log').open('a', encoding='utf-8', buffering=1)
        command = [sys.executable, '-u', str(root / 'dataset_worker.py'),
                   '--dataset', dataset, '--device', 'cuda', '--gpu-index', str(args.gpu_index)]
        process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
        children.append((dataset, process, stdout, stderr))
        print(f'STARTED {dataset}: PID={process.pid}, GPU={args.gpu_index}', flush=True)
    while children:
        for dataset, process, stdout, stderr in list(children):
            code = process.poll()
            if code is None:
                continue
            stdout.close(); stderr.close(); children.remove((dataset, process, stdout, stderr))
            if code != 0:
                raise RuntimeError(f'{dataset} worker exited with {code}; inspect logs before restarting')
            print(f'COMPLETED {dataset}', flush=True)
        time.sleep(5)
finally:
    failed = sys.exc_info()[0] is not None
    if failed:
        for _, process, _, _ in children:
            if process.poll() is None:
                process.terminate()
        for _, process, stdout, stderr in children:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill()
            stdout.close(); stderr.close()
