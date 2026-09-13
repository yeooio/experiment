"""Dataset allocation only; training code and its fingerprint remain unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['cs2', 'oxford', 'mit', 'cx2'])
parser.add_argument('--device', required=True, choices=['cpu', 'cuda'])
parser.add_argument('--gpu-index', type=int, default=0)
parser.add_argument('--limit', type=int)
args = parser.parse_args()
if args.device == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)

import runner

root = Path(__file__).resolve().parent
output = root / 'dataset_results' / args.dataset
output.mkdir(parents=True, exist_ok=True)
lock = output / 'DATASET_RUNNING.lock'
with lock.open('x', encoding='utf-8') as f:
    f.write(str(os.getpid()))
try:
    original_tasks = runner.tasks
    runner.tasks = lambda: (t for t in original_tasks() if t['dataset'] == args.dataset)
    records = {'dataset': args.dataset, 'device': args.device, 'gpu_index': args.gpu_index,
               'pid': os.getpid(), 'task_count': 12160,
               'allocation_wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    runner.atomic_json(output / 'allocation.json', records)
    sys.argv = ['runner.py', 'run', '--cnn-pool', 'first', '--models', ','.join(runner.MODELS),
                '--device', args.device, '--threads', '1', '--output', str(output)]
    if args.limit:
        sys.argv.extend(['--limit', str(args.limit)])
    print('DATASET START', args.dataset, args.device, '12160 tasks; 500 epochs; seeds 1..10', flush=True)
    runner.main()
finally:
    lock.unlink(missing_ok=True)
