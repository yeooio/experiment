import json
import os
from pathlib import Path
import sys
import runner
from allocation import lane_tasks

root = Path(__file__).resolve().parent
machine = sys.argv[1]
assert machine == '5090', 'This package only runs Oxford and CS2'
lane = sys.argv[2]
dataset = 'oxford' if lane == 'oxford' else 'cs2'
all_tasks = runner.tasks
def assigned_tasks():
    yield from lane_tasks(all_tasks(), lane)
runner.tasks = assigned_tasks
task_count = sum(1 for _ in assigned_tasks())
manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
assert runner.fingerprint('first')[0] == manifest['fingerprint']
lock = root / (machine+'_'+lane+'.lock')
with lock.open('x') as f:
    f.write(str(os.getpid()))
try:
    sys.argv = ['runner.py', 'run', '--cnn-pool', 'first', '--device', 'cuda',
                '--models', ','.join(runner.MODELS),
                '--output', str(root/'results'/machine/dataset)]
    print('START', machine, lane, task_count, 'tasks; fixed seed 1; epochs 100/1000', flush=True)
    runner.main()
finally:
    lock.unlink(missing_ok=True)
