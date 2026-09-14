import json
import os
from pathlib import Path
import subprocess
import sys
import time
from allocation import LANES
root = Path(__file__).resolve().parent
machine = sys.argv[1]
assert machine == '5090'
datasets = LANES
lock = root / (machine+'_parallel.lock')
with lock.open('x') as f:
    f.write(str(os.getpid()))
jobs = []
try:
    for ds in datasets:
        with (root/'logs'/(ds+'.log')).open('ab') as log:
            p = subprocess.Popen([sys.executable, '-u', str(root/'worker.py'), machine, ds],
                cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((ds,p))
        print('START', ds, 'PID', p.pid, flush=True)
    while True:
        status = {ds: {'pid': p.pid, 'exit_code': p.poll()} for ds,p in jobs}
        temp = root/'logs'/'status.tmp'
        temp.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temp.replace(root/'logs'/'status.json')
        if all(p.poll() is not None for _,p in jobs):
            break
        time.sleep(10)
    if any(p.returncode != 0 for _,p in jobs):
        raise RuntimeError('A worker failed; inspect dataset logs.')
finally:
    for _,p in jobs:
        if p.poll() is None:
            p.terminate()
            p.wait()
    lock.unlink(missing_ok=True)
