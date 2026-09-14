"""One worker per machine, explicit disjoint partition, no installation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import platform

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('machine', nargs='?', default='5090', choices=['5090'])
p.add_argument('--foreground', action='store_true')
a = p.parse_args()
import torch
import numpy
import pandas
import sklearn
assert torch.cuda.is_available(), 'CUDA unavailable; no automatic installation or CPU fallback'
print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, flush=True)
models = 'LSTM,CNN-LSTM,MS-AgentNet,Transformer,CNN-Transformer'
logs = root / 'logs'
logs.mkdir(exist_ok=True)
envinfo = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
               cuda=torch.version.cuda, numpy=numpy.__version__, pandas=pandas.__version__,
               sklearn=sklearn.__version__, gpu=torch.cuda.get_device_name(0))
envpath = logs / ('environment_' + a.machine + '.json')
envpath.write_text(json.dumps(envinfo, indent=2), encoding='utf-8')
# Check the model/data on the actual machine before requesting background work.
subprocess.run([sys.executable, str(root/'runner.py'), 'check', '--cnn-pool', 'first',
                '--models', models, '--device', 'cuda'], cwd=root, check=True)
cmd = [sys.executable, '-u', str(root/'parallel.py'), a.machine]
if a.foreground:
    subprocess.run(cmd, cwd=root, check=True)
else:
    kwargs = {'start_new_session': True} if os.name != 'nt' else {
        'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    with (logs/(a.machine+'.log')).open('ab') as log:
        proc = subprocess.Popen(cmd, cwd=root, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, **kwargs)
    print('Launch requested PID', proc.pid, 'Inspect logs/'+a.machine+'.log', flush=True)
