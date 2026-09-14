"""Portable deterministic task partitioning, archived training reuse, 500 epochs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import ast
import hashlib
import itertools
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'source'))
import reference_models
import reference_cnn
import own_model

FEATURES = {'cs2': ['CCCT_3p8_4p0', 'IC_peak'], 'cx2': ['Q_dch_3p8_3p4'],
            'oxford': ['CCCT_V_main'], 'mit': ['Q_dch_window', 'energy_efficiency']}
CELLS = {'cs2': ['CS2_36', 'CS2_37'], 'cx2': ['CX2_36', 'CX2_37'],
         'oxford': ['Cell1', 'Cell2'], 'mit': ['b3c8', 'b3c13']}
MODELS = ['LSTM', 'CNN-LSTM', 'MS-AgentNet', 'Transformer', 'CNN-Transformer']


def training_library():
    path = ROOT / 'source/archived_training.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'hp_model')]
    lib = {'__file__': str(path), '__name__': 'archived_training_reuse'}
    exec(compile(tree, str(path), 'exec'), lib)
    lib['INPUT_ROOT'] = ROOT / 'inputs'
    for ds in FEATURES:
        lib['DATASETS'][ds] = {'features': FEATURES[ds], 'batch_size': 128, 'weight_decay': .0003}
    # Search reads only training and configuration cells. Reuse exact old loader
    # and normalizers; leave other report-cell CSVs untouched until final evaluation.
    lib['input_files'] = lambda ds: {r: ROOT / 'inputs' / ds / (r + '.csv') for r in ['train', 'in_selection']}
    return lib


def tasks():
    index = 0
    for ds in FEATURES:
        for model in MODELS:
            dims = [16, 32, 64, 128] if model not in ['LSTM', 'CNN-LSTM'] else [None]
            heads = [1, 2, 4] if model not in ['LSTM', 'CNN-LSTM'] else [None]
            for lr, depth, width, dense, head in itertools.product([.001, .01], [1, 2, 4, 8], [16, 32, 64, 128], dims, heads):
                for epochs in [100, 1000]:
                    yield dict(id=index, dataset=ds, model=model, lr=lr, depth=depth,
                               width=width, dense=dense, heads=head, seed=1, epochs=epochs)
                    index += 1


def build(t, device, pool):
    n = len(FEATURES[t['dataset']])
    if t['model'] == 'LSTM':
        return reference_models.LSTM(n, t['width'], t['depth'], 1, .01, device)
    if t['model'] == 'CNN-LSTM':
        return reference_models.CNN_LSTM(n, t['width'], t['depth'], 1, .01, 31, device)
    if t['model'] == 'MS-AgentNet':
        return own_model.MSAgentNet(n, 1, t['width'], 2, t['heads'], t['dense'], .01, t['depth'], 5)
    if t['model'] == 'CNN-Transformer':
        if pool != 'first':
            raise RuntimeError('CNN-Transformer blocked pending explicit first-encoder-only pooling approval')
        model = reference_cnn.Transformer_cnn(n, t['width'], t['dense'], t['heads'], .01, t['depth'], 1)
        for encoder in list(model.transformer_encoder)[1:]:
            encoder.maxpool = nn.Identity()
    else:
        model = reference_models.Transformer(n, t['width'], t['dense'], t['heads'], .01, t['depth'], 1)
    # Original constructors expect sequence-first while callers provide B,N,C.
    # Correct the layout flag without changing attention weights or layer counts.
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.batch_first = True
    return model


def fingerprint(pool):
    files = sorted((ROOT / 'source').glob('*.py')) + sorted((ROOT / 'inputs').glob('*/*.csv')) + [Path(__file__)]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    encoded = json.dumps({'hashes': hashes, 'pool': pool}, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(), hashes


def atomic_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def check(lib, args):
    records = []
    for ds in FEATURES:
        loaders, _, frames = lib['build_loaders'](ds, 1)
        print('DATA', ds, {r: len(f) for r, f in frames.items()}, flush=True)
        for name in args.models.split(','):
            for depth, width, head in itertools.product([1, 2, 4, 8], [16, 32, 64, 128], [1, 2, 4] if name not in ['LSTM', 'CNN-LSTM'] else [None]):
                t = dict(dataset=ds, model=name, depth=depth, width=width, dense=width, heads=head)
                model = build(t, args.device, args.cnn_pool).to(args.device).eval()
                x = torch.randn(3, 5, len(FEATURES[ds]), device=args.device)
                with torch.no_grad():
                    y, solo = model(x), model(x[:1])
                assert y.shape == (3, 1) and torch.isfinite(y).all(), t
                assert torch.allclose(y[:1], solo, atol=1e-5, rtol=1e-4), ('batch dependence', t, y[:1].tolist(), solo.tolist())
                model.train()
                model(x).square().mean().backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), t
                records.append(t)
    stamp, _ = fingerprint(args.cnn_pool)
    atomic_json(ROOT / 'preflight_pass.json', {'fingerprint': stamp, 'checks': len(records), 'models': args.models.split(','), 'device': str(args.device)})
    print('PREFLIGHT PASS', len(records), flush=True)


def execute(lib, t, args, stamp):
    folder = args.output / f"task_{t['id']:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'result.json'
    if done.exists():
        saved = json.loads(done.read_text(encoding='utf-8'))
        assert saved['fingerprint'] == stamp and saved['task'] == t
        return
    if shutil.disk_usage(folder).free < 2 * 1024**3:
        raise RuntimeError('Less than 2 GiB free; stopping before disk exhaustion')
    lock = folder / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    try:
        lib['seed_everything'](t['seed'])
        loaders, scaler, _ = lib['build_loaders'](t['dataset'], t['seed'])
        model = build(t, args.device, args.cnn_pool).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=.0003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
        checkpoint = folder / 'checkpoint.pt'
        start_epoch = 0
        if checkpoint.exists():
            c = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert c['fingerprint'] == stamp and c['task'] == t
            model.load_state_dict(c['model'])
            opt.load_state_dict(c['optimizer'])
            scheduler.load_state_dict(c['scheduler'])
            start_epoch = c['epoch']
            random.setstate(c['python_rng']); np.random.set_state(c['numpy_rng'])
            torch.set_rng_state(c['torch_rng'])
            loaders['train'].generator.set_state(c['loader_rng'])
            if args.device.type == 'cuda':
                torch.cuda.set_rng_state_all(c['cuda_rng'])
        start = time.perf_counter()
        for epoch in range(start_epoch, t['epochs']):
            loss = lib['train_epoch'](model, loaders['train'], opt, nn.MSELoss(), args.device)
            if not np.isfinite(loss):
                raise RuntimeError('Nonfinite training loss; do not silently skip trial')
            scheduler.step()
            if (epoch + 1) % 50 == 0:
                temp = folder / 'checkpoint.tmp'
                torch.save(dict(task=t, fingerprint=stamp, epoch=epoch+1, model=model.state_dict(),
                    optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    loader_rng=loaders['train'].generator.get_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device.type == 'cuda' else None), temp)
                temp.replace(checkpoint)
        metrics = lib['evaluate'](model, loaders['in_selection'], scaler, args.device)
        assert all(np.isfinite(v) for v in metrics.values()), metrics
        atomic_json(done, dict(task=t, fingerprint=stamp, configuration_metrics=metrics,
            seconds_this_session=time.perf_counter()-start, device=str(args.device), torch=torch.__version__))
        print('DONE', t['id'], t['dataset'], t['model'], metrics, flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['manifest', 'check', 'run', 'benchmark'])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--cnn-pool', default='blocked', choices=['blocked', 'first'])
    parser.add_argument('--models', default=','.join(MODELS[:-1]))
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    args.device = torch.device(args.device)
    assert 0 <= args.shard_index < args.shard_count
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if args.device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    lib = training_library()
    stamp, hashes = fingerprint(args.cnn_pool)
    if args.mode == 'manifest':
        atomic_json(ROOT / 'manifest.json', {'fingerprint': stamp, 'sources': hashes,
            'epochs': [100, 1000], 'seeds': [1], 'cells': CELLS, 'features': FEATURES,
            'selection': 'minimum configuration RMSE; one fixed seed; select only after full dataset/model grid completes',
            'tasks': list(tasks())})
        print('MANIFEST', len(list(tasks())))
    elif args.mode == 'check':
        check(lib, args)
    elif args.mode == 'benchmark':
        rows = []
        for ds in FEATURES:
            loaders, _, _ = lib['build_loaders'](ds, 1)
            for name in args.models.split(','):
                for depth, width in [(1, 16), (8, 128)]:
                    t = dict(dataset=ds, model=name, depth=depth, width=width, dense=width, heads=4)
                    lib['seed_everything'](1)
                    model = build(t, args.device, args.cnn_pool).to(args.device)
                    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0003)
                    criterion = nn.MSELoss()
                    lib['train_epoch'](model, loaders['train'], opt, criterion, args.device)
                    if args.device.type == 'cuda': torch.cuda.synchronize()
                    begin = time.perf_counter()
                    for _ in range(3):
                        lib['train_epoch'](model, loaders['train'], opt, criterion, args.device)
                    if args.device.type == 'cuda': torch.cuda.synchronize()
                    elapsed = time.perf_counter()-begin
                    row = dict(config=t, measured_epochs=3, seconds=elapsed,
                               estimate_500_seconds=elapsed/3*500, device=str(args.device))
                    rows.append(row)
                    print('TIMING', json.dumps(row), flush=True)
                    atomic_json(ROOT / ('timing_' + args.device.type + '.json'), rows)
    else:
        passed = json.loads((ROOT / 'preflight_pass.json').read_text(encoding='utf-8'))
        assert passed['fingerprint'] == stamp
        assert set(args.models.split(',')) <= set(passed['models'])
        count = 0
        for t in tasks():
            if (t['id'] // 2) % args.shard_count != args.shard_index or t['model'] not in args.models.split(','):
                continue
            execute(lib, t, args, stamp)
            count += 1
            if args.limit and count >= args.limit:
                break


if __name__ == '__main__':
    main()
