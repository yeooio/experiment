"""Merge records, detect duplicate/inconsistent trials, rank complete ten-seed groups."""
import argparse
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('result_roots', nargs='+', type=Path)
parser.add_argument('--output', type=Path, default=root / 'summary.json')
args = parser.parse_args()
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
expected = {t['id']: t for t in manifest['tasks']}
seen, groups = {}, {}
for directory in args.result_roots:
    for path in sorted(directory.rglob('result.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        task = r['task']; index = task['id']
        if r['fingerprint'] != manifest['fingerprint'] or task != expected.get(index):
            raise ValueError(f'Protocol mismatch: {path}')
        if index in seen:
            raise ValueError(f'Duplicate task {index}: {seen[index]} and {path}; resolve before merging')
        seen[index] = str(path)
        key = json.dumps({k: v for k, v in task.items() if k not in ['id', 'seed']}, sort_keys=True)
        groups.setdefault(key, []).append(r)
complete = []
for key, rows in groups.items():
    if {r['task']['seed'] for r in rows} != set(range(1, 11)):
        continue
    scores = [r['configuration_metrics']['RMSE'] for r in rows]
    complete.append(dict(config=json.loads(key), mean_configuration_RMSE=statistics.mean(scores),
                         std_configuration_RMSE=statistics.stdev(scores), seeds=10))
complete.sort(key=lambda r: (r['mean_configuration_RMSE'], json.dumps(r['config'], sort_keys=True)))
by_pair = {}
for row in complete:
    c = row['config']; key = c['dataset'] + '/' + c['model']
    by_pair.setdefault(key, row)
summary = dict(expected_tasks=len(expected), completed_tasks=len(seen),
               missing_tasks=len(expected)-len(seen), complete_ten_seed_groups=len(complete),
               provisional_best_among_completed_groups=by_pair,
               warning='No final configuration selection until every planned group for that dataset/model is complete.')
args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'provisional_best_among_completed_groups'}, ensure_ascii=False))
