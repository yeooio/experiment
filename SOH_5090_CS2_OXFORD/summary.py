import collections
import json
from pathlib import Path
import sys
root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
expected = {t['id']: t for t in manifest['tasks']}
seen = {}
for directory in (list(map(Path, sys.argv[1:])) or [root/'results']):
    for path in directory.rglob('result.json'):
        r = json.loads(path.read_text(encoding='utf-8'))
        t = r['task']
        assert r['fingerprint'] == manifest['fingerprint'] and t == expected.get(t['id']), path
        assert t['id'] not in seen, ('Duplicate task', t['id'])
        seen[t['id']] = r
for ds in ['cs2', 'cx2', 'oxford', 'mit']:
    rows = [r for r in seen.values() if r['task']['dataset'] == ds]
    print(ds, len(rows), '/ 2432')
    for model in sorted({t['model'] for t in expected.values()}):
        candidates = [r for r in rows if r['task']['model'] == model]
        count = sum(t['dataset']==ds and t['model']==model for t in expected.values())
        if len(candidates) == count:
            best = min(candidates, key=lambda r: (r['configuration_metrics']['RMSE'], r['task']['id']))
            print('FINAL CONFIG', model, best['task'], best['configuration_metrics'])
print('Total', len(seen), '/ 9728; no best configuration declared for incomplete model grids.')
