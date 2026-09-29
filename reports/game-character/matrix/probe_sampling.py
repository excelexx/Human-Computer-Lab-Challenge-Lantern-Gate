import importlib.util
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
scripts = root / 'outputs/checkin/scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('matrix_runner', scripts / 'evaluate-dialogue-matrix.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
selected = []
for item in runner.cases.matrix_cases():
    parts = item['id'].split('/')
    if (item['section'] == 'presets' and item['stage'] == 'opening') or (
        item['section'] == 'customs' and item['stage'] == 'opening' and parts[2] in
        ('identity','safety','pause','change','weather','will_not','explicit_neutral')) or (
        item['section'] == 'customs' and item['stage'] == 'ready_stairs' and parts[2] in
        ('identity','safety','not_yet','other_route')) or (item['section']=='exploratory' and parts[1]=='camera_claim'):
        selected.append(item)
out = root / 'work/dialogue-matrix/sampling-probe'
out.mkdir()
metadata = {'case_ids': [r['id'] for r in selected], 'system_prompt': runner.character.SYSTEM_PROMPT,
            'source_sha256': {n: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                             for n,m in [('character',runner.character),('quest',runner.quest),('generator',runner.generator)]},
            'protocol': 'Development-only comparison of temperature .5 versus greedy 0 with otherwise identical local requests; synthetic evidence, fixed canonical histories.'}
runner.atomic_json(out/'metadata.json',metadata)
for temperature in (.5,0.):
    runner.SAMPLING['temperature'] = temperature
    with runner.httpx.Client(trust_env=False,timeout=90) as client, (out/f'temperature-{temperature}.jsonl').open('x',encoding='utf8') as handle:
        for item in selected:
            row = runner.generate(client,item)
            row['sampling'] = dict(runner.SAMPLING)
            handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush()
    print(json.dumps({'temperature':temperature,'cases':len(selected)}),flush=True)
