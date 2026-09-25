"""Stress-replicate revealed reserved cases with the unchanged production sampler.

This is not a fresh holdout, candidate comparison or prompt-selection stage.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import httpx

from checkin.generator import SYSTEM_PROMPT
from checkin.response_backtest import canonical_hash, generate, read_protocol, summarize
from checkin.train import atomic_json, file_sha256

out = Path(__file__).resolve().parent
original = read_protocol(out)
selection = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
selected = selection['selected_prompt']
assert SYSTEM_PROMPT == original['prompts'][selected]
protocol_path = out / 'production-replication-protocol.json'
if not protocol_path.exists():
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Production-sampler stress replication of already revealed reserved authored cases; no new holdout, selection or prompt tuning.',
        'original_protocol_sha256': original['protocol_sha256'],
        'original_selection_sha256': file_sha256(out / 'selection.json'),
        'already_revealed_check_responses_sha256': file_sha256(out / 'responses-check.jsonl'),
        'script_sha256': file_sha256(Path(__file__)),
        'base_url': original['base_url'],
        'prompts': {selected: original['prompts'][selected]},
        'prompt_sha256': {selected: original['prompt_sha256'][selected]},
        'model': original['model'],
        'sampler': {'model': 'checkin-qwen', 'temperature': .5, 'top_p': .8, 'top_k': 20, 'min_p': 0., 'max_tokens': 96},
        'seeds': [42, 43, 44],
        'cases': [case for case in original['cases'] if case['split'] == 'check'],
        'planned_generations': 54,
    }
    protocol['protocol_sha256'] = canonical_hash(protocol)
    atomic_json(protocol_path, protocol)
protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
assert protocol['protocol_sha256'] == canonical_hash({key: value for key, value in protocol.items() if key != 'protocol_sha256'})
path = out / 'responses-production-replication.jsonl'
rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
done = {row['id'] for row in rows}
with httpx.Client(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(90, connect=5)) as client:
    with path.open('a', encoding='utf-8') as handle:
        for seed in protocol['seeds']:
            trial = {**protocol, 'sampler': {**protocol['sampler'], 'seed': seed}}
            for case in protocol['cases']:
                for condition in case['conditions']:
                    key = f"{selected}|{case['id']}|{condition['name']}|primary|seed={seed}"
                    if key in done:
                        continue
                    row = generate(client, trial, selected, case, condition)
                    row.update(id=key, seed=seed, evaluation_stage='production_sampler_stress_replication_of_revealed_checks')
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                    rows.append(row)
                    print(f"{len(rows)}/54 {key} {row['completion_ms']:.0f} ms", flush=True)
summary = {
    'completed_at': datetime.now(timezone.utc).isoformat(),
    'protocol_sha256': protocol['protocol_sha256'],
    'responses_sha256': file_sha256(path),
    'purpose': protocol['purpose'],
    'generations': len(rows),
    'errors': sum(row['error'] is not None for row in rows),
    'length_limit_finishes': sum(row['finish_reason'] == 'length' for row in rows),
    'responses_with_narrow_pattern_flags': sum(bool(row['pattern_flags']) for row in rows),
    'per_seed': {str(seed): summarize([row for row in rows if row['seed'] == seed]) for seed in protocol['seeds']},
    'caveat': 'Repeated samples of the same revealed authored inputs are not 54 independent situations. Regex flags and exact response changes are not quality or clinical scores. No matched baseline was requested, so this stage cannot estimate comparative production improvement.'
}
atomic_json(out / 'summary-production-replication.json', summary)
print(json.dumps(summary, indent=2))
