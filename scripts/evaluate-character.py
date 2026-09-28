"""Controlled evidence interventions using the real local generator, no fake video."""
from pathlib import Path
from datetime import datetime, timezone
import json
import time
from checkin.generator import LocalGenerator, _messages

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('.artifacts/reports/game-character'))
args = parser.parse_args()
out = args.output
out.mkdir(exist_ok=True, parents=True)
generator = LocalGenerator()
cases = []
for line in ["Oh, fantastic.", "You want me to cross that?", "Sure. Whatever."]:
    for visual in ["joy", "fear", "anger", None]:
        cases.append((line, "neutral", visual))
cases += [("I'm scared of heights. Let's take the sea stairs.", "fear", "joy"),
          ("I'm excited to try the bridge!", "joy", "sadness"),
          ("Pause the game. What are you?", "neutral", None)]
record = {'created_at': datetime.now(timezone.utc).isoformat(),
          'scope': 'Synthetic emotion interventions; real local Qwen3-4B responses. Not measured vision accuracy or human-rated quality.',
          'model': 'Existing local Qwen3-4B GGUF, unchanged', 'cases': []}
for index, (text, predicted, visual) in enumerate(cases):
    state = {'emotion': {'label': predicted, 'source': 'fusion' if visual else 'text_fallback'},
             'vision': {'available': visual is not None},
             'modalities': {'text_label': predicted, 'vision_label': visual},
             'modality_disagreement': visual is not None and predicted != visual}
    start = time.perf_counter()
    parts, first, error = [], None, None
    try:
        for delta in generator.stream(text, state, []):
            if first is None: first = (time.perf_counter() - start) * 1000
            parts.append(delta)
    except Exception as exc:
        error = str(exc)
    case = {'text': text, 'state': state, 'direction': json.loads(_messages(text,state,[])[-1]['content'])['npc_direction'],
            'response': ''.join(parts), 'first_delta_ms': first, 'total_ms': (time.perf_counter()-start)*1000, 'error': error}
    record['cases'].append(case)
    (out/'controlled-responses.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'case':index, 'visual':visual, 'response':case['response'], 'error':error}),flush=True)
