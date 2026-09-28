"""Paired scripted evidence interventions through the real local generator."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from checkin.character import SYSTEM_PROMPT, character_context
from checkin.generator import LocalGenerator, _messages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
cases = [("Oh, fantastic.", label, {"completed": 0, "route": None, "next_action": "choose_route"})
         for label in ['neutral', 'joy', 'fear', 'anger', 'sadness', 'surprise', 'disgust']]
for text, count, action in [('The sea stairs, then.', 1, 'confirm_departure'), ('Lead the way.', 2, 'walk_and_relight')]:
    cases.extend((text, label, {"completed": count, "route": "stairs", "next_action": action}) for label in ['joy','fear'])
cases.append(("I'm scared of heights. Let's take the sea stairs.", 'joy', {"completed": 1, "route": "stairs", "next_action": "confirm_departure"}))
record = {'created_at': datetime.now(timezone.utc).isoformat(),
          'protocol': '12 fixed synthetic evidence interventions. Real local generation; not webcam accuracy or human-rated quality. Route context fixed independently of emotion.',
          'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
          'system_prompt': SYSTEM_PROMPT, 'cases': []}
generator = LocalGenerator()
for text, visual, game in cases:
    state = {'emotion': {'label': 'neutral', 'source': 'fusion'},
             'vision': {'available': True}, 'modalities': {'text_label': 'neutral', 'vision_label': visual},
             'modality_disagreement': visual != 'neutral', 'game': game}
    started = time.perf_counter()
    chunks, error = [], None
    try:
        chunks.extend(generator.stream(text, state, []))
    except Exception as exc:
        error = str(exc)
    item = {'text': text, 'visual': visual, 'game': game,
            'direction': json.loads(_messages(text,state,[])[-1]['content'])['npc_direction'],
            'response': ''.join(chunks), 'error': error, 'seconds': time.perf_counter()-started}
    record['cases'].append(item)
    args.output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(item),flush=True)
