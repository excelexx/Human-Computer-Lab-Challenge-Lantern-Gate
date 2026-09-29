import json
import time
from datetime import datetime, timezone
from pathlib import Path
from checkin.generator import LocalGenerator

root=Path(__file__).resolve().parent
rows={r['id']:r for r in map(json.loads,(root/'final-run/responses.jsonl').read_text(encoding='utf8').splitlines())}
ids=['preset/opening/0/joy','preset/opening/0/anger','preset/opening/0/fear',
     'custom/ready_bridge/will_not/surprise','custom/ready_bridge/other_route/neutral',
     'custom/opening/identity/neutral','custom/ready_stairs/quote/neutral',
     'journey/bridge_refusal_recovery/4']
results=[]
for case_id in ids:
    row=rows[case_id];start=time.perf_counter();first=None;parts=[];error=None
    try:
        for part in LocalGenerator().stream(row['text'],row['state'],row['history']):
            if first is None:first=time.perf_counter()-start
            parts.append(part)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    result={'id':case_id,'text':row['text'],'emotion':row['emotion_intervention'],
        'response':''.join(parts),'first_delta_seconds':first,'seconds':time.perf_counter()-start,
        'deltas':len(parts),'error':error,'same_response_as_matrix':''.join(parts)==row['response']}
    results.append(result);print(json.dumps(result),flush=True)
record={'at':datetime.now(timezone.utc).isoformat(),'scope':'Actual production LocalGenerator.stream with saved synthetic evidence/history; camera and browser excluded. No sampling overrides.',
        'cases':results,'completed':len(results),'errors':sum(bool(r['error']) for r in results)}
(root/'streaming-smoke.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
assert not record['errors']
