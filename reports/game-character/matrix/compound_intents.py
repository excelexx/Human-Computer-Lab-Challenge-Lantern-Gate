"""Post-review regression probe; synthetic evidence, actual local generation."""
from pathlib import Path
import importlib.util
import sys
import json
import hashlib
import httpx

root=Path(__file__).resolve().parents[2]
scripts=root/'outputs/checkin/scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('matrix_runner',scripts/'evaluate-dialogue-matrix.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
out=root/'work/dialogue-matrix/compound-intents';out.mkdir(exist_ok=True)
texts=[('ready_topic',"I'm ready, but let's talk about something else.",'talking'),
       ('ready_space',"I'm ready. Please give me some space.",'talking'),
       ('ready_company',"Let's take the bridge and stay close.",'depart')]
results=[]
with httpx.Client(trust_env=False,timeout=90) as client, (out/'responses.jsonl').open('w',encoding='utf8') as f:
 for stage in ('ready_bridge','ready_stairs'):
  for slug,text,expected in texts:
   for emotion in runner.cases.EMOTIONS:
    item=runner.cases.case(f'compound/{stage}/{slug}/{emotion}','compound',stage,text,emotion,'Explicit pause/space outranks readiness; added companionship must not contradict confirmed departure.')
    result=runner.generate(client,item)
    result['expected_phase']=expected
    result['action_pass']=result['game_after']['phase']==expected and (expected!='depart' or result['game_after']['route']=='bridge')
    f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();results.append(result)
    print(json.dumps([result['id'],result['game_after'],result['response'],result['error']]),flush=True)
metadata={'sampling':runner.SAMPLING,'source_sha256':{name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in [('quest',runner.quest),('generator',runner.generator),('cases',runner.cases)]},
          'scope':'42 targeted development regressions added after final matrix review, not part of the original 1157 cases.',
          'generated':len(results),'errors':sum(bool(r['error']) for r in results),'action_failures':sum(not r['action_pass'] for r in results)}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
assert metadata['errors']==metadata['action_failures']==0
