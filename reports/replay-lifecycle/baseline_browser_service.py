"""Isolated, bounded browser scheduling fixture. No model inference."""
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_checkin_browser', root / 'app-baseline-47ff75b.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class Fixture:
    def status(self):
        return {'ready': True, 'components': {'fixture': 'Synthetic scheduling; no model inference'}, 'errors': []}
    def cancel(self, session, turn=None):
        return False
    def stream(self, *args, **kwargs):
        raise RuntimeError('Scheduling fixture has no model inference.')

with patch.object(module, 'load_replay_rows', return_value=[{'text': 'OLD REPLAY', 'video_path': None}]):
    app = module.build_app(pipeline=Fixture())
loader = next(f for f in app.fns.values() if f.name == 'select_replay')
original = loader.fn
def delayed(index):
    print(json.dumps({'event': 'loader_entered', 'monotonic': time.monotonic()}), flush=True)
    time.sleep(2.5)
    result = original(index)
    print(json.dumps({'event': 'loader_returned', 'monotonic': time.monotonic()}), flush=True)
    return result
loader.fn = delayed
try:
    app.launch(server_name='127.0.0.1', server_port=7862, share=False, inbrowser=False, prevent_thread_lock=True)
    print('BASELINE_FIXTURE_READY; lifetime150seconds; no inference', flush=True)
    time.sleep(150)
finally:
    app.close()
