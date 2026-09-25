"""Controlled CPU scheduling reproduction, never an inference or browser test."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from unittest.mock import patch

from gradio.state_holder import SessionState

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "app-baseline-47ff75b.py"


class Fixture:
    def status(self):
        return {"ready": True, "errors": [], "components": {"fixture": "synthetic CPU scheduling only"}}

    def cancel(self, session, turn=None):
        return False


async def main():
    spec = importlib.util.spec_from_file_location("replay_baseline", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch.object(module, "load_replay_rows", return_value=[{"text": "OLD REPLAY", "video_path": None}]):
        app = module.build_app(pipeline=Fixture())
    try:
        funcs = {f.name: f for f in app.fns.values() if f.fn is not None}
        loader, reset = funcs["select_replay"], funcs["new_conversation"]
        entered, release = threading.Event(), threading.Event()
        original = loader.fn
        def delayed(index):
            entered.set()
            assert release.wait(5)
            return original(index)
        loader.fn = delayed
        state = SessionState(app)
        state[reset.inputs[0]._id] = "old-conversation"
        state[reset.inputs[1]._id] = 0
        task = asyncio.create_task(app.process_api(loader, ["0"], state=state, session_hash="browser-A"))
        assert await asyncio.to_thread(entered.wait, 2)
        cleared = await app.process_api(reset, [None, None], state=state, session_hash="browser-A")
        release.set()
        late = await task
        report = {
            "kind": "synthetic CPU controlled scheduling; no inference/browser claim",
            "baseline_commit": "47ff75b1286cafcb51322f1631078b4af152ad0b",
            "app_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "reset_text": cleared["data"][7],
            "late_loader_outputs": late["data"],
            "new_session": state[reset.inputs[0]._id],
            "reset_cancellation_dependencies": [f.cancels for f in app.fns.values() if f.cancels],
            "loader_id": loader._id,
            "stale_output_reproduced": late["data"][0] == "OLD REPLAY",
        }
        (ROOT / "baseline-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    finally:
        app.close()


if __name__ == "__main__":
    asyncio.run(main())
