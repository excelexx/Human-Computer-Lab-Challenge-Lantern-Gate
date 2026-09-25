"""Upload preparation must not overwrite inputs after another UI action."""
import asyncio
import inspect
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from gradio.components.video import VideoData
from gradio.data_classes import FileData
from gradio.state_holder import SessionState


class FixturePipeline:
    def status(self):
        return {"ready": True, "errors": [], "components": {"fixture": "not inference"}}

    def cancel(self, session, turn=None):
        return False


@pytest.fixture
def upload_ui(tmp_path, monkeypatch):
    import checkin.app as module
    monkeypatch.setattr(module, "load_replay_rows", lambda home: [])
    app = module.build_app(tmp_path, pipeline=FixturePipeline())
    funcs = {f.name: f for f in app.fns.values() if f.fn is not None}
    yield app, funcs, funcs["admit_upload"].inputs[0]
    app.close()


def payload(path="fixture.webm"):
    return VideoData(video=FileData(path=path)).model_dump()


def admit(funcs, guard, path="fixture.webm"):
    return next(funcs["admit_upload"].fn(payload(path), guard["epoch"], guard))[0]


def test_upload_admission_is_lightweight_cancellable_and_keeps_validated_paths_server_side(upload_ui):
    app, funcs, _ = upload_ui
    app.validate_queue_settings()
    admission, loader = funcs["admit_upload"], funcs["normalize_upload"]
    assert not admission.preprocess
    assert admission.queue and loader.queue
    assert inspect.isgeneratorfunction(admission.fn) and inspect.isgeneratorfunction(loader.fn)
    assert admission.concurrency_id != "gpu" and loader.concurrency_id == "gpu"
    assert not admission.inputs[1].stateful and not loader.inputs[0].stateful
    for name in ("new_conversation", "stop_current", "run_turn", "abandon_replay"):
        canceller = next(fn for fn in app.fns.values() if fn.cancels and fn.targets == funcs[name].targets)
        assert {admission._id, loader._id} <= set(canceller.cancels)
    guard = {"owner": None, "epoch": "current"}
    ticket = admit(funcs, guard)
    assert set(ticket) == {"id", "epoch"}
    assert guard["upload_payload"] == payload()


@pytest.mark.parametrize("action", ["new_conversation", "stop_current", "abandon_replay"])
def test_action_during_normalization_suppresses_all_late_updates(upload_ui, monkeypatch, action):
    _, funcs, camera = upload_ui
    guard = {"owner": None, "epoch": "current"}
    ticket = admit(funcs, guard)
    def slow_preprocess(value):
        if action == "abandon_replay":
            funcs[action].fn(guard)
        else:
            funcs[action].fn("session", 0, guard)
        return "normalized.webm"
    monkeypatch.setattr(camera, "preprocess", slow_preprocess)
    stream = funcs["normalize_upload"].fn(ticket, guard)
    next(stream)
    assert all(value == {"__type__": "update"} for value in next(stream))
    assert guard["owner"] is None and "upload_payload" not in guard
    stream.close()


def test_new_upload_supersedes_old_preparation_without_using_client_ticket_path(upload_ui, monkeypatch):
    _, funcs, camera = upload_ui
    guard = {"owner": None, "epoch": "current"}
    old = admit(funcs, guard, "first.webm")
    stream = funcs["normalize_upload"].fn(old, guard)
    next(stream)
    new = admit(funcs, guard, "second.webm")
    seen = []
    monkeypatch.setattr(camera, "preprocess", lambda item: seen.append(item.video.path) or "prepared.webm")
    assert all(value == {"__type__": "update"} for value in next(stream))
    assert guard["owner"] == new["id"]
    result = list(funcs["normalize_upload"].fn({**new, "path": "untrusted.webm"}, guard))[-1]
    assert result[3]["value"] == "prepared.webm"
    assert seen == ["first.webm", "second.webm"]
    assert "upload_payload" not in guard


def test_old_epoch_admission_cannot_take_ownership_after_reset(upload_ui):
    _, funcs, _ = upload_ui
    guard = {"owner": None, "epoch": "old"}
    funcs["new_conversation"].fn("session", 0, guard)
    result = next(funcs["admit_upload"].fn(payload(), "old", guard))
    assert all(value == {"__type__": "update"} for value in result)
    assert guard["owner"] is None


def test_failure_restores_controls_and_preserves_message(upload_ui, monkeypatch):
    _, funcs, camera = upload_ui
    guard = {"owner": None, "epoch": "current"}
    ticket = admit(funcs, guard)
    def broken(value):
        raise ValueError("controlled failed clip")
    monkeypatch.setattr(camera, "preprocess", broken)
    result = list(funcs["normalize_upload"].fn(ticket, guard))[-1]
    assert "could not be prepared" in result[0]
    assert result[4] == {"interactive": True, "__type__": "update"}
    assert guard["owner"] is None and "upload_payload" not in guard


def test_admission_still_enforces_gradio_upload_path_validation(upload_ui, tmp_path):
    from gradio.context import LocalContext
    app, funcs, _ = upload_ui
    private_path = tmp_path / "outside-upload-cache.webm"
    private_path.write_bytes(b"not a permitted upload")
    async def check():
        state = SessionState(app)
        fn = funcs["admit_upload"]
        epoch = state[fn.inputs[2]._id]["epoch"]
        # Reproduce the request-local access checks without launching a server.
        token = LocalContext.blocks.set(app)
        app.has_launched = True
        try:
            with pytest.raises(Exception, match="not uploaded|allowed_paths|upload folder"):
                await app.process_api(fn, [payload(str(private_path)), epoch, None], state=state, session_hash="browser")
        finally:
            app.has_launched = False
            LocalContext.blocks.reset(token)
    asyncio.run(check())


def test_valid_missing_duration_webm_is_published_with_finite_preview(upload_ui, tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for the real upload regression")
    _, funcs, _ = upload_ui
    source = tmp_path / "browser.webm"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                    "testsrc2=size=128x96:rate=12:duration=1.5", "-c:v", "libvpx", "-an", "-live", "1", str(source)],
                   check=True, capture_output=True, timeout=30)
    guard = {"owner": None, "epoch": "current"}
    ticket = admit(funcs, guard, str(source))
    result = list(funcs["normalize_upload"].fn(ticket, guard))[-1]
    prepared = Path(result[3]["value"])
    assert prepared != source and prepared.is_file()
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-of", "json", str(prepared)],
                           check=True, capture_output=True, text=True, timeout=30)
    assert float(json.loads(probe.stdout)["format"]["duration"]) == pytest.approx(1.5)
    assert result[4] == {"interactive": True, "__type__": "update"}
