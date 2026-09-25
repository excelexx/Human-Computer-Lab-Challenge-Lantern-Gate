"""UI integration checks: camera state stays independent of message lifecycle."""
from unittest.mock import Mock
import numpy as np
import pytest

from checkin.app import build_app, live_emotion_html
from checkin.live_vision import LiveVisionBuffer


class Pipeline:
    def __init__(self):
        self.observe_live = Mock()
        self.calls = []

    def status(self):
        return {"ready": True, "errors": []}

    def cancel(self, *args):
        return False

    def stream(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        yield {"type": "done", "state": {"emotion": {"label": "neutral"},
            "response": {"text": "A local fixture reply.", "status": "complete"}}}


@pytest.fixture
def live_app(tmp_path):
    pipe = Pipeline()
    app = build_app(tmp_path, pipeline=pipe)
    funcs = {fn.name: fn for fn in app.fns.values() if fn.fn is not None}
    yield app, funcs, pipe
    app.close()


def test_live_stream_has_its_own_queue_and_never_owns_composer_outputs(live_app):
    app, funcs, _ = live_app
    app.validate_queue_settings()
    live, turn = funcs["observe_camera"], funcs["run_turn"]
    assert live.concurrency_id != turn.concurrency_id
    assert live.outputs == []
    assert live.inputs[0] not in turn.outputs
    assert funcs["live_status"].queue is False
    assert funcs["camera_lifecycle"].queue is False


def test_control_token_invalidates_frames_and_late_stream_cannot_restart_camera(live_app):
    _, funcs, pipe = live_app
    buffer = LiveVisionBuffer()
    control, observe = funcs["camera_lifecycle"].fn, funcs["observe_camera"].fn
    control("on:track-one", buffer, "s")
    epoch = buffer.epoch
    control("on:track-one", buffer, "s")
    assert buffer.epoch == epoch
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    observe(frame, buffer, "s", "on:track-one")
    assert pipe.observe_live.call_count == 1
    control("off:track-one", buffer, "s")
    observe(frame, buffer, "s", "on:track-one")
    assert pipe.observe_live.call_count == 1 and not buffer.enabled
    control("on:track-two", buffer, "s")
    observe(frame, buffer, "s", "on:track-one")
    assert pipe.observe_live.call_count == 1


def test_send_uses_recent_live_snapshot_and_ignores_old_upload_in_camera_mode(live_app):
    _, funcs, pipe = live_app
    buffer = Mock()
    observation = object()
    buffer.snapshot.return_value = observation
    last = list(funcs["run_turn"].fn("My day", "old.mp4", [], "s", 0, None, buffer, "camera"))[-1]
    assert pipe.calls[-1][0][1] is None
    assert pipe.calls[-1][1]["live_observation"] is observation
    buffer.snapshot.assert_called_once_with("s")
    assert "stay on" in last[4]


def test_upload_mode_does_not_accidentally_use_live_camera(live_app):
    _, funcs, pipe = live_app
    buffer = Mock()
    list(funcs["run_turn"].fn("My day", "uploaded.mp4", [], "s", 0, None, buffer, "clip"))
    assert pipe.calls[-1][0][1] == "uploaded.mp4"
    assert pipe.calls[-1][1] == {}
    buffer.snapshot.assert_not_called()


def test_new_conversation_clears_evidence_but_leaves_camera_enabled(live_app):
    _, funcs, _ = live_app
    buffer = LiveVisionBuffer("old", enabled=True)
    buffer.client_control = "on:track-one"
    old_epoch = buffer.epoch
    result = funcs["new_conversation"].fn("old", 1, None, buffer)
    assert buffer.enabled and buffer.epoch > old_epoch
    assert buffer.session_id == result[5] != "old"
    assert buffer.snapshot(result[5]) is None
    assert buffer.client_control == "on:track-one"


def test_timer_clears_old_display_and_never_runs_inference(live_app):
    _, funcs, pipe = live_app
    buffer = LiveVisionBuffer()
    html, state = funcs["live_status"].fn(buffer, "s")
    assert "Camera off" in html and state["label"] is None
    pipe.observe_live.assert_not_called()


def test_live_tag_requires_available_evidence_and_escapes_labels():
    assert "Joy" not in live_emotion_html({"status": "off", "label": "joy"})
    assert "&lt;script&gt;" in live_emotion_html({"status": "ready", "available": True, "label": "<script>"})
    assert "Face not in view" in live_emotion_html({"status": "no_face"})
