"""Local UI contracts; fixtures simulate events, never claim model inference."""

from pathlib import Path

import pytest

from checkin.app import clean_history, emotion_html, load_replay_rows, resolve_home, status_message, video_path


class EventFixturePipeline:
    def __init__(self):
        self.calls = []
        self.closed = []
        self.cancel_calls = []
        self.cancel_turns = []

    def cancel(self, session_id, turn_id=None):
        self.cancel_calls.append(session_id)
        self.cancel_turns.append(turn_id)
        return True

    def status(self):
        return {"ready": True, "errors": [], "components": {"fixture": "not an ML model"}}

    def stream(self, text, clip, history, session_id, turn_id, **kwargs):
        self.calls.append({"text": text, "clip": clip, "history": history, "session": session_id, "turn": turn_id})
        state = {"emotion": {"label": "joy", "probabilities": {"joy": 1.0}}, "vision": {"available": bool(clip)}, "response": {"text": "Fixture response.", "status": "complete"}}
        try:
            yield {"type": "state", "state": state}
            yield {"type": "text_delta", "text": "Fixture "}
            yield {"type": "text_delta", "text": "response."}
            yield {"type": "done", "state": state}
        finally:
            self.closed.append(session_id)


@pytest.fixture
def ui(tmp_path):
    pytest.importorskip("gradio")
    from checkin.app import build_app

    pipeline = EventFixturePipeline()
    app = build_app(tmp_path, pipeline=pipeline)
    handler = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "run_turn")
    yield app, handler, pipeline
    app.close()


def test_history_is_copied_and_only_plain_conversation_is_kept():
    source = [{"role": "user", "content": "My day", "extra": "discard"}, {"role": "system", "content": "ignore"}, {"role": "assistant", "content": {"path": "private"}}, {"role": "assistant", "content": "  "}]
    result = clean_history(source)
    assert result == [{"role": "user", "content": "My day"}]
    result[0]["content"] = "changed"
    assert source[0]["content"] == "My day"


@pytest.mark.parametrize("value, expected", [
    (None, None), ("clip.mp4", "clip.mp4"), (Path("clip.mp4"), "clip.mp4"),
    ({"video": {"path": "clip.mp4"}}, "clip.mp4"), (("clip.mp4", None), "clip.mp4"),
])
def test_gradio_media_argument_forms(value, expected):
    assert video_path(value) == expected


def test_emotion_text_is_escaped_and_missing_input_is_not_neutral():
    assert "<script>" not in emotion_html({"emotion": {"label": "<script>alert(1)</script>"}})
    assert "&lt;script&gt;" in emotion_html({"emotion": {"label": "<script>alert(1)</script>"}})
    assert "Awaiting your line" in emotion_html({})
    assert "Neutral" not in emotion_html({})


def test_local_path_and_status(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKIN_HOME", str(tmp_path))
    assert resolve_home() == tmp_path.resolve()
    assert "Setup is incomplete" in status_message({"ready": False, "errors": ["missing weights"]})
    assert "Ready on this computer" in status_message({"ready": True})


def test_replay_only_offers_existing_test_media(tmp_path):
    import json

    (tmp_path / "manifests").mkdir()
    (tmp_path / "clip.mp4").touch()
    common = {"text": "Example", "video_path": "clip.mp4"}
    records = [{**common, "split": "train"}, {**common, "split": "test"}, {**common, "split": "test", "video_path": "missing.mp4"}]
    (tmp_path / "manifests/meld.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    rows = load_replay_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["split"] == "test"
    assert Path(rows[0]["video_path"]).is_absolute()


def test_app_capture_and_queue_are_local_and_serial(ui):
    import gradio as gr

    app, _, _ = ui
    camera = next(component for component in app.blocks.values() if isinstance(component, gr.Video))
    assert camera.sources == ["upload"]
    live_camera = next(component for component in app.blocks.values() if isinstance(component, gr.Image))
    assert live_camera.sources == ["webcam"]
    assert live_camera.streaming is True
    assert camera.include_audio is False
    assert camera.max_length == 20
    assert app.analytics_enabled is False
    send_handler = next(fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "run_turn")
    assert send_handler.concurrency_limit == 1
    assert send_handler.concurrency_id == "gpu"
    assert app.theme.body_background_fill == app.theme.body_background_fill_dark
    assert app.theme.body_text_color == app.theme.body_text_color_dark
    assert app.theme.input_background_fill == app.theme.input_background_fill_dark
    assert app.theme.block_label_text_color == app.theme.block_label_text_color_dark


def test_launch_allows_exact_external_replay_clips_only(tmp_path, monkeypatch):
    import json
    import checkin.app as module

    home = tmp_path / "outside-repository"
    (home / "manifests").mkdir(parents=True)
    clip = home / "clip.mp4"
    clip.touch()
    (home / "private.txt").write_text("not served", encoding="utf-8")
    (home / "manifests/meld.jsonl").write_text(json.dumps({
        "split": "test", "text": "Example", "video_path": str(clip)
    }), encoding="utf-8")
    captured = {}

    class LaunchRecorder:
        def launch(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "build_app", lambda _: LaunchRecorder())
    module.main(["--home", str(home)])
    assert captured["allowed_paths"] == [str(clip.resolve())]
    assert captured["server_name"] == "127.0.0.1"
    assert captured["share"] is False


def test_http_routes_serve_the_full_local_app(ui):
    from fastapi.testclient import TestClient

    app, _, _ = ui
    with TestClient(app.app) as client:
        page = client.get("/")
        config = client.get("/config")
    assert page.status_code == 200
    assert "Mara, keeper of Lantern Gate" in page.text
    assert config.status_code == 200
    settings = config.json()
    assert settings["version"] == "5.49.1"
    assert settings["analytics_enabled"] is False
    assert settings["enable_queue"] is True
    assert any(component["type"] == "video" for component in settings["components"])


def test_sample_click_submits_once_with_selected_line_and_preserves_camera_mode(ui):
    from checkin.scene import SAMPLE_LINES
    app, _, pipeline = ui
    handlers = [fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "submit_sample"]
    assert len(handlers) == len(SAMPLE_LINES) == 4
    for index, (handler, line) in enumerate(zip(handlers, SAMPLE_LINES)):
        guard = {"owner": "old-upload", "epoch": "old", "upload_payload": {"path": "old.mp4"}}
        result = list(handler(line, "stale.mp4", [], f"session-{index}", 0, guard, None, "camera"))
        assert result[0][9]["value"] == line
        assert pipeline.calls[-1]["text"] == line
        assert pipeline.calls[-1]["clip"] is None
        assert guard["owner"] is None and "upload_payload" not in guard
        assert "turn_owner" not in guard
    assert len(pipeline.calls) == 4


def test_sample_stream_closure_releases_turn_and_closes_backend(ui):
    app, _, pipeline = ui
    handler = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "submit_sample")
    guard = {"owner": None, "epoch": "original"}
    stream = handler("Oh, fantastic.", None, [], "session", 0, guard, None, "camera")
    next(stream)
    next(stream)
    assert guard["turn_owner"]
    stream.close()
    assert "turn_owner" not in guard
    assert pipeline.closed == ["session"]


def test_three_completed_replies_update_examples_and_emit_one_game_action(ui):
    from checkin import quest
    app, handler, pipeline = ui
    guard = {"owner": None, "epoch": "original"}
    history = []
    for turn, line in enumerate(["Oh, fantastic.", "The sea stairs, then.", "Lead the way."]):
        results = list(handler(line, None, history, "session", turn, guard))
        assert results[0][-6]["interactive"] is False
        final = results[-1]
        history = final[1]
        assert guard["quest"]["completed"] == turn + 1
        assert final[-6]["value"] == quest.options(guard["quest"])[0]
    assert guard["quest"]["route"] == "stairs"
    assert 'data-phase="depart"' in final[-1]
    assert final[6]["interactive"] is False
    list(handler("Lead the way.", None, history, "session", 3, guard))
    assert len(pipeline.calls) == 3


def test_stale_sample_does_not_turn_into_a_different_option(ui):
    app, handler, pipeline = ui
    guard = {"owner": None, "epoch": "original"}
    first = list(handler("Oh, fantastic.", None, [], "session", 0, guard))[-1]
    sample = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "submit_sample")
    result = list(sample("Oh, fantastic.", None, first[1], "session", 1, guard, None, "camera"))[-1]
    assert len(pipeline.calls) == 1
    assert "new example replies" in result[4]
    assert guard["quest"]["completed"] == 1


def test_cancelled_or_failed_response_cannot_advance_quest(ui):
    _, handler, pipeline = ui
    guard = {"owner": None, "epoch": "original"}
    run = handler("Oh, fantastic.", None, [], "session", 0, guard)
    next(run); next(run); run.close()
    assert guard["quest"]["completed"] == 0
    def failed(*args, **kwargs):
        yield {"type": "error", "error": "fixture failure"}
    pipeline.stream = failed
    list(handler("Oh, fantastic.", None, [], "session", 1, guard))
    assert guard["quest"]["completed"] == 0


def test_streams_keep_sessions_and_prior_history_separate(ui):
    _, handler, pipeline = ui
    prior = [{"role": "user", "content": "Earlier A"}, {"role": "assistant", "content": "Earlier reply"}]
    run_a = handler("Current A", "a.mp4", prior, "session-a", 2)
    run_b = handler("Current B", None, [], "session-b", 0)
    next(run_a)
    next(run_b)
    result_a = list(run_a)[-1]
    result_b = list(run_b)[-1]
    by_session = {call["session"]: call for call in pipeline.calls}
    assert by_session["session-a"]["history"] == prior
    assert by_session["session-b"]["history"] == []
    assert by_session["session-a"]["turn"] == "3"
    assert by_session["session-b"]["turn"] == "1"
    assert result_a[0][-1] == {"role": "assistant", "content": "Fixture response."}
    assert result_b[0][0]["content"] == "Current B"
    assert "Earlier A" not in str(result_b)
    assert prior == [{"role": "user", "content": "Earlier A"}, {"role": "assistant", "content": "Earlier reply"}]
    assert sorted(pipeline.closed) == ["session-a", "session-b"]


def test_empty_submission_does_not_start_pipeline(ui):
    _, handler, pipeline = ui
    outputs = list(handler("   ", None, [], "session", 0))
    assert len(outputs) == 1
    assert pipeline.calls == []
    assert outputs[-1][5] == 0


def test_oversized_submission_is_rejected_before_inference(ui):
    _, handler, pipeline = ui
    result = list(handler("a" * 4001, None, [], "session", 0))[-1]
    assert pipeline.calls == []
    assert "4,000" in result[4]
    assert result[5] == 0


def test_cancelled_turn_closes_underlying_stream(ui):
    _, handler, pipeline = ui
    generator = handler("Hello", None, [], "cancelled-session", 0)
    next(generator)
    next(generator)
    generator.close()
    assert pipeline.closed == ["cancelled-session"]


def test_stop_requests_cooperative_cancellation_for_current_session(ui):
    app, _, pipeline = ui
    stop = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "stop_current")
    result = stop("owned-session")
    assert pipeline.cancel_calls == ["owned-session"]
    assert "Stopping after the current processing step" in result[0]


def test_new_conversation_cancels_old_session_and_resets_all_turn_state(ui):
    app, _, pipeline = ui
    reset = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "new_conversation")
    result = reset("old-session")
    assert pipeline.cancel_calls == ["old-session"]
    assert result[0] == result[1] == []
    assert result[3] == {}
    assert result[5] != "old-session"
    assert result[6] == 0
    assert result[8]["value"] is None
    assert result[8]["interactive"] is True


def test_pipeline_error_is_explicit_and_does_not_invent_a_tag(ui):
    _, handler, pipeline = ui

    def failed(*args):
        yield {"type": "error", "error": "Required checkpoint is missing"}

    pipeline.stream = failed
    result = list(handler("Hello", None, [], "session", 0))[-1]
    assert "checkpoint is missing" in result[3]["error"]
    assert "Awaiting your line" in result[2]
    assert result[0] == [{"role": "user", "content": "Hello"}]


def test_completed_text_without_deltas_uses_nested_response(ui):
    _, handler, pipeline = ui

    def completed(*args):
        yield {"type": "done", "state": {"emotion": {"label": "neutral"}, "response": {"text": "A finished fixture response.", "status": "complete"}}}

    pipeline.stream = completed
    result = list(handler("Hello", None, [], "session", 0))[-1]
    assert result[0][-1]["content"] == "A finished fixture response."


def test_clip_is_retained_during_processing_and_cleared_only_after_turn(ui):
    _, handler, pipeline = ui
    results = list(handler("My day", "fresh.mp4", [], "session", 0))
    assert results[0][8] == {"interactive": False, "__type__": "update"}
    assert results[0][9] == {"interactive": False, "__type__": "update"}
    assert all("value" not in result[8] for result in results[:-1])
    assert pipeline.calls[0]["clip"] == "fresh.mp4"
    assert results[-1][8] == {"value": None, "interactive": True, "__type__": "update"}
    assert results[-1][9] == {"value": "", "interactive": True, "__type__": "update"}
    list(handler("Another day", results[-1][8]["value"], results[-1][1], "session", 1))
    assert pipeline.calls[-1]["clip"] is None


def test_failure_keeps_media_for_retry_and_restores_composer(ui):
    _, handler, pipeline = ui
    def failed(*args):
        yield {"type": "error", "error": "Busy"}
    pipeline.stream = failed
    last = list(handler("My day", "retry.mp4", [], "session", 0))[-1]
    assert last[8] == {"interactive": True, "__type__": "update"}
    assert last[9] == {"interactive": True, "__type__": "update"}
    assert "kept so you can retry" in last[4]


def test_stop_clears_consumed_clip_when_gradio_cancels_generator(ui):
    app, handler, _ = ui
    run = handler("My day", "consumed.mp4", [], "session", 0)
    next(run)
    next(run)
    run.close()
    stop = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "stop_current")
    result = stop("session")
    assert result[3]["value"] is None
    assert result[3]["interactive"] is True
    assert result[4]["value"] == ""


def test_stop_and_reset_scope_cancellation_to_the_submitted_turn(ui):
    app, _, pipeline = ui
    stop = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "stop_current")
    reset = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "new_conversation")
    stop("session", 3)
    reset("session", 4)
    assert pipeline.cancel_turns == ["3", "4"]


def test_cold_cancellation_displays_stop_without_inventing_emotion_or_history(ui):
    _, handler, pipeline = ui
    def cancelled(*args):
        yield {"type": "cancelled", "session_id": "session", "turn_id": "1", "phase": "before_classification", "reason": "Stopped"}
    pipeline.stream = cancelled
    result = list(handler("Hello", None, [], "session", 0))[-1]
    assert "Stopped before the emotion signal was ready" in result[4]
    assert "Awaiting your line" in result[2]
    assert result[3]["cancelled"] is True and "emotion" not in result[3]
    assert result[1] == []


def test_failed_retry_does_not_duplicate_the_user_message_in_model_history(ui):
    _, handler, pipeline = ui
    original = pipeline.stream
    def failed(*args):
        yield {"type": "error", "error": "Still stopping"}
    pipeline.stream = failed
    prior = [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Prior reply"}]
    failed_result = list(handler("Try this", None, prior, "session", 0))[-1]
    assert failed_result[1] == prior
    pipeline.stream = original
    list(handler("Try this", None, failed_result[1], "session", failed_result[5]))
    assert pipeline.calls[-1]["history"] == prior
    assert pipeline.calls[-1]["text"] == "Try this"


@pytest.mark.parametrize("error,partial,expected", [
    ("Please shorten your message: it exceeds the local model's token budget even without conversation history.", "", "Shorten it and retry"),
    ("The local response reached its length limit. Any partial response is preserved; try a shorter check-in.", "A partial response", "Try a shorter check-in"),
])
def test_model_length_errors_show_actionable_status_without_losing_state_or_draft(ui, error, partial, expected):
    _, handler, pipeline = ui
    state = {"emotion": {"label": "neutral"}, "vision": {"available": True},
             "response": {"text": partial, "status": "error", "error": error}}
    def failed(*args):
        yield {"type": "state", "state": state}
        if partial:
            yield {"type": "text_delta", "text": partial}
        yield {"type": "error", "error": error}
    pipeline.stream = failed
    prior = [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Prior reply"}]
    last = list(handler("Day " * 1000, "retry.mp4", prior, "session", 0))[-1]
    assert expected in last[4] and "draft and clip are kept" in last[4]
    assert last[3]["emotion"] == state["emotion"]
    assert error in last[3]["error"]
    assert last[1] == prior
    assert last[8] == {"interactive": True, "__type__": "update"}
    assert last[9] == {"interactive": True, "__type__": "update"}
    if partial:
        assert last[0][-1] == {"role": "assistant", "content": partial}
