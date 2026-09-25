"""Deterministic CPU scheduling probes; actual GPU evidence is recorded separately."""

import threading

import pytest

from checkin.pipeline import CheckInPipeline
from checkin.settings import LABELS


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    pipe = CheckInPipeline(tmp_path, device="cpu")
    monkeypatch.setattr(pipe, "_load", lambda: None)
    def classify(text, video, session, turn, **kwargs):
        return {"session_id": str(session), "turn_id": str(turn), "input": {"text": text, "video_present": bool(video)},
                "emotion": {"label": "neutral", "source": "text_fallback", "probabilities": {label: float(label == "neutral") for label in LABELS}},
                "vision": {"available": False}, "modalities": {}, "modality_disagreement": False,
                "timing": {"classification_ms": 1., "first_token_ms": None, "completion_ms": None},
                "response": {"status": "pending", "text": "", "error": None}}
    monkeypatch.setattr(pipe, "classify", classify)
    monkeypatch.setattr(pipe.generator, "stream", lambda *args: iter(["First", " second"]))
    return pipe


def next_terminal(stream):
    while True:
        event = next(stream)
        if event["type"] in {"done", "error", "cancelled"}:
            return event


def test_cold_cancel_has_no_fabricated_state_and_releases_before_terminal(pipeline, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    def load():
        entered.set()
        assert finish.wait(2)
    monkeypatch.setattr(pipeline, "_load", load)
    monkeypatch.setattr(pipeline, "classify", lambda *a, **k: pytest.fail("Cancelled load must not classify"))
    stream = pipeline.stream("Hello", session_id="cold", turn_id="1")
    events = []
    worker = threading.Thread(target=lambda: events.append(next(stream)))
    worker.start()
    assert entered.wait(2)
    assert not pipeline.cancel("cold", "0")
    assert pipeline.cancel("cold", "1")
    finish.set()
    worker.join(2)
    assert not worker.is_alive()
    assert events[0]["type"] == "cancelled" and "state" not in events[0]
    assert not pipeline._lock.locked() and pipeline._active_session is None
    stream.close()


def test_cancel_after_state_does_not_open_generator(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline.generator, "stream", lambda *args: pytest.fail("Generator request must not open"))
    stream = pipeline.stream("Hello", session_id="s", turn_id="1")
    assert next(stream)["type"] == "state"
    assert pipeline.cancel("s", "1")
    terminal = next(stream)
    assert terminal["type"] == "done"
    assert terminal["state"]["response"]["status"] == "cancelled"
    assert not pipeline._lock.locked()
    stream.close()


def test_cancel_during_classification_preserves_genuine_state(pipeline, monkeypatch):
    original = pipeline.classify
    def classify(*args, **kwargs):
        state = original(*args, **kwargs)
        assert pipeline.cancel("s", "1")
        return state
    monkeypatch.setattr(pipeline, "classify", classify)
    stream = pipeline.stream("Hello", session_id="s", turn_id="1")
    terminal = next(stream)
    assert terminal["type"] == "done"
    assert terminal["state"]["emotion"]["label"] == "neutral"
    assert terminal["state"]["response"]["status"] == "cancelled"
    assert not pipeline._lock.locked()
    stream.close()


def test_cancel_suspended_at_delta_never_requests_another_delta(pipeline, monkeypatch):
    closed = []
    def response(*args):
        try:
            yield "Partial"
            pytest.fail("No next delta after accepted cancellation")
        finally:
            closed.append(True)
    monkeypatch.setattr(pipeline.generator, "stream", response)
    stream = pipeline.stream("Hello", session_id="s", turn_id="1")
    next(stream)
    assert next(stream)["text"] == "Partial"
    assert pipeline.cancel("s", "1")
    terminal = next(stream)
    assert terminal["state"]["response"] == {"text": "Partial", "status": "cancelled", "error": None}
    assert closed == [True] and not pipeline._lock.locked()
    stream.close()


def test_cancel_at_exhaustion_wins_over_completion(pipeline, monkeypatch):
    def response(*args):
        yield "Real fixture output"
        assert pipeline.cancel("s", "1")
    monkeypatch.setattr(pipeline.generator, "stream", response)
    stream = pipeline.stream("Hello", session_id="s", turn_id="1")
    assert next_terminal(stream)["state"]["response"]["status"] == "cancelled"
    assert not pipeline._lock.locked()
    stream.close()


@pytest.mark.parametrize("end_old", ["close", "exhaust"])
def test_retained_terminal_and_old_finally_cannot_release_new_owner(pipeline, end_old):
    old = pipeline.stream("Old", session_id="same-session", turn_id="1")
    next(old)
    old_token = pipeline._active_turn
    assert next_terminal(old)["state"]["response"]["status"] == "complete"
    assert not pipeline._lock.locked()
    new = pipeline.stream("New", session_id="same-session", turn_id="2")
    next(new)
    assert not pipeline.cancel("same-session", "1")
    assert pipeline._active_turn is not old_token
    assert pipeline._cancel is not old_token.cancelled
    old_token.cancelled.set()  # Even a retained old token cannot cancel the new turn.
    old.close() if end_old == "close" else list(old)
    assert pipeline._lock.locked() and pipeline._active_session == "same-session"
    assert not pipeline._active_turn.cancelled.is_set()
    assert next_terminal(new)["state"]["response"]["status"] == "complete"
    new.close()


def test_busy_and_wrong_session_leave_active_ownership_intact(pipeline):
    first = pipeline.stream("First", session_id="a", turn_id="1")
    next(first)
    owner = pipeline._active_turn
    assert not pipeline.cancel("b", "1")
    second = pipeline.stream("Second", session_id="b", turn_id="1")
    with pytest.raises(RuntimeError, match="Another turn"):
        next(second)
    assert pipeline._active_turn is owner and not owner.cancelled.is_set()
    first.close()
    assert not pipeline._lock.locked()


def test_cleanup_failure_still_releases_before_error_and_allows_recovery(pipeline, monkeypatch):
    class BrokenClose:
        def __next__(self):
            raise StopIteration
        def close(self):
            raise RuntimeError("fixture close failure")
    monkeypatch.setattr(pipeline.generator, "stream", lambda *a: BrokenClose())
    stream = pipeline.stream("Hello", session_id="s", turn_id="1")
    terminal = next_terminal(stream)
    assert terminal["type"] == "error" and "close failure" in terminal["error"]
    assert not pipeline._lock.locked()
    stream.close()
    monkeypatch.setattr(pipeline.generator, "stream", lambda *a: iter(["Recovered"]))
    recovery = pipeline.stream("Next", session_id="other", turn_id="1")
    assert next_terminal(recovery)["state"]["response"]["status"] == "complete"
    recovery.close()


def test_generation_error_releases_before_terminal_even_without_consumer_close(pipeline, monkeypatch):
    def broken(*args):
        raise RuntimeError("fixture transport error")
        yield
    monkeypatch.setattr(pipeline.generator, "stream", broken)
    stream = pipeline.stream("Hello", session_id="s", turn_id="1")
    assert next_terminal(stream)["type"] == "error"
    assert not pipeline._lock.locked() and pipeline._active_session is None
    stream.close()


def test_accepted_cancel_during_transport_error_remains_cancelled(pipeline, monkeypatch):
    def broken(*args):
        assert pipeline.cancel("s", "1")
        raise RuntimeError("transport ended while stopping")
        yield
    monkeypatch.setattr(pipeline.generator, "stream", broken)
    stream = pipeline.stream("Hello", session_id="s", turn_id="1")
    assert next_terminal(stream)["state"]["response"]["status"] == "cancelled"
    stream.close()


@pytest.mark.parametrize("close_error", [False, True])
def test_cancellation_accepted_during_close_wins_and_releases_ownership(pipeline, monkeypatch, close_error):
    class CancelOnClose:
        def __next__(self):
            raise StopIteration

        def close(self):
            assert pipeline.cancel("s", "1")
            if close_error:
                raise RuntimeError("fixture close error after accepted cancellation")

    monkeypatch.setattr(pipeline.generator, "stream", lambda *args: CancelOnClose())
    stream = pipeline.stream("Hello", session_id="s", turn_id="1")
    terminal = next_terminal(stream)
    assert terminal["type"] == "done"
    assert terminal["state"]["response"] == {"text": "", "status": "cancelled", "error": None}
    assert not pipeline._lock.locked() and pipeline._active_session is None
    assert not pipeline.cancel("s", "1")
    stream.close()
