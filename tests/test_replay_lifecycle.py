"""Controlled scheduling probes with Gradio's actual per-browser session state."""
import asyncio
import inspect
import threading

import pytest


class FixturePipeline:
    def status(self):
        return {"ready": True, "errors": [], "components": {"fixture": "not inference"}}

    def cancel(self, session, turn=None):
        return False


@pytest.fixture
def replay_ui(tmp_path, monkeypatch):
    import checkin.app as module
    monkeypatch.setattr(module, "load_replay_rows", lambda home: [{"text": "OLD REPLAY", "video_path": None}])
    app = module.build_app(tmp_path, pipeline=FixturePipeline())
    funcs = {f.name: f for f in app.fns.values() if f.fn is not None}
    yield app, funcs, module
    app.close()


def test_replay_and_admission_are_cancellable_and_loader_serializes_with_send(replay_ui):
    app, funcs, _ = replay_ui
    app.validate_queue_settings()
    admission, loader, send = (funcs[name] for name in ("begin_replay", "select_replay", "run_turn"))
    assert admission.queue and loader.queue
    assert inspect.isgeneratorfunction(admission.fn) and inspect.isgeneratorfunction(loader.fn)
    assert loader.concurrency_id == send.concurrency_id == "gpu"
    assert admission.concurrency_id != "gpu"
    assert not loader.inputs[0].stateful  # Capture ticket at queue submission.
    for name in ("new_conversation", "stop_current", "run_turn", "abandon_replay"):
        trigger = funcs[name].targets
        cancel = next(fn for fn in app.fns.values() if fn.cancels and fn.targets == trigger)
        assert {admission._id, loader._id} <= set(cancel.cancels)


@pytest.mark.parametrize("boundary", ["before_start", "during_load"])
def test_reset_preserves_fresh_composer_and_invalidates_fixed_ticket(replay_ui, monkeypatch, boundary):
    from gradio.state_holder import SessionState
    app, funcs, module = replay_ui
    async def probe():
        state = SessionState(app)
        reset, admission, loader = (funcs[name] for name in ("new_conversation", "begin_replay", "select_replay"))
        state[reset.inputs[0]._id] = "old-session"
        state[reset.inputs[1]._id] = 0
        admitted = await app.process_api(admission, ["0", state[reset.inputs[2]._id]["epoch"], None], state=state, session_hash="browser-A", simple_format=True)
        ticket = admitted["data"][0].model_dump()
        guard = state[reset.inputs[2]._id]
        assert guard["owner"] == ticket["id"]
        admitted["iterator"].aclose()
        entered, release = threading.Event(), threading.Event()
        if boundary == "during_load":
            original = module.replay_values
            def delayed(row):
                entered.set()
                assert release.wait(3)
                return original(row)
            monkeypatch.setattr(module, "replay_values", delayed)
            first = await app.process_api(loader, [ticket, None], state=state, session_hash="browser-A", simple_format=True)
            pending = asyncio.create_task(app.process_api(loader, [ticket, None], state=state, iterator=first["iterator"], session_hash="browser-A", simple_format=True))
            assert await asyncio.to_thread(entered.wait, 2)
        reset_result = await app.process_api(reset, [None] * len(reset.inputs), state=state, session_hash="browser-A", simple_format=True)
        assert reset_result["data"][7]["value"] == ""
        assert state[reset.inputs[2]._id] is guard and guard["owner"] is None
        if boundary == "during_load":
            release.set()
            stale = await pending
        else:
            stale = await app.process_api(loader, [ticket, None], state=state, session_hash="browser-A", simple_format=True)
        assert all(value == {"__type__": "update"} for value in stale["data"])
        stale["iterator"].aclose()
    asyncio.run(probe())


def test_browser_states_isolate_guards_and_old_loader_cannot_unlock_new_owner(replay_ui):
    from gradio.state_holder import SessionState
    app, funcs, _ = replay_ui
    guard_component = funcs["select_replay"].inputs[1]
    left, right = SessionState(app), SessionState(app)
    guard_a, guard_b = left[guard_component._id], right[guard_component._id]
    assert guard_a is not guard_b
    ticket_a = next(funcs["begin_replay"].fn("0", guard_a["epoch"], guard_a))[0]
    assert guard_b["owner"] is None
    old = funcs["select_replay"].fn(ticket_a, guard_a)
    next(old)
    funcs["new_conversation"].fn("session", 0, guard_a)
    ticket_b = next(funcs["begin_replay"].fn("0", guard_a["epoch"], guard_a))[0]
    assert all(value == {"__type__": "update"} for value in next(old))
    old.close()
    assert guard_a["owner"] == ticket_b["id"]
    assert guard_b["owner"] is None


@pytest.mark.parametrize("cancel_action", ["stop_current", "abandon_replay", "run_turn"])
@pytest.mark.parametrize("close_first", [True, False])
def test_cancellation_restores_or_transfers_owned_controls_in_either_order(replay_ui, cancel_action, close_first):
    _, funcs, _ = replay_ui
    guard = {"owner": None, "epoch": "initial"}
    ticket = next(funcs["begin_replay"].fn("0", guard["epoch"], guard))[0]
    stream = funcs["select_replay"].fn(ticket, guard)
    next(stream)
    if close_first:
        stream.close()
    if cancel_action == "run_turn":
        send = funcs[cancel_action].fn("A new draft", None, [], "session", 0, guard)
        result = next(send)
        assert result[8]["interactive"] is False  # Send owns the composer now.
        send.close()
    elif cancel_action == "stop_current":
        result = funcs[cancel_action].fn("session", 0, guard)
        assert all(value["interactive"] is True for value in result[3:-1])
    else:
        result = funcs[cancel_action].fn(guard)
        assert all(value["interactive"] is True for value in result[3:-1])
    assert guard["owner"] is None
    if not close_first:
        assert all(value == {"__type__": "update"} for value in next(stream))
        stream.close()


def test_failed_replay_restores_controls_without_overwriting_draft(replay_ui, monkeypatch):
    _, funcs, module = replay_ui
    def broken(row):
        raise ValueError("controlled preparation failure")
    monkeypatch.setattr(module, "replay_values", broken)
    guard = {"owner": None, "epoch": "initial"}
    ticket = next(funcs["begin_replay"].fn("0", guard["epoch"], guard))[0]
    result = list(funcs["select_replay"].fn(ticket, guard))[-1]
    assert "could not load" in result[0]
    assert all(update == {"interactive": True, "__type__": "update"} for update in result[3:])
    assert guard["owner"] is None


@pytest.mark.parametrize("action", ["new_conversation", "stop_current", "abandon_replay", "run_turn"])
def test_admission_captured_before_reset_stop_or_edit_cannot_take_ownership_later(replay_ui, action):
    _, funcs, _ = replay_ui
    guard = {"owner": None, "epoch": "captured-before-action"}
    if action in {"new_conversation", "stop_current"}:
        funcs[action].fn("session", 0, guard)
    elif action == "run_turn":
        turn = funcs[action].fn("Fresh check-in", None, [], "session", 0, guard)
        next(turn)
        turn.close()
    else:
        funcs[action].fn(guard)
    assert guard["epoch"] != "captured-before-action"
    late = list(funcs["begin_replay"].fn("0", "captured-before-action", guard))
    assert len(late) == 1
    assert all(value == {"__type__": "update"} for value in late[0])
    assert guard["owner"] is None


def test_successful_replay_commits_once_and_restores_dropdown_and_composer(replay_ui):
    _, funcs, _ = replay_ui
    guard = {"owner": None, "epoch": "initial"}
    admission = list(funcs["begin_replay"].fn("0", "initial", guard))[-1]
    assert all(update["interactive"] is False for update in admission[4:])
    result = list(funcs["select_replay"].fn(admission[0], guard))[-1]
    assert result[3] == {"value": None, "interactive": True, "__type__": "update"}
    assert result[4] == {"value": "OLD REPLAY", "interactive": True, "__type__": "update"}
    assert result[-1]["interactive"] is True
    assert guard["owner"] is None
