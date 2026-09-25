"""Explicit pipeline fixtures exercise the runner; they are not model evidence."""

import copy
import json

import numpy as np
import pytest

from checkin import interaction_backtest as module
from checkin.settings import LABELS
from checkin.train import file_sha256


@pytest.fixture
def declared(tmp_path, monkeypatch):
    rows = []
    for index in range(14):
        video = tmp_path / f"dev-{index}.mp4"
        video.write_bytes(f"fixture media {index}".encode())
        rows.append({"id": f"dev:{index}:0", "split": "dev", "text": f"Fixture message {index}",
                     "video_path": str(video), "media_sha256": file_sha256(video)})
    cache = {"ids": np.array([row["id"] for row in rows]), "quality": np.ones((14, 3), np.float32),
             "feature_identity": "fixture-contract", "sha256": "fixture-cache"}
    cache["quality"][0, 2] = 0
    monkeypatch.setattr(module, "load_cache", lambda *a: cache)
    monkeypatch.setattr(module, "read_manifest", lambda *a: rows)
    monkeypatch.setattr(module, "fingerprints", lambda *a: {"explicit_cpu_fixture": True})
    home, output = tmp_path / "home", tmp_path / "declared-run"
    protocol = module.prepare(home, output)
    return home, output, protocol, rows, cache


class PipelineFixture:
    def __init__(self, fault=None):
        self.calls, self.closed = [], []
        self.fault = fault

    def stream(self, text, video_path, history, session_id, turn_id):
        self.calls.append({"text": text, "video": video_path, "history": copy.deepcopy(history), "session": session_id, "turn": turn_id})
        available = bool(video_path)
        state = {"session_id": session_id, "turn_id": turn_id,
                 "input": {"text": text, "video_present": available},
                 "emotion": {"label": "neutral", "probabilities": {label: float(label == "neutral") for label in LABELS},
                             "source": "fusion" if available else "text_fallback"},
                 "vision": {"available": available}, "modalities": {}, "modality_disagreement": False,
                 "timing": {"classification_ms": 1., "first_token_ms": 2., "completion_ms": 3., "model_load_ms": 0.},
                 "response": {"text": "", "status": "streaming"}}
        event_ids = {"session_id": session_id, "turn_id": turn_id}
        try:
            if self.fault == "identity":
                event_ids["session_id"] = "another-session"
            if self.fault == "fallback" and not available:
                state["emotion"]["source"] = "fusion"
            yield {"type": "state", "state": state, **event_ids}
            if self.fault == "error":
                yield {"type": "error", "error": "fixture generator unavailable", **event_ids}
                return
            yield {"type": "text_delta", "text": "Fixture reply.", **event_ids}
            state["response"] = {"text": "Fixture reply.", "status": "complete"}
            if self.fault == "text":
                state["response"]["text"] = "Different final reply"
            if self.fault == "timing":
                state["timing"]["completion_ms"] = None
            if self.fault != "no_done":
                yield {"type": "done", "state": state, **event_ids}
        finally:
            self.closed.append((session_id, turn_id))


def test_protocol_predeclares_six_session_transitions_without_labels(declared):
    _, _, protocol, rows, cache = declared
    sessions = protocol["sessions"]
    assert len(sessions) == 6
    assert module.select_sessions(rows, cache) == sessions
    assert [session["mode"] for session in sessions] == ["missing_vision"] * 2 + ["fresh_real_pair"] * 2 + ["synthetic_mismatch"] * 2
    for session in sessions:
        first, second = session["turns"]
        assert first["text_source_id"] == first["video_source_id"]
        assert first["video_source_id"] != "dev:0:0"
        assert first["video_path"] != second["video_path"]
        assert first["history_roles"] == []
        assert second["history_roles"] == ["user", "assistant"]
        assert first["true_emotion_label"] is second["true_emotion_label"] is None
        if session["mode"] == "missing_vision":
            assert second["video_path"] is None
            assert second["text"] in module.FOLLOWUPS
        elif session["mode"] == "synthetic_mismatch":
            assert second["text_source_id"] != second["video_source_id"]
        else:
            assert second["text_source_id"] == second["video_source_id"]


def test_exact_12_turns_have_isolated_histories_actual_outputs_and_snapshot_events(declared):
    home, output, protocol, _, _ = declared
    fixture = PipelineFixture()
    report = module.run(home, output, pipeline=fixture)
    assert report["status"] == "complete"
    assert report["completed_turns"] == report["attempted_turns"] == len(fixture.calls) == 12
    assert report["quality_score"] is None
    assert len(fixture.closed) == 12
    for index in range(6):
        first, second = fixture.calls[2 * index:2 * index + 2]
        assert first["history"] == []
        assert second["history"] == [{"role": "user", "content": first["text"]}, {"role": "assistant", "content": "Fixture reply."}]
        assert second["video"] != first["video"]
        saved = report["sessions"][index]["turns"][0]
        assert saved["events"][0]["state"]["response"]["text"] == ""
        assert saved["checks"]["final_state"]["response"]["text"] == "Fixture reply."
    assert json.loads((output / "report.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("fault", ["identity", "error", "text", "timing", "no_done", "fallback"])
def test_contract_failures_are_retained_and_never_reported_as_completed(declared, fault):
    home, output, _, _, _ = declared
    report = module.run(home, output, pipeline=PipelineFixture(fault))
    assert report["status"] == "failed"
    assert report["completed_turns"] < 12
    assert report["sessions"][-1]["turns"][-1]["checks"]["errors"]
    assert (output / "report.json").is_file()


def test_changed_prompt_or_model_contract_is_rejected_before_any_pipeline_call(declared, monkeypatch):
    home, output, _, _, _ = declared
    monkeypatch.setattr(module, "fingerprints", lambda *a: {"different_prompt_or_head_hash": True})
    fixture = PipelineFixture()
    with pytest.raises(ValueError, match="changed after preparation"):
        module.run(home, output, pipeline=fixture)
    assert fixture.calls == []
    assert not (output / "run-started.json").exists()


def test_protocol_preparation_is_idempotent_but_a_started_run_cannot_be_hidden(declared):
    home, output, protocol, _, _ = declared
    assert module.prepare(home, output) == protocol
    module.run(home, output, pipeline=PipelineFixture())
    before = (output / "report.json").read_bytes()
    with pytest.raises(ValueError, match="already started"):
        module.run(home, output, pipeline=PipelineFixture())
    assert (output / "report.json").read_bytes() == before


def test_missing_declaration_and_changed_media_fail_before_execution(declared, tmp_path):
    home, output, protocol, _, _ = declared
    with pytest.raises(ValueError, match="Prepare"):
        module.run(home, tmp_path / "undeclared", pipeline=PipelineFixture())
    from pathlib import Path
    Path(protocol["sessions"][0]["turns"][0]["video_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="media bytes differ"):
        module.run(home, output, pipeline=PipelineFixture())
