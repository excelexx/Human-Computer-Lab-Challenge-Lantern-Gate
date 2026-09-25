"""Contract tests with explicit stubs; not model-quality or MELD evaluations."""
import numpy as np
import pytest
from checkin.pipeline import CheckInPipeline
from checkin.schema import Emotion

def _state():
    return {"schema_version":"1.0","session_id":"s","turn_id":"1","input":{},
        "emotion":{"label":"neutral","probabilities":dict(neutral=1.,surprise=0.,fear=0.,sadness=0.,joy=0.,disgust=0.,anger=0.),"source":"text_fallback"},
        "vision":{"available":False},"modalities":{},"modality_disagreement":False,
        "timing":{"classification_ms":1.,"first_token_ms":None,"completion_ms":None},
        "response":{"text":"","status":"pending","error":None}}

def test_schema_rejects_bad_distribution():
    with pytest.raises(ValueError):
        Emotion(label="neutral",probabilities={"neutral":1.},source="text_fallback")

def test_close_releases_lock(tmp_path,monkeypatch):
    pipe=CheckInPipeline(tmp_path,device="cpu")
    monkeypatch.setattr(pipe,"_load",lambda:None)
    monkeypatch.setattr(pipe,"classify",lambda *a,**k:_state())
    stream=pipe.stream("hello",session_id="s",turn_id=1)
    assert next(stream)["turn_id"]=="1"
    assert pipe._lock.locked()
    stream.close()
    assert not pipe._lock.locked()
    assert pipe._active_session is None

def test_cancel_is_session_scoped(tmp_path,monkeypatch):
    pipe=CheckInPipeline(tmp_path,device="cpu")
    monkeypatch.setattr(pipe,"_load",lambda:None)
    monkeypatch.setattr(pipe,"classify",lambda *a,**k:_state())
    monkeypatch.setattr(pipe.generator,"stream",lambda *a:iter(["one","two"]))
    stream=pipe.stream("hello",session_id="s",turn_id="1")
    next(stream)
    assert not pipe.cancel("other")
    assert pipe.cancel("s")
    event=next(stream)
    assert event["type"]=="done" and event["state"]["response"]["status"]=="cancelled"
    list(stream)
    assert not pipe._lock.locked()

def test_generation_error_preserves_state(tmp_path,monkeypatch):
    pipe=CheckInPipeline(tmp_path,device="cpu")
    monkeypatch.setattr(pipe,"_load",lambda:None)
    monkeypatch.setattr(pipe,"classify",lambda *a,**k:_state())
    def broken(*a):
        raise RuntimeError("local server unavailable")
        yield
    monkeypatch.setattr(pipe.generator,"stream",broken)
    events=list(pipe.stream("hello",session_id="s",turn_id="1"))
    assert events[-1]["type"]=="error"
    assert events[-2]["state"]["emotion"]["label"]=="neutral"
    assert events[-2]["state"]["response"]["status"]=="error"
    assert not pipe._lock.locked()
