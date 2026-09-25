"""CPU contract fixtures only; these do not estimate webcam emotion accuracy."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest

from checkin.data import VideoProcessor, l2_normalize
from checkin.live_vision import (FRESH_SECONDS, LiveSample, LiveVisionBuffer,
    aggregate_samples, expected_text_metadata, extract_live_face)
from checkin.pipeline import CheckInPipeline
from checkin.settings import LABELS


def face(x=20.3, y=10.7, width=60.5, height=70.2, score=.95):
    return np.array([x,y,width,height]+[0.0]*10+[score], dtype=np.float32)


class Detector:
    def __init__(self, faces=None):
        self.faces = np.array([face()]) if faces is None else faces
        self.calls = 0
    def setInputSize(self, size):
        self.size = size
    def detect(self, image):
        self.calls += 1
        return None, self.faces


@pytest.mark.parametrize("width", [639,640,641,1281])
@pytest.mark.parametrize("box", [face(), face(-10.2,-3.1,100.7,90.4)])
def test_live_geometry_bitwise_matches_file_preprocessing(tmp_path, monkeypatch, width, box):
    rgb = np.arange(111*width*3,dtype=np.uint8).reshape(111,width,3)
    bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    class Capture:
        def get(self, flag):
            return 30 if flag == cv2.CAP_PROP_FPS else 8
        def isOpened(self): return True
        def set(self, *args): pass
        def read(self): return True,bgr.copy()
        def release(self): pass
    monkeypatch.setattr(cv2,"VideoCapture",lambda *args:Capture())
    processor=object.__new__(VideoProcessor)
    processor.frames,processor.min_valid=8,6
    processor.detector=Detector(np.array([box]))
    path=tmp_path/"fixture.mp4"
    path.touch()
    crops,quality=processor.process(path)
    live,_,score,reason=extract_live_face(rgb,Detector(np.array([box])))
    assert quality["available"] and reason == "ok" and score == pytest.approx(.95)
    assert len(crops)==8 and all(np.array_equal(live,crop) for crop in crops)


def test_detector_face_size_threshold_and_ambiguity():
    image=np.zeros((240,1280,3),dtype=np.uint8)
    assert extract_live_face(image,Detector(np.array([face(width=20,height=20)])))[-1]=="ok"
    assert extract_live_face(image,Detector(np.array([face(width=19.99,height=20)])))[-1]=="no_face"
    assert extract_live_face(image,Detector(np.array([face(),face()])))[-1]=="multiple_faces"
    assert extract_live_face(image,Detector(np.array([face(),face(width=10,height=10)])))[-1]=="ok"
    detector=Detector();detector.faces=None
    assert extract_live_face(image,detector)[-1]=="no_face"


@pytest.mark.parametrize("frame", [None,np.zeros((10,10)),np.zeros((10,10,4),dtype=np.uint8),np.zeros((10,10,3))])
def test_invalid_camera_frames_never_reach_detector(frame):
    detector=Detector()
    assert extract_live_face(frame,detector)[-1]=="invalid_frame"
    assert detector.calls==0


def test_pool_raw_embeddings_then_normalize_and_require_six_samples():
    a=np.zeros(1408,dtype=np.float32);a[0]=10
    b=np.zeros(1408,dtype=np.float32);b[1]=1
    samples=[LiveSample(float(i),"utc","ok",tuple(a if i<3 else b),(.1,.1,.2,.2),.9) for i in range(6)]
    assert aggregate_samples(samples[:5])[0] is None
    feature,quality=aggregate_samples(samples)
    np.testing.assert_allclose(feature,l2_normalize((a+b)/2))
    assert quality["valid_frame_fraction"]==.75
    absent=LiveSample(6.,"utc","no_face")
    assert aggregate_samples(samples+[absent])[1]["reason"]=="no_face"
    changed=replace(samples[-1],box=(.8,.8,.1,.1))
    assert aggregate_samples(samples[:-1]+[changed])[1]["reason"]=="track_change"


@pytest.fixture
def live_pipe(tmp_path,monkeypatch):
    clock=[100.0]
    monkeypatch.setattr("checkin.live_vision.time.monotonic",lambda:clock[0])
    pipe=CheckInPipeline(tmp_path,device="cpu")
    pipe.processor=SimpleNamespace(detector=Detector(),process=lambda _: ([],{"available":False,"reason":"missing_video"}))
    features=np.zeros((1,1408),dtype=np.float32);features[0,0]=2
    pipe.vision_encoder=SimpleNamespace(encode_faces=lambda faces:features.copy())
    pipe.text_encoder=SimpleNamespace(encode=lambda texts:np.ones((len(texts),1024),dtype=np.float32),
        tokenizer=SimpleNamespace(encode=lambda *a,**k:[1,2]),max_length=128)
    pipe.heads={name:name for name in ("vision","text","fusion")}
    monkeypatch.setattr(pipe,"_load",lambda:None)
    monkeypatch.setattr(pipe,"_load_vision",lambda:None)
    def predict(model,values):
        p=np.zeros(7,dtype=np.float32);p[4 if model=="vision" else 1 if model=="fusion" else 0]=1
        return p
    monkeypatch.setattr(pipe,"_predict",predict)
    monkeypatch.setattr(pipe.generator,"stream",lambda *a:iter(["Fixture response"]))
    return pipe,clock


def populate(pipe,clock,buffer,n=8):
    image=np.zeros((160,200,3),dtype=np.uint8)
    statuses=[]
    for _ in range(n):
        statuses.append(pipe.observe_live_frame(image,buffer,buffer.session_id)[1])
        clock[0]+=.5
    return statuses


def test_bounded_live_vision_warms_without_text_or_generator(live_pipe,monkeypatch):
    pipe,clock=live_pipe
    monkeypatch.setattr(pipe.text_encoder,"encode",lambda *a:pytest.fail("Idle updates must not encode text"))
    monkeypatch.setattr(pipe.generator,"stream",lambda *a:pytest.fail("Idle updates must not generate"))
    buffer=LiveVisionBuffer("s",enabled=True)
    reports=populate(pipe,clock,buffer,30)
    assert all(r["label"] is None for r in reports[:5])
    assert reports[5]["label"]=="joy" and reports[-1]["available"]
    assert len(buffer.samples)==8 and all(isinstance(s.feature,tuple) for s in buffer.samples)
    assert buffer.observation.feature[0]==1.0
    assert buffer.current_status("s")["score_semantics"]=="uncalibrated_softmax"
    assert buffer.snapshot("other") is None


def test_off_expiry_and_missing_face_revoke_snapshots(live_pipe):
    pipe,clock=live_pipe;buffer=LiveVisionBuffer("s",enabled=True)
    populate(pipe,clock,buffer)
    saved=buffer.snapshot("s")
    assert saved is not buffer.observation and saved.valid_for("s")
    pipe.processor.detector.faces=None
    _,report=pipe.observe_live_frame(np.zeros((160,200,3),dtype=np.uint8),buffer,"s")
    assert report["status"]=="no_face" and report["label"] is None
    assert buffer.snapshot("s") is None and not saved.valid_for("s")
    buffer.clear()
    assert not buffer.enabled and not buffer.samples and buffer.current_status("s")["status"]=="off"
    pipe.processor.detector.faces=np.array([face()]);buffer.start("new")
    populate(pipe,clock,buffer)
    observation=buffer.observation
    clock[0]=observation.received_monotonic+FRESH_SECONDS-1e-5
    assert buffer.snapshot("new") is not None
    clock[0]+=1e-5
    assert buffer.snapshot("new") is None
    assert buffer.current_status("new")["status"]=="stale" and not buffer.samples


def test_deepcopy_buffer_has_independent_revocation(live_pipe):
    pipe,clock=live_pipe;buffer=LiveVisionBuffer("s",enabled=True)
    populate(pipe,clock,buffer)
    copied=deepcopy(buffer)
    buffer.clear()
    assert copied.snapshot("s") is not None and buffer.snapshot("s") is None
    copied.clear()
    assert copied.observation is None


@pytest.mark.parametrize("reset", ["off","new_session","restart"])
def test_reset_during_encoder_work_cannot_repopulate_camera(live_pipe,monkeypatch,reset):
    pipe,clock=live_pipe;buffer=LiveVisionBuffer("s",enabled=True)
    entered,finish=threading.Event(),threading.Event()
    original=pipe.vision_encoder.encode_faces
    def blocked(faces):
        entered.set();assert finish.wait(2);return original(faces)
    monkeypatch.setattr(pipe.vision_encoder,"encode_faces",blocked)
    reports=[]
    worker=threading.Thread(target=lambda:reports.append(pipe.observe_live_frame(np.zeros((160,200,3),dtype=np.uint8),buffer,"s")[1]))
    worker.start();assert entered.wait(2)
    buffer.clear(enabled=reset!="off",session_id="new" if reset=="new_session" else "s")
    finish.set();worker.join(2)
    assert not worker.is_alive() and reports[0]["status"]=="discarded"
    assert buffer.samples==[] and buffer.observation is None and not pipe._lock.locked()


def test_shared_lock_drops_camera_work_without_affecting_turn(live_pipe,monkeypatch):
    pipe,clock=live_pipe;buffer=LiveVisionBuffer("s",enabled=True)
    stream=pipe.stream("Hi",session_id="s")
    assert next(stream)["type"]=="state"
    owner=pipe._active_turn
    _,report=pipe.observe_live_frame(np.zeros((160,200,3),dtype=np.uint8),buffer,"s")
    assert report["status"]=="busy" and pipe._active_turn is owner and not owner.cancelled.is_set()
    assert pipe.processor.detector.calls==0 and not buffer.samples
    stream.close()
    assert not pipe._lock.locked()


def test_encoder_error_releases_lock_and_next_frame_recovers(live_pipe,monkeypatch):
    pipe,clock=live_pipe;buffer=LiveVisionBuffer("s",enabled=True)
    good=pipe.vision_encoder.encode_faces
    monkeypatch.setattr(pipe.vision_encoder,"encode_faces",lambda _:(_ for _ in ()).throw(RuntimeError("fixture fault")))
    report=pipe.observe_live(np.zeros((160,200,3),dtype=np.uint8),buffer,"s")
    assert report["status"]=="error" and not pipe._lock.locked()
    monkeypatch.setattr(pipe.vision_encoder,"encode_faces",good)
    clock[0]+=.5
    assert populate(pipe,clock,buffer)[-1]["status"]=="ready"


def test_send_waits_for_existing_camera_work_and_prevents_more_idle_work(live_pipe,monkeypatch):
    pipe,clock=live_pipe;buffer=LiveVisionBuffer("s",enabled=True)
    entered,finish=threading.Event(),threading.Event()
    original=pipe.vision_encoder.encode_faces
    def blocked(faces):
        entered.set();assert finish.wait(3);return original(faces)
    monkeypatch.setattr(pipe.vision_encoder,"encode_faces",blocked)
    image=np.zeros((160,200,3),dtype=np.uint8)
    camera=threading.Thread(target=lambda:pipe.observe_live_frame(image,buffer,"s"))
    camera.start();assert entered.wait(2)
    stream=pipe.stream("User's check-in",session_id="s")
    events=[]
    sender=threading.Thread(target=lambda:events.append(next(stream)))
    sender.start()
    for _ in range(1000):
        if pipe._pending_turn is not None:break
        sender.join(.001)
    assert pipe._pending_turn is not None
    clock[0]+=.5
    assert pipe.observe_live_frame(image,buffer,"s")[1]["status"]=="busy"
    duplicate=pipe.stream("Duplicate",session_id="s")
    with pytest.raises(RuntimeError,match="Another turn"):
        next(duplicate)
    finish.set();camera.join(2);sender.join(2)
    assert not camera.is_alive() and not sender.is_alive()
    assert events[0]["type"]=="state" and pipe._pending_turn is None and not pipe._live_busy
    assert pipe.cancel("s")
    assert next(stream)["state"]["response"]["status"]=="cancelled"
    stream.close()
    assert not pipe._lock.locked()


def test_stop_pending_send_is_scoped_and_does_not_release_live_owner(live_pipe,monkeypatch):
    pipe,clock=live_pipe;buffer=LiveVisionBuffer("s",enabled=True)
    entered,finish=threading.Event(),threading.Event()
    original=pipe.vision_encoder.encode_faces
    def blocked(faces):
        entered.set();assert finish.wait(3);return original(faces)
    monkeypatch.setattr(pipe.vision_encoder,"encode_faces",blocked)
    monkeypatch.setattr(pipe,"classify",lambda *a,**k:pytest.fail("Cancelled pending Send must not classify"))
    monkeypatch.setattr(pipe.generator,"stream",lambda *a:pytest.fail("Cancelled pending Send must not generate"))
    image=np.zeros((160,200,3),dtype=np.uint8)
    camera=threading.Thread(target=lambda:pipe.observe_live_frame(image,buffer,"s"))
    camera.start();assert entered.wait(2)
    events=[]
    sender=threading.Thread(target=lambda:events.extend(pipe.stream("Hello",session_id="s",turn_id="2")))
    sender.start()
    for _ in range(1000):
        if pipe._pending_turn is not None:break
        sender.join(.001)
    assert pipe._pending_turn is not None
    assert not pipe.cancel("other","2") and not pipe.cancel("s","1")
    assert pipe.cancel("s","2")
    sender.join(1)
    assert not sender.is_alive() and len(events)==1 and events[0]["type"]=="cancelled"
    assert "state" not in events[0] and pipe._pending_turn is None and pipe._active_turn is None
    assert pipe._lock.locked() and pipe._live_busy
    assert not pipe.cancel("s","2")
    buffer.clear()
    finish.set();camera.join(2)
    assert not camera.is_alive() and not pipe._lock.locked() and not buffer.samples


def test_live_fusion_and_file_precedence(live_pipe):
    pipe,clock=live_pipe;buffer=LiveVisionBuffer("s",enabled=True)
    populate(pipe,clock,buffer)
    state=pipe.classify("Hello",None,"s","1",live_observation=buffer.snapshot("s"))
    assert state["emotion"]["source"]=="fusion" and state["modalities"]["vision_label"]=="joy"
    assert state["input"]["video_present"] and state["input"]["camera_age_at_send_seconds"]==.5
    assert state["vision"]["input_kind"]=="live_camera"
    # A supplied clip remains the selected modality, even when live evidence exists.
    state=pipe.classify("Hello","fixture.mp4","s","2",live_observation=buffer.snapshot("s"))
    assert state["emotion"]["source"]=="text_fallback" and state["vision"]["reason"]=="missing_video"


@pytest.mark.parametrize("camera_off",[False,True])
def test_fresh_at_send_survives_cold_load_but_camera_off_revokes(live_pipe,monkeypatch,camera_off):
    pipe,clock=live_pipe;buffer=LiveVisionBuffer("s",enabled=True)
    populate(pipe,clock,buffer)
    snapshot=buffer.snapshot("s")
    def load():
        clock[0]+=10
        if camera_off:buffer.clear()
    monkeypatch.setattr(pipe,"_load",load)
    events=list(pipe.stream("Hello",session_id="s",live_observation=snapshot))
    state=events[0]["state"]
    assert state["emotion"]["source"]==("text_fallback" if camera_off else "fusion")
    assert events[-1]["state"]["response"]["status"]=="complete"


def test_pinned_text_metadata_without_constructing_text_encoder(monkeypatch):
    from checkin.encoders import TextEncoder,TEXT_SHA256,TEXT_PARAMETER_COUNT
    monkeypatch.setattr(TextEncoder,"__init__",lambda *a,**k:pytest.fail("No text model construction"))
    data=expected_text_metadata()
    assert data["sha256"]==TEXT_SHA256 and data["parameter_count"]==TEXT_PARAMETER_COUNT
    assert data["max_length"]==128 and data["feature_dim"]==1024
