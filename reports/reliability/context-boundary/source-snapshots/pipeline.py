"""Shared local inference path, with turn serialization and explicit fallbacks."""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass,field
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .data import VideoProcessor,l2_normalize
from .encoders import VisionEncoder,TextEncoder
from .generator import LocalGenerator
from .models import DIMENSIONS,load_head
from .schema import CheckInState,Emotion
from .settings import LABELS,runtime_home


@dataclass(frozen=True)
class _TurnOwner:
    session_id: str
    turn_id: str
    cancelled: threading.Event = field(default_factory=threading.Event)


class CheckInPipeline:
    def __init__(self, home=None, device=None):
        self.home=runtime_home(home)
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator=LocalGenerator()
        self._lock=threading.Lock()
        self._lifecycle_lock=threading.Lock()
        self._active_turn=None
        self._cancel=threading.Event()
        self._active_session=None
        self._loaded=False

    def cancel(self,session_id=None,turn_id=None):
        session_id=str(session_id) if session_id is not None else None
        turn_id=str(turn_id) if turn_id is not None else None
        with self._lifecycle_lock:
            owner=self._active_turn
            if owner is None or (session_id is not None and session_id!=owner.session_id) or (turn_id is not None and turn_id!=owner.turn_id):
                return False
            owner.cancelled.set()
            return True

    def _claim_turn(self,session_id,turn_id):
        with self._lifecycle_lock:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError("Another turn is still running. Wait or stop it first.")
            owner=_TurnOwner(str(session_id),str(turn_id))
            self._active_turn=owner
            # Compatibility aliases for diagnostics; ownership uses the record.
            self._active_session=owner.session_id
            self._cancel=owner.cancelled
            return owner

    def _release_turn(self,owner):
        with self._lifecycle_lock:
            cancelled=owner.cancelled.is_set()
            if self._active_turn is owner:
                self._active_turn=None
                self._active_session=None
                self._lock.release()
            return cancelled

    def status(self):
        from .audit import runtime_inventory
        required=["models/vision/yunet.onnx","models/vision/enet_b2_7.pt","models/text/pytorch_model.bin",
                  "models/text/config.json","models/text/tokenizer_config.json","models/text/spm.model",
                  "checkpoints/vision.pt","checkpoints/text.pt","checkpoints/fusion.pt"]
        missing=[name for name in required if not (self.home/name).is_file()]
        generator=self.generator.status()
        errors=["Missing artifact: "+name for name in missing]
        inventory,head_checks=runtime_inventory(self.home,loaded_heads=self.heads if self._loaded else None,verify_hashes=False)
        invalid_heads=[check for check in head_checks if check["status"]=="invalid"]
        errors.extend(f"Invalid {check['name']} classifier: {check['error']}" for check in invalid_heads)
        if not generator.get("available"):
            errors.append("Local response generator is not running. Start the local generator script.")
        return {"ready":not errors,"classification_ready":not missing and not invalid_heads,"loaded":self._loaded,
                "device":self.device,"running":self._active_session is not None,"errors":errors,"components":{"generator":generator},
                "artifact_validation":"loaded" if self._loaded else "pending_first_load",
                "parameter_budget":inventory["total_parameter_upper_bound"],
                "parameter_limit":inventory["parameter_limit"],
                "parameter_accounting_status":"invalid" if invalid_heads else "loaded_runtime" if self._loaded else "checkpoints_counted" if all(check["status"]=="verified" for check in head_checks) else "planned",
                "head_inventory":head_checks}

    def _load(self):
        if self._loaded:
            return
        torch.set_num_threads(4)
        self.processor=VideoProcessor(self.home/"models/vision/yunet.onnx")
        self.vision_encoder=VisionEncoder(self.home/"models/vision/enet_b2_7.pt",device=self.device)
        self.text_encoder=TextEncoder(self.home/"models/text",device=self.device,batch_size=1)
        self.heads={}
        identities=set()
        for stage in ("vision","text","fusion"):
            model,payload=load_head(self.home/f"checkpoints/{stage}.pt","cpu")
            if payload.get("stage")!=stage or payload.get("input_dim")!=DIMENSIONS[stage] or payload.get("labels")!=LABELS:
                raise ValueError(f"The {stage} checkpoint has an incompatible stage, dimension or label order; retrain it")
            self.heads[stage]=model
            if not payload.get("feature_identity"):
                raise ValueError("Classifier checkpoint has no feature provenance; retrain it")
            identities.add(payload["feature_identity"])
        from .features import feature_metadata,feature_identity
        actual=feature_identity(feature_metadata(self.vision_encoder,self.text_encoder,self.processor))
        if identities!={actual}:
            raise ValueError("Classifier checkpoints do not match current encoders/preprocessing; re-extract and retrain")
        if self.device.startswith("cuda"):
            # CUDA lazily initializes kernels for each accepted face-batch size.
            # Pay that one-time cost during model loading, not mid-conversation.
            # Warm-up images never enter a classifier or a user-visible state.
            blank=np.zeros((260,260,3),dtype=np.uint8)
            for count in range(self.processor.min_valid,self.processor.frames+1):
                self.vision_encoder.encode_faces([blank]*count)
            self.text_encoder.encode(["How was your day?"])
            self._synchronize()
        self._loaded=True

    def _synchronize(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @staticmethod
    def _predict(model,features):
        with torch.inference_mode():
            return torch.softmax(model(torch.from_numpy(np.asarray(features,dtype=np.float32)).reshape(1,-1)),dim=-1)[0].numpy()

    def classify(self,text,video_path,session_id,turn_id,received_at=None):
        received=received_at or datetime.now(timezone.utc).isoformat()
        self._load()
        self._synchronize()
        started=time.perf_counter()
        try:
            crops,quality=self.processor.process(video_path)
        except Exception as error:
            crops=[]
            quality={"available":False,"reason":"preprocess_error","error":str(error),
                     "valid_frame_fraction":0.0,"mean_detection_score":0.0,"speaker_attribution":"unverified_heuristic"}
        visual=np.zeros(1408,dtype=np.float32)
        if crops:
            visual=l2_normalize(self.vision_encoder.encode_faces(crops).mean(axis=0))
        textual=l2_normalize(self.text_encoder.encode([text]))[0]
        text_truncated=len(self.text_encoder.tokenizer.encode(text,add_special_tokens=True))>self.text_encoder.max_length
        text_p=self._predict(self.heads["text"],textual)
        visual_p=self._predict(self.heads["vision"],visual) if crops else None
        if crops:
            q=np.array([quality["valid_frame_fraction"],quality["mean_detection_score"],1.0],dtype=np.float32)
            probabilities=self._predict(self.heads["fusion"],np.concatenate([visual,textual,q]))
            source="fusion"
        else:
            probabilities=text_p
            source="text_fallback"
        self._synchronize()
        elapsed=(time.perf_counter()-started)*1000
        state=CheckInState(session_id=str(session_id),turn_id=str(turn_id),
            input={"text":text,"video_present":bool(video_path),"received_at":received,
                   "text_truncated_for_classification":text_truncated,
                   "clip_started_at":None,"clip_ended_at":None,"clock_source":"backend_utc; capture timestamps unavailable"},
            emotion=Emotion(label=LABELS[int(probabilities.argmax())],
                probabilities={label:float(p) for label,p in zip(LABELS,probabilities)},source=source),
            vision=quality,modalities={"text_label":LABELS[int(text_p.argmax())],
                "vision_label":LABELS[int(visual_p.argmax())] if visual_p is not None else None},
            modality_disagreement=visual_p is not None and int(text_p.argmax())!=int(visual_p.argmax()),
            timing={"classification_ms":elapsed,"first_token_ms":None,"completion_ms":None})
        return state.model_dump()

    def stream(self,text,video_path=None,history=None,session_id="local",turn_id="turn"):
        if not isinstance(text,str) or not text.strip():
            raise ValueError("Write a message before sending your check-in.")
        if len(text)>4000:
            raise ValueError("Keep your message below 4,000 characters.")
        if video_path is not None and not Path(video_path).is_file():
            raise ValueError("The selected video is no longer available. Record or select it again.")
        owner=self._claim_turn(session_id,turn_id)
        event_ids={"session_id":owner.session_id,"turn_id":owner.turn_id}
        received=datetime.now(timezone.utc).isoformat()
        stream=None
        state=None
        started=None
        released=False
        cancelled_at_release=False

        def cleanup():
            nonlocal released,cancelled_at_release
            if not released:
                released=True
                try:
                    close=getattr(stream,"close",None)
                    if callable(close):
                        close()
                finally:
                    # Snapshot cancellation and release atomically, before a
                    # terminal yield. An old iterator can never unlock a new one.
                    cancelled_at_release=self._release_turn(owner)
            return cancelled_at_release

        try:
            cold_started=time.perf_counter()
            self._load()
            if owner.cancelled.is_set():
                cleanup()
                yield {"type":"cancelled","phase":"before_classification",
                       "reason":"Stopped before an emotion state was produced.",**event_ids}
                return
            load_ms=(time.perf_counter()-cold_started)*1000
            started=time.perf_counter()
            state=self.classify(text.strip(),video_path,session_id,turn_id,received_at=received)
            state["timing"]["model_load_ms"]=load_ms
            if not owner.cancelled.is_set():
                state["response"]["status"]="streaming"
                yield {"type":"state","state":state,**event_ids}
            if not owner.cancelled.is_set():
                stream=self.generator.stream(text.strip(),state,history or [])
                first=True
                while not owner.cancelled.is_set():
                    try:
                        delta=next(stream)
                    except StopIteration:
                        break
                    if owner.cancelled.is_set():
                        break
                    if first:
                        state["timing"]["first_token_ms"]=(time.perf_counter()-started)*1000
                        first=False
                    state["response"]["text"]+=delta
                    yield {"type":"text_delta","text":delta,**event_ids}
            was_cancelled=cleanup()
            state["timing"]["completion_ms"]=(time.perf_counter()-started)*1000
            state["response"]["status"]="cancelled" if was_cancelled else "complete"
            yield {"type":"done","state":CheckInState.model_validate(state).model_dump(),**event_ids}
        except GeneratorExit:
            raise
        except Exception as error:
            try:
                was_cancelled=cleanup()
            except Exception as close_error:
                was_cancelled=cancelled_at_release
                error=RuntimeError(f"{error}; response cleanup also failed: {close_error}")
            if was_cancelled:
                if state is None:
                    yield {"type":"cancelled","phase":"before_classification",
                           "reason":"Stopped before an emotion state was produced.",**event_ids}
                else:
                    state["response"].update(status="cancelled",error=None)
                    state["timing"]["completion_ms"]=(time.perf_counter()-started)*1000
                    yield {"type":"done","state":CheckInState.model_validate(state).model_dump(),**event_ids}
                return
            if state is not None:
                state["response"].update(status="error",error=str(error))
                yield {"type":"state","state":state,**event_ids}
            yield {"type":"error","error":str(error),**event_ids}
        finally:
            cleanup()
