"""Shared local inference path, with turn serialization and explicit fallbacks."""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass,field
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .data import VideoProcessor,l2_normalize,box_iou
from .encoders import VisionEncoder,TextEncoder
from .generator import LocalGenerator, _emotion_evidence
from .character import character_context
from .models import DIMENSIONS,load_head
from .schema import CheckInState,Emotion
from .settings import LABELS,runtime_home
from .live_vision import (LiveVisionBuffer, LiveVisionObservation, LiveSample,
    MAX_SAMPLES, WINDOW_SECONDS, FRESH_SECONDS, MIN_INTERVAL_SECONDS,
    aggregate_samples, extract_live_face, expected_text_metadata, status_report, update_display, utc_now)


@dataclass(frozen=True)
class _TurnOwner:
    session_id: str
    turn_id: str
    cancelled: threading.Event = field(default_factory=threading.Event)


class _PendingTurnCancelled(Exception):
    def __init__(self,owner):
        self.owner=owner
        super().__init__("Stopped while waiting for the camera sample")


class CheckInPipeline:
    def __init__(self, home=None, device=None):
        self.home=runtime_home(home)
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator=LocalGenerator()
        self._lock=threading.Lock()
        self._lifecycle_lock=threading.Lock()
        self._active_turn=None
        self._pending_turn=None
        self._live_busy=False
        self._cancel=threading.Event()
        self._active_session=None
        self._loaded=False
        self._vision_loaded=False

    def cancel(self,session_id=None,turn_id=None):
        session_id=str(session_id) if session_id is not None else None
        turn_id=str(turn_id) if turn_id is not None else None
        with self._lifecycle_lock:
            owner=self._active_turn or self._pending_turn
            if owner is None or (session_id is not None and session_id!=owner.session_id) or (turn_id is not None and turn_id!=owner.turn_id):
                return False
            owner.cancelled.set()
            return True

    def _claim_turn(self,session_id,turn_id):
        with self._lifecycle_lock:
            if self._active_turn is not None or self._pending_turn is not None:
                raise RuntimeError("Another turn is still running. Wait or stop it first.")
            owner=_TurnOwner(str(session_id),str(turn_id))
            if self._lock.acquire(blocking=False):
                self._active_turn=owner
                self._active_session=owner.session_id
                self._cancel=owner.cancelled
                return owner
            if not self._live_busy:
                raise RuntimeError("Another turn is still running. Wait or stop it first.")
            # Reserve Send priority; no further idle callback may enter while
            # this turn waits for the currently executing camera sample.
            self._pending_turn=owner
        acquired=False
        try:
            deadline=time.monotonic()+30.0
            while not acquired:
                if owner.cancelled.is_set():
                    raise _PendingTurnCancelled(owner)
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise RuntimeError("The camera is still starting. Please try sending again in a moment.")
                acquired=self._lock.acquire(timeout=min(.05,remaining))
            with self._lifecycle_lock:
                if owner.cancelled.is_set():
                    raise _PendingTurnCancelled(owner)
                self._pending_turn=None
                self._active_turn=owner
                self._active_session=owner.session_id
                self._cancel=owner.cancelled
                return owner
        except BaseException:
            with self._lifecycle_lock:
                if self._pending_turn is owner:
                    self._pending_turn=None
                if acquired:
                    self._lock.release()
            raise

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
                "vision_loaded":self._vision_loaded,
                "device":self.device,"running":self._active_turn is not None or self._pending_turn is not None,"errors":errors,"components":{"generator":generator},
                "artifact_validation":"loaded" if self._loaded else "pending_first_load",
                "parameter_budget":inventory["total_parameter_upper_bound"],
                "parameter_limit":inventory["parameter_limit"],
                "parameter_accounting_status":"invalid" if invalid_heads else "loaded_runtime" if self._loaded else "checkpoints_counted" if all(check["status"]=="verified" for check in head_checks) else "planned",
                "head_inventory":head_checks}

    def _load_vision(self):
        """Load only the visual path for idle webcam updates; never run text/LLM."""
        if self._vision_loaded:
            return
        torch.set_num_threads(4)
        self.processor=VideoProcessor(self.home/"models/vision/yunet.onnx")
        self.vision_encoder=VisionEncoder(self.home/"models/vision/enet_b2_7.pt",device=self.device)
        model,payload=load_head(self.home/"checkpoints/vision.pt","cpu")
        if payload.get("stage")!="vision" or payload.get("input_dim")!=DIMENSIONS["vision"] or payload.get("labels")!=LABELS:
            raise ValueError("The vision checkpoint has an incompatible stage, dimension or label order; retrain it")
        from .features import feature_metadata,feature_identity
        from types import SimpleNamespace
        descriptor=SimpleNamespace(metadata=expected_text_metadata)
        actual=feature_identity(feature_metadata(self.vision_encoder,descriptor,self.processor))
        if payload.get("feature_identity")!=actual:
            raise ValueError("The vision checkpoint does not match current encoders/preprocessing; re-extract and retrain")
        self.heads={"vision":model}
        if self.device.startswith("cuda"):
            self.vision_encoder.encode_faces([np.zeros((260,260,3),dtype=np.uint8)])
            self._synchronize()
        self._vision_loaded=True

    def prepare_live(self):
        """Warm the visual path before accepting camera frames; no text or LLM.

        Uses the same exclusive GPU ownership as idle callbacks and Send. The
        caller gets a clear error if another operation already owns the device.
        """
        with self._lifecycle_lock:
            if self._pending_turn is not None or not self._lock.acquire(blocking=False):
                raise RuntimeError("Cannot prepare the live camera while another operation is running")
            self._live_busy=True
        started=time.perf_counter()
        try:
            self._load_vision()
            self._synchronize()
            return {"ready":True,"load_ms":(time.perf_counter()-started)*1000}
        finally:
            with self._lifecycle_lock:
                self._live_busy=False
                self._lock.release()

    def _load(self):
        if self._loaded:
            return
        self._load_vision()
        self.text_encoder=TextEncoder(self.home/"models/text",device=self.device,batch_size=1)
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

    def observe_live_frame(self,frame,buffer,session_id,*,enabled=True):
        """Nonblocking idle vision update; the same lock serializes check-in ML.

        UI camera start/stop owns buffer.start()/clear(). A callback may never
        enable a buffer by itself, nor republish into a newer camera epoch.
        """
        if not isinstance(buffer,LiveVisionBuffer):
            raise TypeError("Expected server-side LiveVisionBuffer state")
        if not enabled or not buffer.enabled:
            return buffer,status_report("off",buffer)
        if buffer.session_id!=str(session_id):
            return buffer,status_report("discarded",buffer)
        now=time.monotonic()
        if now-buffer.last_attempt<MIN_INTERVAL_SECONDS:
            return buffer,{**buffer.current_status(session_id),"skipped":"cooldown"}
        with self._lifecycle_lock:
            if self._pending_turn is not None or not self._lock.acquire(blocking=False):
                # A user turn has priority over further idle camera samples.
                return buffer,status_report("busy",buffer)
            self._live_busy=True
        epoch=buffer.epoch
        window=buffer._window
        window.sequence+=1
        sequence=window.sequence
        window.last_attempt=now
        received=utc_now()
        process_started=time.perf_counter()
        try:
            self._load_vision()
            if not buffer.is_current(session_id,epoch,sequence):
                return buffer,status_report("discarded",buffer)
            crop,box,score,reason=extract_live_face(frame,self.processor.detector)
            values=None
            if crop is not None:
                encoded=np.asarray(self.vision_encoder.encode_faces([crop]),dtype=np.float32)
                if encoded.shape!=(1,1408) or not np.isfinite(encoded).all():
                    raise ValueError("The visual encoder returned invalid live features")
                values=tuple(float(v) for v in encoded[0])
            finished=time.monotonic()
            if not buffer.is_current(session_id,epoch,sequence):
                return buffer,status_report("discarded",buffer)
            if finished-now>=FRESH_SECONDS:
                window.samples.clear()
                window.observation=None
                window.display=None
                window.last_box=None
                window.status=status_report("stale",buffer)
                return buffer,buffer.current_status(session_id)
            recent=[sample for sample in window.samples if 0<=now-sample.received_monotonic<WINDOW_SECONDS]
            recent=(recent+[LiveSample(now,received,reason,values,box,score)])[-MAX_SAMPLES:]
            feature,quality=aggregate_samples(recent)
            # Fast feedback uses a fresh normalized frame, never a partially
            # filled fusion vector. An abrupt face change clears the display
            # for this frame; the next stable frame can produce a new tag.
            track_jump=(box is not None and window.last_box is not None
                        and box_iou(window.last_box,box)<.05)
            display=None
            display_reason="track_change" if track_jump else reason
            if values is not None and not track_jump:
                raw_probabilities=self._predict(self.heads["vision"],l2_normalize(np.asarray(values,dtype=np.float32)))
                display=update_display(raw_probabilities,window.display,now,received)
            observation=None
            if feature is not None:
                probabilities=self._predict(self.heads["vision"],feature)
                observation=LiveVisionObservation(str(session_id),epoch,sequence,now,received,
                    recent[0].received_at,tuple(float(v) for v in feature),
                    tuple(float(v) for v in probabilities),len(recent),quality["selected_frames"],
                    quality["mean_detection_score"],now-recent[0].received_monotonic,buffer._lease)
            if not buffer.is_current(session_id,epoch,sequence):
                return buffer,status_report("discarded",buffer)
            buffer._lease.sequence=sequence
            if observation is not None and not observation.valid_for(session_id,finished):
                raise ValueError("Live emotion output is invalid")
            window.samples=recent
            window.observation=observation
            window.display=display
            if display is not None:
                window.last_display=display
            elif display_reason in {"track_change","multiple_faces"}:
                # Do not attach a previous person's tag to an ambiguous face.
                window.last_display=None
            window.last_box=box
            window.status=status_report("ready" if display is not None else display_reason,buffer,
                quality=observation.quality if observation is not None else quality,
                fusion_quality=quality,fusion_ready=observation is not None,
                sampled_frames=len(recent),selected_frames=quality["selected_frames"],
                process_latency_ms=(time.perf_counter()-process_started)*1000)
            if display is not None:
                window.status.update(available=True,tentative=True,
                    label=LABELS[int(np.argmax(display.probabilities))],
                    probabilities={label:float(p) for label,p in zip(LABELS,display.probabilities)},
                    display_smoothed=display.smoothed,
                    display_age_seconds=finished-now,observation_age_seconds=finished-now,
                    display_quality={"available":True,"reason":"single_face","detection_score":score},
                    observed_at=received)
            return buffer,buffer.current_status(session_id)
        except Exception as error:
            if not buffer.is_current(session_id,epoch,sequence):
                return buffer,status_report("discarded",buffer)
            window.samples.clear()
            window.observation=None
            window.display=None
            window.last_box=None
            buffer._lease.sequence=sequence
            window.status=status_report("error",buffer,error=str(error))
            return buffer,dict(window.status)
        finally:
            with self._lifecycle_lock:
                self._live_busy=False
                self._lock.release()

    def observe_live(self,frame,buffer,session_id):
        return self.observe_live_frame(frame,buffer,session_id)[1]

    def _synchronize(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @staticmethod
    def _predict(model,features):
        with torch.inference_mode():
            return torch.softmax(model(torch.from_numpy(np.asarray(features,dtype=np.float32)).reshape(1,-1)),dim=-1)[0].numpy()

    def classify(self,text,video_path,session_id,turn_id,received_at=None,*,live_observation=None,live_admitted_at=None):
        received=received_at or datetime.now(timezone.utc).isoformat()
        self._load()
        self._synchronize()
        started=time.perf_counter()
        visual=np.zeros(1408,dtype=np.float32)
        admitted=time.monotonic() if live_admitted_at is None else live_admitted_at
        live=(live_observation if not video_path and isinstance(live_observation,LiveVisionObservation)
              and live_observation.valid_for(session_id,admitted) else None)
        if live is not None:
            visual=np.asarray(live.feature,dtype=np.float32)
            quality=live.quality
            visual_available=True
        else:
            try:
                crops,quality=self.processor.process(video_path)
            except Exception as error:
                crops=[]
                quality={"available":False,"reason":"preprocess_error","error":str(error),
                         "valid_frame_fraction":0.0,"mean_detection_score":0.0,"speaker_attribution":"unverified_heuristic"}
            visual_available=bool(crops)
            if crops:
                visual=l2_normalize(self.vision_encoder.encode_faces(crops).mean(axis=0))
            elif live_observation is not None and not video_path:
                quality.update(reason="stale_or_invalid_live_observation",input_kind="live_camera")
        textual=l2_normalize(self.text_encoder.encode([text]))[0]
        text_truncated=len(self.text_encoder.tokenizer.encode(text,add_special_tokens=True))>self.text_encoder.max_length
        text_p=self._predict(self.heads["text"],textual)
        # An off/reset during text encoding invalidates the old camera lease.
        if live is not None and not live.valid_for(session_id,admitted):
            live=None
            visual_available=False
            quality={**quality,"available":False,"reason":"stale_or_invalid_live_observation"}
        visual_p=self._predict(self.heads["vision"],visual) if visual_available else None
        if visual_available:
            q=np.array([quality["valid_frame_fraction"],quality["mean_detection_score"],1.0],dtype=np.float32)
            probabilities=self._predict(self.heads["fusion"],np.concatenate([visual,textual,q]))
            source="fusion"
        else:
            probabilities=text_p
            source="text_fallback"
        self._synchronize()
        elapsed=(time.perf_counter()-started)*1000
        state=CheckInState(session_id=str(session_id),turn_id=str(turn_id),
            input={"text":text,"video_present":bool(video_path) or live is not None,"received_at":received,
                   "text_truncated_for_classification":text_truncated,
                   "clip_started_at":live.started_at if live is not None else None,
                   "clip_ended_at":live.received_at if live is not None else None,
                   "clock_source":"backend_utc_receipt; browser_capture_timestamps_unavailable" if live is not None else "backend_utc; capture timestamps unavailable"},
            emotion=Emotion(label=LABELS[int(probabilities.argmax())],
                probabilities={label:float(p) for label,p in zip(LABELS,probabilities)},source=source),
            vision=quality,modalities={"text_label":LABELS[int(text_p.argmax())],
                "vision_label":LABELS[int(visual_p.argmax())] if visual_p is not None else None},
            modality_disagreement=visual_p is not None and int(text_p.argmax())!=int(visual_p.argmax()),
            timing={"classification_ms":elapsed,"first_token_ms":None,"completion_ms":None})
        if live is not None:
            state.input["camera_age_at_send_seconds"]=admitted-live.received_monotonic
        result=state.model_dump()
        result["interaction"]=character_context(_emotion_evidence(result), text)
        return result

    def stream(self,text,video_path=None,history=None,session_id="local",turn_id="turn",*,live_observation=None,game_context=None):
        if not isinstance(text,str) or not text.strip():
            raise ValueError("Write a message before sending your check-in.")
        if len(text)>4000:
            raise ValueError("Keep your message below 4,000 characters.")
        if video_path is not None and not Path(video_path).is_file():
            raise ValueError("The selected video is no longer available. Record or select it again.")
        live_admitted_at=time.monotonic()
        try:
            owner=self._claim_turn(session_id,turn_id)
        except _PendingTurnCancelled as error:
            owner=error.owner
            yield {"type":"cancelled","phase":"before_classification",
                   "reason":"Stopped before an emotion state was produced.",
                   "session_id":owner.session_id,"turn_id":owner.turn_id}
            return
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
            live_args={"live_observation":live_observation,"live_admitted_at":live_admitted_at} if live_observation is not None else {}
            state=self.classify(text.strip(),video_path,session_id,turn_id,received_at=received,**live_args)
            if game_context is not None:
                from .quest import safe_context
                state["game"] = safe_context(game_context)
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
