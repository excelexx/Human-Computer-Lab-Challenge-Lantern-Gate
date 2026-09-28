"""Bounded, session-owned webcam evidence; no images persist between callbacks.

This uses the clip path's face geometry and mean-then-normalize feature pooling.
Sampling is a rolling browser stream rather than eight positions in a clip, so
webcam behavior is an interaction feature, not an evaluated MELD accuracy claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import math
import time

import cv2
import numpy as np

from .data import box_iou, l2_normalize
from .settings import LABELS

MAX_SAMPLES = 8
MIN_VALID = 6
WINDOW_SECONDS = 4.0
FRESH_SECONDS = 4.0
MIN_INTERVAL_SECONDS = 0.12
DISPLAY_ALPHA = 0.85
DISPLAY_EMA_GAP_SECONDS = 0.4
FEATURE_DIM = 1408


@dataclass
class _Lease:
    session_id: str = ""
    epoch: int = 0
    enabled: bool = False
    sequence: int = 0


@dataclass(frozen=True)
class LiveVisionObservation:
    """Immutable evidence with a revocable, deepcopy-safe camera/session lease."""
    session_id: str
    epoch: int
    sequence: int
    received_monotonic: float
    received_at: str
    started_at: str
    feature: tuple[float, ...]
    probabilities: tuple[float, ...]
    sampled_frames: int
    selected_frames: int
    mean_detection_score: float
    duration_seconds: float
    _lease: _Lease = field(repr=False, compare=False)

    def valid_for(self, session_id, now=None):
        now = time.monotonic() if now is None else now
        return (self._lease.enabled and self._lease.epoch == self.epoch and self._lease.sequence == self.sequence
                and self._lease.session_id == self.session_id == str(session_id)
                and 0 <= now - self.received_monotonic < FRESH_SECONDS
                and MIN_VALID <= self.selected_frames <= self.sampled_frames <= MAX_SAMPLES
                and len(self.feature) == FEATURE_DIM and all(math.isfinite(v) for v in self.feature)
                and len(self.probabilities) == len(LABELS)
                and all(math.isfinite(p) and 0 <= p <= 1 for p in self.probabilities)
                and abs(sum(self.probabilities) - 1) < 1e-4)

    @property
    def quality(self):
        return {"available": True, "reason": "ok", "input_kind": "live_camera",
                "valid_frame_fraction": self.selected_frames / MAX_SAMPLES,
                "mean_detection_score": self.mean_detection_score,
                "sampled_frames": self.sampled_frames, "selected_frames": self.selected_frames,
                "duration_seconds": self.duration_seconds, "truncated": False,
                "speaker_attribution": "unverified_heuristic",
                "sampling": "up_to_eight_recent_camera_frames_over_four_seconds",
                "clock_source": "backend_utc_receipt; browser_capture_timestamps_unavailable",
                "window_started_at": self.started_at, "window_ended_at": self.received_at}


@dataclass(frozen=True)
class LiveSample:
    received_monotonic: float
    received_at: str
    reason: str
    feature: tuple[float, ...] | None = None
    box: tuple[float, ...] | None = None
    score: float = 0.0


@dataclass(frozen=True)
class LiveDisplay:
    """Fast tentative tag, deliberately independent of fusion eligibility."""
    probabilities: tuple[float, ...]
    received_monotonic: float
    received_at: str
    smoothed: bool = False


def update_display(probabilities, previous, received_monotonic, received_at):
    values=np.asarray(probabilities,dtype=np.float32)
    if (values.shape!=(len(LABELS),) or not np.isfinite(values).all()
            or np.any(values<0) or np.any(values>1) or abs(float(values.sum())-1)>1e-4):
        raise ValueError("The live display requires seven finite emotion scores")
    smoothed=(previous is not None
              and 0<=received_monotonic-previous.received_monotonic<=DISPLAY_EMA_GAP_SECONDS)
    if smoothed:
        values=DISPLAY_ALPHA*values+(1-DISPLAY_ALPHA)*np.asarray(previous.probabilities,dtype=np.float32)
    return LiveDisplay(tuple(float(p) for p in values),received_monotonic,received_at,smoothed)


@dataclass
class _Window:
    observation: LiveVisionObservation | None = None
    display: LiveDisplay | None = None
    last_box: tuple[float, ...] | None = None
    samples: list[LiveSample] = field(default_factory=list, repr=False)
    sequence: int = 0
    last_attempt: float = float("-inf")
    status: dict = field(default_factory=dict, repr=False)


@dataclass
class LiveVisionBuffer:
    """One browser's bounded state. Locks and camera images are never stored here.

    Reset swaps the complete window object. An in-flight callback may finish
    writing its old window, but cannot repopulate the new camera epoch.
    """
    session_id: str = ""
    enabled: bool = False
    _lease: _Lease = field(default_factory=_Lease, repr=False)
    _window: _Window = field(default_factory=_Window, repr=False)

    def __post_init__(self):
        self.session_id = str(self.session_id)
        self._lease.session_id, self._lease.enabled = self.session_id, self.enabled

    @property
    def epoch(self):
        return self._lease.epoch

    @property
    def observation(self):
        return self._window.observation

    @property
    def samples(self):
        return self._window.samples

    @property
    def sequence(self):
        return self._window.sequence

    @property
    def last_attempt(self):
        return self._window.last_attempt

    def clear(self, enabled=False, session_id=None):
        # Invalidate outstanding snapshots before clearing any retained state.
        self._lease.epoch += 1
        self.enabled = bool(enabled)
        if session_id is not None:
            self.session_id = str(session_id)
        self._lease.enabled, self._lease.session_id = self.enabled, self.session_id
        self._lease.sequence = 0
        self._window = _Window()
        self._window.status = status_report("warming" if self.enabled else "off", self)
        return self

    reset = clear

    def start(self, session_id):
        return self.clear(enabled=True, session_id=session_id)

    def is_current(self, session_id, epoch, sequence=None):
        return (self.enabled and self.session_id == str(session_id) and self.epoch == epoch
                and (sequence is None or self.sequence == sequence))

    def snapshot(self, session_id):
        value = self.observation
        return replace(value) if value is not None and value.valid_for(session_id) else None

    def current_status(self, session_id):
        if not self.enabled:
            return status_report("off", self)
        if self.session_id != str(session_id):
            return status_report("discarded", self)
        now = time.monotonic()
        window = self._window
        observation = window.observation
        display = window.display
        last_attempt = window.last_attempt
        sequence = window.sequence
        if observation is not None:
            if not observation.valid_for(session_id, now) and window.observation is observation:
                window.observation = None
        if display is not None:
            age=now-display.received_monotonic
            if 0<=age<FRESH_SECONDS:
                value=dict(window.status)
                value.update(display_age_seconds=age,observation_age_seconds=age,
                             fusion_ready=window.observation is not None)
                return value
            if window.display is display:
                window.display=None
                window.last_box=None
                window.status=status_report("stale",self)
        if (last_attempt != float("-inf") and now-last_attempt>=FRESH_SECONDS
                and window.last_attempt==last_attempt and window.sequence==sequence
                and (window.display is None or window.display is display)):
            window.samples.clear()
            if window.display is display:
                window.display=None
            window.last_box=None
            window.status = status_report("stale", self)
        return dict(window.status) if window.status else status_report("warming", self)


def status_report(status, buffer, **values):
    return {"status": status, "available": False, "label": None, "probabilities": None,
            "score_semantics": "uncalibrated_softmax", "source": "vision",
            "display_scope": "tentative_single_frame_with_short_probability_ema",
            "tentative": False, "fusion_ready": False,
            "display_age_seconds": None, "process_latency_ms": None,
            "display_smoothing_alpha": DISPLAY_ALPHA,
            "session_id": buffer.session_id, "epoch": buffer.epoch, "sequence": buffer.sequence,
            "sampled_frames": len(buffer.samples), "selected_frames": 0,
            "observation_age_seconds": None, **values}


def extract_live_face(frame, detector):
    """Mirror VideoProcessor's detector/crop math for a browser RGB frame.

    Converting before resize ensures OpenCV sees the identical BGR pixels to the
    file path. Frame size is bounded before allocating conversion buffers.
    """
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
            or frame.shape[2] != 3 or min(frame.shape[:2]) < 1
            or max(frame.shape[:2]) > 4096 or frame.shape[0] * frame.shape[1] > 8_388_608):
        return None, None, 0.0, "invalid_frame"
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    ratio = min(1.0, 640 / width)
    target = (int(width * ratio), int(height * ratio))
    if min(target) < 1:
        return None, None, 0.0, "invalid_frame"
    resized = cv2.resize(bgr, target) if ratio < 1 else bgr
    h, w = resized.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(resized)
    faces = [] if faces is None else [f for f in faces if min(f[2], f[3]) >= 40 * ratio]
    if len(faces) != 1:
        return None, None, 0.0, "multiple_faces" if len(faces) > 1 else "no_face"
    face = faces[0]
    if not np.isfinite(face).all() or not 0 <= float(face[-1]) <= 1:
        return None, None, 0.0, "invalid_detection"
    x, y, fw, fh = face[:4]
    x1, y1 = max(0, int(x - fw * .15)), max(0, int(y - fh * .15))
    x2, y2 = min(w, int(x + fw * 1.15)), min(h, int(y + fh * 1.15))
    crop = resized[y1:y2, x1:x2]
    if not crop.size:
        return None, None, 0.0, "no_face"
    return (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            tuple(float(v) for v in (x / w, y / h, fw / w, fh / h)), float(face[-1]), "ok")


def aggregate_samples(samples):
    """Return normalized mean of raw embeddings, using the clip eligibility rule."""
    valid = [sample for sample in samples if sample.feature is not None]
    boxes = [sample.box for sample in valid]
    abrupt = any(box_iou(a, b) < .05 for a, b in zip(boxes, boxes[1:]))
    multiple = any(sample.reason == "multiple_faces" for sample in samples)
    reason = ("multiple_faces" if multiple else "track_change" if abrupt else
              samples[-1].reason if samples and samples[-1].reason != "ok" else
              "warming" if len(valid) < MIN_VALID else "ready")
    quality = {"available": reason == "ready", "reason": reason,
               "sampled_frames": len(samples), "selected_frames": len(valid),
               "valid_frame_fraction": len(valid) / MAX_SAMPLES,
               "mean_detection_score": float(np.mean([v.score for v in valid])) if valid else 0.0}
    if reason != "ready":
        return None, quality
    feature = l2_normalize(np.asarray([v.feature for v in valid], dtype=np.float32).mean(axis=0))
    return feature, quality


def expected_text_metadata():
    """The unchanged pinned text contract, without constructing/running text ML."""
    from types import SimpleNamespace
    from .encoders import TextEncoder, TEXT_FEATURE_DIM, TEXT_PARAMETER_COUNT, TEXT_SHA256
    descriptor = SimpleNamespace(artifact_sha256=TEXT_SHA256, adapted_sha256=None,
                                 feature_dim=TEXT_FEATURE_DIM, max_length=128,
                                 parameter_count=lambda: TEXT_PARAMETER_COUNT)
    return TextEncoder.metadata(descriptor)


def utc_now():
    return datetime.now(timezone.utc).isoformat()
