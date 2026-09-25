"""Split-aware MELD preparation and one shared visual preprocessing path."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from .settings import LABEL_TO_ID, MELD_SHA256

ANNOTATION_REVISION = "e8cedf27b5d2877e198332c957127e16eb214afe"
ANNOTATION_SPECS = {
    "train": {"count": 9989, "sha256": "d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db"},
    "dev": {"count": 1109, "sha256": "2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c"},
    "test": {"count": 2610, "sha256": "8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888"},
}

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_manifest(home, split=None):
    path = Path(home) / "manifests/meld.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if split is None or r["split"] == split]

def _verified_annotations(home, split):
    """Use exact official annotation bytes; cached setup also works offline."""
    spec = ANNOTATION_SPECS[split]
    path = Path(home) / "manifests" / f"{split}_sent_emo.csv"
    url = f"https://raw.githubusercontent.com/declare-lab/MELD/{ANNOTATION_REVISION}/data/MELD/{split}_sent_emo.csv"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        downloaded_hash = hashlib.sha256(content).hexdigest()
        if downloaded_hash != spec["sha256"]:
            raise ValueError(f"Downloaded {split} annotations have an unexpected SHA256: {downloaded_hash}")
        temporary = path.with_suffix(".csv.download")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise ValueError(f"Cached {split} annotations have an unexpected SHA256: {actual}; expected {spec['sha256']}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != spec["count"]:
        raise ValueError(f"Official {split} annotations require {spec['count']} rows; found {len(rows)}")
    return rows, {"revision": ANNOTATION_REVISION, "url": url,
                  "sha256": actual, "rows": len(rows)}


def _cross_split_media_duplicates(rows):
    by_hash = defaultdict(list)
    for row in rows:
        if row.get("media_sha256"):
            by_hash[row["media_sha256"]].append(row)
    return [
        {"sha256": digest, "ids": sorted(row["id"] for row in group),
         "splits": sorted({row["split"] for row in group})}
        for digest, group in sorted(by_hash.items())
        if len({row["split"] for row in group}) > 1
    ]


def prepare(home):
    home = Path(home)
    archive = home / "downloads/MELD.Raw.tar.gz"
    target = home / "data/raw"
    target.mkdir(parents=True, exist_ok=True)
    complete = target / ".extracted"
    if not complete.exists():
        actual = sha256(archive)
        if actual != MELD_SHA256:
            raise ValueError(f"MELD checksum mismatch: {actual}")
        print("Verified raw archive; extracting outer archive", flush=True)
        with tarfile.open(archive) as tf:
            tf.extractall(target, filter="data")
        for inner in list(target.rglob("*.tar.gz")):
            destination = inner.parent / inner.name.removesuffix(".tar.gz")
            marker = destination / ".extracted"
            if marker.exists():
                continue
            destination.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {inner.name}", flush=True)
            with tarfile.open(inner) as tf:
                tf.extractall(destination, filter="data")
            marker.touch()
        complete.touch()
    index = {}
    for path in target.rglob("*.mp4"):
        rel = str(path.relative_to(target)).lower()
        split = "train" if "train" in rel else "dev" if "dev" in rel else "test" if "test" in rel else None
        if split:
            key = (split, path.name.lower())
            if key in index and index[key] != str(path.resolve()):
                raise ValueError(f"Duplicate clip identity: {key}")
            index[key] = str(path.resolve())
    if not index:
        raise ValueError("No split-identifiable videos found; inspect archive layout")
    all_rows, annotation_provenance = [], {}
    for split in ("train", "dev", "test"):
        annotations, annotation_provenance[split] = _verified_annotations(home, split)
        for source in annotations:
            did, uid = int(source["Dialogue_ID"]), int(source["Utterance_ID"])
            emotion = source["Emotion"].lower().strip()
            name = f"dia{did}_utt{uid}.mp4"
            video_path = index.get((split, name))
            all_rows.append({"id": f"{split}:{did}:{uid}", "split": split, "dialogue_id": did,
                "utterance_id": uid, "text": source["Utterance"], "speaker_audit_only": source["Speaker"],
                "label": LABEL_TO_ID[emotion], "emotion": emotion, "video_path": video_path,
                "media_available": video_path is not None,
                "media_sha256": sha256(video_path) if video_path else None})
    identities = [r["id"] for r in all_rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate annotation identities")
    out = home / "manifests/meld.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows), encoding="utf-8")
    report = {"split_counts": dict(Counter(r["split"] for r in all_rows)),
              "missing_media": [r["id"] for r in all_rows if not r["media_available"]],
              "archive_sha256": MELD_SHA256,
              "annotation_provenance": annotation_provenance,
              "cross_split_exact_media_duplicates": _cross_split_media_duplicates(all_rows)}
    (home / "reports/preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return all_rows

def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    union = a[2]*a[3] + b[2]*b[3] - intersection
    return float(intersection / union) if union else 0.0

class VideoProcessor:
    """Face availability heuristic, explicitly not active-speaker identification."""
    version = "yunet-eightframes-v1-score08-singleface-sixvalid"

    def __init__(self, model_path, frames=8, min_valid=6):
        self.frames, self.min_valid = frames, min_valid
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.8, 0.3, 5000)

    def process(self, video_path, contact_path=None):
        quality = {"available": False, "reason": "missing_video", "valid_frame_fraction": 0.0,
                   "mean_detection_score": 0.0, "sampled_frames": 0, "selected_frames": 0,
                   "duration_seconds": None, "truncated": False, "speaker_attribution": "unverified_heuristic"}
        if not video_path or not Path(video_path).is_file():
            return [], quality
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count <= 0 or fps <= 0:
            capture.release(); quality["reason"] = "decode_failed"
            return [], quality
        duration = count / fps
        quality.update(duration_seconds=duration, truncated=duration > 10)
        end = min(count - 1, max(0, int(min(duration, 10) * fps)-1))
        positions = np.linspace(0, end, self.frames).astype(int)
        crops, scores, previews, normalized_boxes = [], [], [], []
        multiple = False
        try:
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if not ok:
                    continue
                quality["sampled_frames"] += 1
                height, width = frame.shape[:2]
                ratio = min(1.0, 640 / width)
                resized = cv2.resize(frame, (int(width*ratio), int(height*ratio))) if ratio < 1 else frame
                h, w = resized.shape[:2]
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(resized)
                faces = [] if faces is None else [f for f in faces if min(f[2], f[3]) >= 40*ratio]
                preview = resized.copy()
                if len(faces) > 1:
                    multiple = True
                if len(faces) == 1:
                    f = faces[0]
                    x, y, fw, fh = f[:4]
                    normalized_boxes.append(np.array([x/w, y/h, fw/w, fh/h]))
                    x1, y1 = max(0, int(x-fw*.15)), max(0, int(y-fh*.15))
                    x2, y2 = min(w, int(x+fw*1.15)), min(h, int(y+fh*1.15))
                    crop = resized[y1:y2, x1:x2]
                    if crop.size:
                        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        scores.append(float(f[-1]))
                    cv2.rectangle(preview, (x1,y1), (x2,y2), (70,200,110), 2)
                if contact_path:
                    preview = cv2.resize(preview, (256,144))
                    cv2.putText(preview, f"f{position} faces:{len(faces)}", (5,137), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255),1)
                    previews.append(preview)
        finally:
            capture.release()
        abrupt = any(box_iou(a,b) < .05 for a,b in zip(normalized_boxes, normalized_boxes[1:]))
        available = len(crops) >= self.min_valid and not multiple and not abrupt
        reason = "ok" if available else "multiple_faces" if multiple else "track_change" if abrupt else "insufficient_face_frames"
        quality.update(available=available, reason=reason, selected_frames=len(crops),
                       valid_frame_fraction=len(crops)/self.frames,
                       mean_detection_score=float(np.mean(scores)) if scores else 0.0)
        if contact_path and previews:
            while len(previews) < 8:
                previews.append(np.zeros_like(previews[0]))
            sheet = np.vstack([np.hstack(previews[:4]), np.hstack(previews[4:8])])
            Path(contact_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(contact_path), sheet)
        return crops if available else [], quality

def l2_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
