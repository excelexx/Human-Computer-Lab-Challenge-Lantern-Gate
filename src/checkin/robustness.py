"""Reproducible visual-corruption and mismatched-modality development backtests.

No training or checkpoint selection occurs here. Text features stay fixed while
video is transformed and reprocessed by the real detector and visual encoder.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import cv2
import numpy as np
from sklearn.metrics import f1_score
import torch

from .data import VideoProcessor, l2_normalize, read_manifest
from .encoders import VisionEncoder, TextEncoder
from .features import feature_identity
from .models import DIMENSIONS, load_head
from .settings import LABELS, YUNET_SHA256, runtime_home
from .train import atomic_json, file_sha256, load_cache, predict_probabilities


VARIANTS = {
    "original": {"description": "Original raw MELD video, freshly processed", "filter": None},
    "silent_copy": {"description": "Audio removed with video stream copy", "filter": None},
    "h264_reencode": {"description": "H.264 CRF23 medium, yuv420p; legacy-style mute conversion", "filter": None},
    "half_resolution": {"description": "Half each dimension, rounded to even; H.264 CRF23", "filter": "scale=trunc(iw/4)*2:trunc(ih/4)*2"},
    "half_brightness": {"description": "RGB values multiplied by 0.5; H.264 CRF23", "filter": "lutrgb=r=val*0.5:g=val*0.5:b=val*0.5"},
    "gaussian_blur": {"description": "Gaussian blur sigma=3; H.264 CRF23", "filter": "gblur=sigma=3"},
    "horizontal_flip": {"description": "Horizontal mirroring; H.264 CRF23", "filter": "hflip"},
    "black_control": {"description": "All pixels black, no visual evidence expected; text unchanged", "filter": "lutrgb=r=0:g=0:b=0"},
}


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verified_feature_contract(home, cache):
    """Check current feature code against cached provenance without loading text."""
    metadata = json.loads((Path(home) / "cache/dev.metadata.json").read_text(encoding="utf-8"))
    contract = metadata["feature_metadata"]
    code = "\n".join(inspect.getsource(obj) for obj in (VideoProcessor, l2_normalize, VisionEncoder, TextEncoder))
    code_sha = hashlib.sha256(code.encode()).hexdigest()
    if not cache["feature_identity"] or feature_identity(contract) != cache["feature_identity"] or metadata["feature_identity"] != cache["feature_identity"]:
        raise ValueError("Cache metadata does not match its feature identity")
    if code_sha != contract["implementation_sha256"]:
        raise ValueError("Live encoder/preprocessing implementation differs from cached features")
    detector_sha = file_sha256(Path(home) / "models/vision/yunet.onnx")
    if detector_sha != YUNET_SHA256:
        raise ValueError("YuNet bytes differ from the pinned detector")
    return {"feature_metadata": contract, "live_implementation_sha256": code_sha, "yunet_sha256": detector_sha}


def selected_manifest_rows(cache, selected, rows):
    """Bind complete selected dev records and verify their actual source bytes."""
    by_id = {}
    for row in rows:
        if row["id"] in by_id:
            raise ValueError("Duplicate prepared development identity")
        by_id[row["id"]] = row
    selected_rows = []
    for index in selected:
        row = by_id.get(str(cache["ids"][index]))
        if not row or row.get("split") != "dev" or row.get("label") != int(cache["labels"][index]):
            raise ValueError("Prepared dev media or label mismatch")
        if not row.get("video_path") or not Path(row["video_path"]).is_file():
            raise ValueError("Prepared development video is missing")
        if file_sha256(row["video_path"]) != row.get("media_sha256"):
            raise ValueError(f"Source media changed: {row['id']}")
        selected_rows.append(dict(row))
    return selected_rows


def empty_example(row, text, plan):
    count = len(VARIANTS)
    sample = {"id": row["id"], "label": LABELS[row["label"]], "text": row["text"],
              "source_media_sha256": row["media_sha256"], "protocol_identity": plan["protocol_identity"],
              "conditions": {name: {"error": "not_processed"} for name in VARIANTS}}
    arrays = {"vision": np.zeros((count, 1408), dtype=np.float32), "quality": np.zeros((count, 3), dtype=np.float32),
              "text": np.asarray(text, dtype=np.float32).copy(), "valid": np.zeros(count, dtype=bool),
              "condition_names": np.array(list(VARIANTS)), "id": np.array(row["id"]),
              "feature_identity": np.array(plan["feature_identity"]), "protocol_identity": np.array(plan["protocol_identity"]),
              "source_media_sha256": np.array(row["media_sha256"])}
    return sample, arrays


def save_example(output, result_file, sample, arrays):
    """Publish immutable features before their JSON pointer; an interruption cannot mix generations."""
    directory = Path(output) / "features"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / (uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        digest = file_sha256(temporary)
        target = directory / f"{sample['id'].replace(':', '-')}-{digest}.npz"
        if not target.exists():
            os.replace(temporary, target)
        elif file_sha256(target) != digest:
            raise ValueError("Existing immutable feature artifact is corrupt")
        sample["features"] = {"path": target.relative_to(output).as_posix(), "sha256": digest}
        atomic_json(result_file, sample)
    finally:
        temporary.unlink(missing_ok=True)


def load_example(output, result_file, row, text, plan):
    """Fail closed on corrupted/stale records; retain successes and retry explicit errors."""
    sample = json.loads(Path(result_file).read_text(encoding="utf-8"))
    expected, _ = empty_example(row, text, plan)
    for key in ("id", "label", "text", "source_media_sha256", "protocol_identity"):
        if sample.get(key) != expected[key]:
            raise ValueError(f"Resumed example has mismatched {key}: {row['id']}")
    if not isinstance(sample.get("conditions"), dict) or set(sample["conditions"]) != set(VARIANTS):
        raise ValueError("Resumed example must contain every condition")
    record = sample.get("features", {})
    digest = record.get("sha256", "")
    expected_path = f"features/{row['id'].replace(':', '-')}-{digest}.npz"
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or record.get("path") != expected_path:
        raise ValueError("Invalid resumed feature artifact identity")
    path = Path(output) / expected_path
    if not path.is_file() or file_sha256(path) != digest:
        raise ValueError("Resumed feature artifact is missing or its hash changed")
    with np.load(path, allow_pickle=False) as source:
        required = {"vision", "quality", "text", "valid", "condition_names", "id", "feature_identity", "protocol_identity", "source_media_sha256"}
        if set(source.files) != required:
            raise ValueError("Resumed feature artifact has an invalid schema")
        arrays = {name: source[name].copy() for name in required}
    for key, value in (("id", row["id"]), ("feature_identity", plan["feature_identity"]),
                       ("protocol_identity", plan["protocol_identity"]), ("source_media_sha256", row["media_sha256"])):
        if arrays[key].shape != () or arrays[key].item() != value:
            raise ValueError(f"Resumed feature artifact has mismatched {key}")
    if arrays["condition_names"].tolist() != list(VARIANTS):
        raise ValueError("Resumed feature condition order differs")
    for name, shape in (("vision", (len(VARIANTS), 1408)), ("quality", (len(VARIANTS), 3)), ("text", (1, 1024))):
        if arrays[name].shape != shape or arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Invalid resumed {name} feature shape/dtype/values")
    if not np.array_equal(arrays["text"], text):
        raise ValueError("Resumed text features differ from the fixed cached utterance")
    valid = arrays["valid"]
    if valid.shape != (len(VARIANTS),) or valid.dtype != bool:
        raise ValueError("Invalid resumed condition-valid mask")
    quality = arrays["quality"]
    if np.any((quality < 0) | (quality > 1)) or not np.isin(quality[:, 2], (0, 1)).all():
        raise ValueError("Invalid resumed quality values")
    for index, name in enumerate(VARIANTS):
        condition = sample["conditions"][name]
        if not isinstance(condition, dict):
            raise ValueError("Invalid resumed condition")
        if "error" in condition:
            if not isinstance(condition["error"], str) or valid[index] or np.any(arrays["vision"][index]) or np.any(quality[index]):
                raise ValueError("Failed conditions must have an explicit invalid/zero feature row")
            continue
        probabilities = np.asarray(condition.get("probabilities", []), dtype=float)
        info = condition.get("quality", {})
        available = bool(quality[index, 2])
        if (not valid[index] or probabilities.shape != (7,) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5)
                or condition.get("label") != LABELS[int(probabilities.argmax())]
                or info.get("available") is not available or not isinstance(info.get("reason"), str)
                or condition.get("source") != ("fusion" if available else "text_fallback")):
            raise ValueError("Invalid resumed predictions or availability contract")
        if available:
            if not np.isclose(np.linalg.norm(arrays["vision"][index]), 1, atol=2e-3):
                raise ValueError("Resumed visual features must be normalized")
            if not np.allclose(quality[index, :2], [info.get("valid_frame_fraction"), info.get("mean_detection_score")]):
                raise ValueError("Resumed quality metadata differs from features")
        elif np.any(arrays["vision"][index]) or np.any(quality[index]):
            raise ValueError("Unavailable vision must contain zero feature/quality values")
    return sample, arrays


def sample_indices(cache, per_class=16, seed=20260925):
    if per_class < 1:
        raise ValueError("per_class must be positive")
    rng = np.random.default_rng(seed)
    chosen = []
    for label in range(len(LABELS)):
        candidates = np.flatnonzero((cache["labels"] == label) & (cache["quality"][:, 2] > 0))
        chosen.extend(rng.choice(candidates, min(len(candidates), per_class), replace=False).tolist())
    if not chosen:
        raise ValueError("No visually eligible development examples")
    return np.array(sorted(chosen), dtype=np.int64)


def ffmpeg_command(source, target, variant):
    if variant not in VARIANTS or variant == "original":
        raise ValueError("Expected a supported non-original variant")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
               "-i", str(source), "-map", "0:v:0", "-an", "-map_metadata", "-1"]
    if variant == "silent_copy":
        command += ["-c:v", "copy"]
    else:
        if VARIANTS[variant]["filter"]:
            command += ["-vf", VARIANTS[variant]["filter"]]
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "1"]
    return command + [str(target)]


def metric_summary(truth, probabilities, reference=None):
    truth, probabilities = np.asarray(truth), np.asarray(probabilities)
    if len(truth) == 0 or probabilities.shape != (len(truth), 7) or not np.isfinite(probabilities).all():
        raise ValueError("Expected nonempty finite Nx7 probabilities and aligned labels")
    prediction = probabilities.argmax(axis=1)
    result = {"n": len(truth), "accuracy": float(np.mean(truth == prediction)),
              "macro_f1": float(f1_score(truth, prediction, labels=list(range(7)), average="macro", zero_division=0)),
              "weighted_f1": float(f1_score(truth, prediction, labels=list(range(7)), average="weighted", zero_division=0))}
    if reference is not None:
        reference = np.asarray(reference)
        if reference.shape != probabilities.shape:
            raise ValueError("Reference predictions must use the same cohort")
        result.update(label_flips=int(np.sum(prediction != reference.argmax(axis=1))),
                      mean_distribution_l1=float(np.abs(probabilities - reference).sum(axis=1).mean()),
                      max_distribution_l1=float(np.abs(probabilities - reference).sum(axis=1).max()))
    return result


def operational_predictions(heads, visual, textual, quality):
    """Use exactly the deployed availability switch, never zero-vision fusion."""
    visual, textual, quality = map(lambda x: np.asarray(x, dtype=np.float32), (visual, textual, quality))
    result = predict_probabilities(heads["text"], textual)
    available = quality[:, 2] > 0
    if available.any():
        combined = np.concatenate((visual[available], textual[available], quality[available]), axis=1)
        result[available] = predict_probabilities(heads["fusion"], combined)
    return result


def repeated_mismatch(cache, heads, repeats=20, seed=20260925):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    eligible = np.flatnonzero(cache["quality"][:, 2] > 0)
    visual, textual, quality, truth = (cache[k][eligible] for k in ("vision", "text", "quality", "labels"))
    reference = operational_predictions(heads, visual, textual, quality)
    fallback = predict_probabilities(heads["text"], textual)
    rows = []
    for trial in range(repeats):
        permutation = np.random.default_rng(seed + trial).permutation(len(eligible))
        probabilities = operational_predictions(heads, visual[permutation], textual, quality[permutation])
        rows.append({"seed": seed + trial, "fixed_points": int(np.sum(permutation == np.arange(len(eligible)))),
                     "visual_source_ids": cache["ids"][eligible][permutation].tolist(),
                     **metric_summary(truth, probabilities, reference)})
    return {"cohort": "all visually eligible development examples", "count": len(eligible),
            "paired": metric_summary(truth, reference), "deployed_text_fallback": metric_summary(truth, fallback, reference),
            "permutations": rows,
            "summary": {key: {"mean": float(np.mean([r[key] for r in rows])),
                               "min": float(min(r[key] for r in rows)), "max": float(max(r[key] for r in rows))}
                        for key in ("macro_f1", "weighted_f1", "label_flips", "mean_distribution_l1")},
            "interpretation": "Sensitivity to constructed mismatched evidence; shuffled pairs have no validated new emotion labels. Not evidence of causal emotional understanding."}


def run(home=None, output=None, per_class=16, seed=20260925, device="cuda", repeats=20):
    started = time.perf_counter()
    home = runtime_home(home)
    output = Path(output or home / "reports/robustness-dev").resolve()
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    cache = load_cache(home, "dev")
    feature_contract = verified_feature_contract(home, cache)
    selected = sample_indices(cache, per_class, seed)
    cohort = selected_manifest_rows(cache, selected, read_manifest(home, "dev"))
    by_id = {row["id"]: row for row in cohort}
    heads, hashes = {}, {}
    for stage in ("vision", "text", "fusion"):
        path = home / "checkpoints" / f"{stage}.pt"
        model, payload = load_head(path)
        if payload.get("stage") != stage or payload.get("feature_identity") != cache["feature_identity"] or payload.get("labels") != LABELS or payload.get("input_dim") != DIMENSIONS[stage]:
            raise ValueError("Checkpoint and feature contracts differ")
        heads[stage], hashes[stage] = model, file_sha256(path)
    plan = {"schema_version": 2, "split": "dev", "seed": seed, "per_class_limit": per_class, "device": device,
            "ids": cache["ids"][selected].tolist(), "variants": VARIANTS,
            "selected_manifest_rows": cohort, "feature_contract": feature_contract,
            "cache_sha256": cache["sha256"], "checkpoint_sha256": hashes,
            "feature_identity": cache["feature_identity"], "script_sha256": file_sha256(__file__),
            "selection": "Class-stratified random sample from visually eligible dev; no candidate selection or test examples used.",
            "fixed_text": "Cached exact text feature per utterance, unchanged across video conditions.",
            "feature_artifacts": "One immutable NPZ per saved example generation: normalized pooled visual features (8x1408), quality (8x3), fixed text (1x1024), valid-condition mask, ordered names and provenance. Error rows are invalid zeros, never text-fallback observations.",
            "repeats": repeats, "versions": {"torch": str(torch.__version__), "opencv": cv2.__version__,
                "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True,
                    timeout=30, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).stdout.splitlines()[0]}}
    plan["protocol_identity"] = identity(plan)
    plan_path = output / "protocol.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing run has a different protocol; choose another output directory")
    else:
        atomic_json(plan_path, plan)
    atomic_json(output / "mismatched-modality.json", repeated_mismatch(cache, heads, repeats, seed))
    processor = VideoProcessor(home / "models/vision/yunet.onnx")
    encoder = VisionEncoder(home / "models/vision/enet_b2_7.pt", device=device)
    cached_contract = feature_contract["feature_metadata"]
    if encoder.metadata() != cached_contract["vision"] or processor.version != cached_contract["processor"] or processor.frames != cached_contract["frames"] or processor.min_valid != cached_contract["min_valid"]:
        raise ValueError("Live visual encoder or sampling contract differs from the feature cache")
    results = []
    work = output / "transformed-media"
    work.mkdir(exist_ok=True)
    for number, i in enumerate(selected, 1):
        row_id = str(cache["ids"][i])
        row = by_id[row_id]
        source = Path(row["video_path"])
        if file_sha256(source) != row["media_sha256"]:
            raise ValueError(f"Source media changed: {row_id}")
        result_file = output / "examples" / f"{row_id.replace(':','-')}.json"
        if result_file.exists():
            sample, arrays = load_example(output, result_file, row, cache["text"][i:i+1], plan)
        else:
            sample, arrays = empty_example(row, cache["text"][i:i+1], plan)
        for condition_index, name in enumerate(VARIANTS):
            if "error" not in sample["conditions"][name]:
                continue
            before = time.perf_counter()
            target = source if name == "original" else work / f"{row_id.replace(':','-')}-{name}.mp4"
            command = None
            try:
                if name != "original":
                    command = ffmpeg_command(source, target, name)
                    subprocess.run(command, check=True, capture_output=True, timeout=180,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                crops, info = processor.process(target)
                visual = np.zeros((1, 1408), dtype=np.float32)
                quality = np.zeros((1, 3), dtype=np.float32)
                if crops:
                    visual[0] = l2_normalize(encoder.encode_faces(crops).mean(axis=0))
                    quality[0] = [info["valid_frame_fraction"], info["mean_detection_score"], 1.0]
                probability = operational_predictions(heads, visual, cache["text"][i:i+1], quality)[0]
                sample["conditions"][name] = {"probabilities": probability.tolist(), "label": LABELS[int(probability.argmax())],
                                              "quality": info, "source": "fusion" if crops else "text_fallback",
                                              "wall_ms": (time.perf_counter() - before) * 1000,
                                              "media_sha256": file_sha256(target),
                                              "feature_max_abs_delta_from_cached": float(np.max(np.abs(visual[0] - cache["vision"][i])))}
                arrays["vision"][condition_index] = visual[0]
                arrays["quality"][condition_index] = quality[0]
                arrays["valid"][condition_index] = True
            except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as error:
                sample["conditions"][name] = {"error": str(error), "command": command}
                arrays["vision"][condition_index] = 0
                arrays["quality"][condition_index] = 0
                arrays["valid"][condition_index] = False
            finally:
                # Temporary transformed TV media is never distributed or reused.
                if name != "original":
                    target.unlink(missing_ok=True)
            save_example(output, result_file, sample, arrays)
        results.append(sample)
        print(f"robustness {number}/{len(selected)} {row_id} {time.perf_counter()-started:.1f}s", flush=True)
    reference_rows = [row for row in results if "error" not in row["conditions"]["original"]]
    summaries = {}
    for name in VARIANTS:
        usable = [row for row in reference_rows if "error" not in row["conditions"][name]]
        probabilities = [row["conditions"][name]["probabilities"] for row in usable]
        reference = [row["conditions"]["original"]["probabilities"] for row in usable]
        truth = [LABELS.index(row["label"]) for row in usable]
        summaries[name] = {**metric_summary(truth, probabilities, reference),
                           "comparison_original": metric_summary(truth, reference),
                           "vision_available": sum(row["conditions"][name]["quality"]["available"] for row in usable),
                           "quality_reasons": dict(Counter(row["conditions"][name]["quality"]["reason"] for row in usable)),
                           "errors": len(results) - len(usable)} if usable else {"n": 0, "errors": len(results)}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": plan,
              "status": "complete_with_errors" if any(v["errors"] for v in summaries.values()) else "complete",
              "sample_count": len(results), "class_counts": dict(Counter(row["label"] for row in results)),
              "condition_count": len(VARIANTS), "summaries": summaries, "examples": results,
              "elapsed_this_invocation_seconds": time.perf_counter()-started,
              "limitations": ["Development data was used previously for head selection; this is not independent test accuracy.",
                              "Class-stratified convenience sample is not the natural MELD or webcam class distribution.",
                              "Video alterations may remove/change emotional evidence; label consistency metrics measure stress sensitivity, not guaranteed semantic invariance.",
                              "Classification uses cached text to isolate visual effects. This is not a capture, upload or generation latency benchmark.",
                              "Faces remain heuristic selections with unverified speaker identity; no clinical claim."]}
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home")
    parser.add_argument("--output")
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps({"status": report["status"], "samples": report["sample_count"], "summaries": report["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
