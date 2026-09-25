"""CPU-only replay of a fixed fallback replacement on saved dev stress features.

No video processing, encoder inference, fitting, test access or model selection.
The original extraction must finish before this comparison can run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluate import classification_metrics
from .models import DIMENSIONS, load_head
from .robustness import VARIANTS, identity, load_example, operational_predictions
from .settings import LABELS
from .train import atomic_json, file_sha256, load_cache


BASELINE_ATOL = 2e-6


def compare_predictions(baseline_heads, candidate_text, visual, textual, quality, recorded):
    """Reproduce the old path first; only unavailable-vision outputs may change."""
    recorded = np.asarray(recorded, dtype=np.float32)
    baseline = operational_predictions(baseline_heads, visual, textual, quality)
    if recorded.shape != baseline.shape or not np.isfinite(recorded).all():
        raise ValueError("Saved baseline probabilities are malformed or misaligned.")
    delta = np.abs(recorded - baseline)
    if not np.allclose(recorded, baseline, atol=BASELINE_ATOL, rtol=0):
        raise ValueError("Preserved baseline does not reproduce saved robustness probabilities.")
    if not np.array_equal(recorded.argmax(1), baseline.argmax(1)):
        raise ValueError("Preserved baseline changed a recorded classification label.")
    candidate_heads = {**baseline_heads, "text": candidate_text}
    candidate = operational_predictions(candidate_heads, visual, textual, quality)
    available = np.asarray(quality)[:, 2] > 0
    if not np.array_equal(candidate[available], baseline[available]):
        raise AssertionError("A fallback-only replacement changed an eligible fusion output.")
    return baseline, candidate, float(delta.max()) if len(delta) else 0.0


def _verified_head(path, stage, expected_sha, feature_id, expected_caches):
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"Frozen {stage} checkpoint hash differs: {path}")
    model, payload = load_head(path, device="cpu")
    if (payload.get("stage") != stage or payload.get("input_dim") != DIMENSIONS[stage]
            or payload.get("labels") != LABELS or payload.get("feature_identity") != feature_id
            or payload.get("cache_sha256") != expected_caches):
        raise ValueError(f"Frozen {stage} checkpoint metadata does not match the feature study.")
    if not isinstance(payload.get("train_count"), int) or payload["train_count"] < 1:
        raise ValueError(f"Frozen {stage} checkpoint lacks recorded training examples.")
    return model, {"sha256": actual_sha, "stage": stage, "architecture": payload.get("architecture", "mlp"),
                   "parameters": sum(value.numel() for value in model.parameters()),
                   "train_count": payload["train_count"], "feature_identity": feature_id,
                   "cache_sha256": expected_caches}


def _metrics(rows, key):
    if not rows:
        return {"status": "unavailable", "count": 0}
    truth = np.array([row["true_label_id"] for row in rows])
    predictions = np.array([row[key] for row in rows]).argmax(1)
    return classification_metrics(truth, predictions)


def _paired_summary(rows):
    baseline, candidate = _metrics(rows, "baseline_probabilities"), _metrics(rows, "candidate_probabilities")
    if not rows:
        return {"n": 0, "baseline": baseline, "candidate": candidate}
    bp = np.array([row["baseline_probabilities"] for row in rows])
    cp = np.array([row["candidate_probabilities"] for row in rows])
    return {"n": len(rows), "baseline": baseline, "candidate": candidate,
            "candidate_minus_baseline": {key: candidate[key] - baseline[key] for key in ("macro_f1", "weighted_f1", "accuracy")},
            "candidate_baseline_label_changes": int(np.sum(bp.argmax(1) != cp.argmax(1))),
            "candidate_baseline_mean_distribution_l1": float(np.abs(cp - bp).sum(1).mean())}


def summarize_conditions(records):
    """Pair on identical successful rows; expose every failure and routing switch."""
    summaries = {}
    for name in VARIANTS:
        rows, failures = [], []
        for example in records:
            condition = example["conditions"][name]
            if "error" in condition:
                failures.append({"id": example["id"], "error": condition["error"]})
            else:
                rows.append({"id": example["id"], "true_label_id": example["true_label_id"], **condition})
        paired_original = []
        transitions = {name: [] for name in ("eligible_to_eligible", "eligible_to_fallback", "fallback_to_eligible", "fallback_to_fallback")}
        by_id = {row["id"]: row for row in rows}
        for example in records:
            condition = by_id.get(example["id"])
            original = example["conditions"]["original"]
            if condition is None or "error" in original:
                continue
            route = ("eligible" if original["vision_available"] else "fallback") + "_to_" + ("eligible" if condition["vision_available"] else "fallback")
            transitions[route].append(condition)
            paired_original.append((condition, original))
        shifts = {}
        for model in ("baseline", "candidate"):
            key = model + "_probabilities"
            current = np.array([row[key] for row, _ in paired_original])
            original = np.array([row[key] for _, row in paired_original])
            shifts[model] = {"paired_n": len(paired_original),
                             "label_changes_vs_own_original": int(np.sum(current.argmax(1) != original.argmax(1))) if len(current) else 0,
                             "mean_distribution_l1_vs_own_original": float(np.abs(current - original).sum(1).mean()) if len(current) else None}
        summaries[name] = {**_paired_summary(rows), "requested_n": len(records),
                           "scored_coverage_fraction": len(rows) / len(records) if records else None,
                           "vision_available": sum(row["vision_available"] for row in rows),
                           "fallback_count": sum(not row["vision_available"] for row in rows),
                           "fallback_fraction_of_scored": sum(not row["vision_available"] for row in rows) / len(rows) if rows else None,
                           "quality_reasons": dict(Counter(row["quality_reason"] for row in rows)),
                           "errors": failures, "original_pair_omissions": len(rows) - len(paired_original),
                           "availability_transitions": {name: _paired_summary(values) for name, values in transitions.items()},
                           "shift_vs_own_original": shifts}
    return summaries


def run(home, robustness, baseline_dir, candidate, definition, output):
    started = time.perf_counter()
    home, robustness, baseline_dir, candidate, definition, output = map(Path, (home, robustness, baseline_dir, candidate, definition, output))
    report_path = robustness / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError("Original robustness extraction must finish before comparison.")
    original_report = json.loads(report_path.read_text(encoding="utf-8"))
    plan = json.loads((robustness / "protocol.json").read_text(encoding="utf-8"))
    if original_report.get("status") not in {"complete", "complete_with_errors"} or original_report.get("protocol") != plan:
        raise ValueError("Original robustness report is incomplete or its protocol differs.")
    if (plan.get("split") != "dev" or plan.get("schema_version") != 2
            or identity({key: value for key, value in plan.items() if key != "protocol_identity"}) != plan.get("protocol_identity")):
        raise ValueError("Expected immutable schema-v2 development robustness protocol.")
    if plan.get("variants") != VARIANTS:
        raise ValueError("Variant definitions differ from the saved feature protocol.")
    frozen = json.loads(definition.read_text(encoding="utf-8"))
    if frozen.get("baseline_head_sha256") != plan["checkpoint_sha256"] or frozen.get("feature_identity") != plan["feature_identity"]:
        raise ValueError("Hybrid definition and original robustness baseline differ.")
    cache = load_cache(home, "dev")
    expected_caches = frozen["train_dev_cache_sha256"]
    if (cache["sha256"] != plan["cache_sha256"] or cache["feature_identity"] != plan["feature_identity"]
            or expected_caches.get("dev") != cache["sha256"]):
        raise ValueError("Saved feature protocol does not match the development cache.")
    torch.set_num_threads(2)
    heads, checkpoint_provenance = {}, {}
    for stage in ("vision", "text", "fusion"):
        heads[stage], checkpoint_provenance[stage] = _verified_head(baseline_dir / f"{stage}.pt", stage,
            plan["checkpoint_sha256"][stage], plan["feature_identity"], expected_caches)
    candidate_text, candidate_provenance = _verified_head(candidate, "text", frozen["prepared_text_sha256"], plan["feature_identity"], expected_caches)
    cached_index = {str(row_id): index for index, row_id in enumerate(cache["ids"])}
    manifest_rows = plan["selected_manifest_rows"]
    if len({row["id"] for row in manifest_rows}) != len(manifest_rows) or [row["id"] for row in manifest_rows] != plan["ids"]:
        raise ValueError("Selected robustness identities are duplicated or reordered.")
    report_rows = original_report["examples"]
    if [row["id"] for row in report_rows] != plan["ids"] or original_report.get("sample_count") != len(manifest_rows):
        raise ValueError("Final robustness report does not contain the complete declared cohort.")
    saved_by_id = {row["id"]: row for row in report_rows}
    records, artifacts, max_reproduction_error = [], [], 0.0
    for row in manifest_rows:
        index = cached_index.get(row["id"])
        if index is None or row["label"] != int(cache["labels"][index]) or row.get("split") != "dev":
            raise ValueError("Declared robustness row does not resolve to its dev feature/label.")
        sample_path = robustness / "examples" / f"{row['id'].replace(':', '-')}.json"
        sample, arrays = load_example(robustness, sample_path, row, cache["text"][index:index + 1], plan)
        if sample != saved_by_id[row["id"]]:
            raise ValueError("An example changed after the original report was finalized.")
        valid = arrays["valid"]
        record = {"id": row["id"], "true_label_id": row["label"], "text": row["text"], "conditions": {}}
        artifacts.append({"id": row["id"], "example_sha256": file_sha256(sample_path), **sample["features"]})
        if valid.any():
            names = np.array(list(VARIANTS))[valid].tolist()
            recorded = [sample["conditions"][name]["probabilities"] for name in names]
            textual = np.repeat(arrays["text"], int(valid.sum()), axis=0)
            baseline_p, candidate_p, error = compare_predictions(heads, candidate_text, arrays["vision"][valid], textual, arrays["quality"][valid], recorded)
            max_reproduction_error = max(max_reproduction_error, error)
            for index_in_batch, name in enumerate(names):
                original = sample["conditions"][name]
                record["conditions"][name] = {
                    "vision_available": original["quality"]["available"], "quality_reason": original["quality"]["reason"], "source": original["source"],
                    "baseline_probabilities": baseline_p[index_in_batch].tolist(), "candidate_probabilities": candidate_p[index_in_batch].tolist(),
                    "baseline_label": LABELS[int(baseline_p[index_in_batch].argmax())], "candidate_label": LABELS[int(candidate_p[index_in_batch].argmax())]}
        for condition_index, name in enumerate(VARIANTS):
            if not valid[condition_index]:
                record["conditions"][name] = {"error": sample["conditions"][name]["error"]}
        records.append(record)
    summaries = summarize_conditions(records)
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed hybrid vs preserved baseline on identical saved dev corruption features; post-hoc stress replay, no selection or new holdout.",
        "split": "dev", "device": "cpu", "samples": len(records), "conditions": list(VARIANTS),
        "status": "complete_with_extraction_errors" if any(item["errors"] for item in summaries.values()) else "complete",
        "provenance": {"original_report_sha256": file_sha256(report_path), "protocol_identity": plan["protocol_identity"],
                       "protocol_file_sha256": file_sha256(robustness / "protocol.json"), "hybrid_definition_sha256": file_sha256(definition),
                       "dev_cache_sha256": cache["sha256"], "feature_identity": cache["feature_identity"],
                       "baseline_checkpoints": checkpoint_provenance, "candidate_text_checkpoint": candidate_provenance,
                       "comparison_source_sha256": file_sha256(__file__), "feature_artifacts": artifacts},
        "verification": {"baseline_probability_max_abs_error": max_reproduction_error, "baseline_probability_atol": BASELINE_ATOL,
                         "baseline_labels_reproduced": True, "all_eligible_candidate_probabilities_exactly_unchanged": True,
                         "successful_condition_rows": sum(item["n"] for item in summaries.values()),
                         "eligible_condition_rows_verified": sum(item["vision_available"] for item in summaries.values())},
        "summaries": summaries, "examples": records, "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["No test cache, encoder, camera, language generator, fitting or threshold selection was used.",
                        "Development data already influenced earlier model selection; this is not independent accuracy evidence.",
                        "The source sample is stratified by class and visual availability; its fallback fractions are stress coverage, not natural deployment prevalence.",
                        "Altered video can remove or change emotional evidence; scores measure label-consistency stress sensitivity rather than proven semantic invariance.",
                        "Candidate changes only text fallback; any eligible classification change is a hard error. Benefit cannot be attributed to better vision.",
                        "Failures are retained and excluded from scoring; they are never relabeled as successful missing-vision observations."]}
    atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("home", "robustness", "baseline-dir", "candidate", "definition", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    report = run(**vars(parser.parse_args()))
    print(json.dumps({"status": report["status"], "samples": report["samples"], "verification": report["verification"],
                      "conditions": {name: {key: value[key] for key in ("n", "vision_available", "fallback_count", "candidate_minus_baseline", "candidate_baseline_label_changes") if key in value}
                                     for name, value in report["summaries"].items()}}, indent=2))


if __name__ == "__main__":
    main()
