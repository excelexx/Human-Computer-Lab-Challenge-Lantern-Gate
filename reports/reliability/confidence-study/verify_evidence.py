"""Independently check saved evidence using NumPy; no project inference imports.

Run with the project environment: python verify_evidence.py
Optional: --heads PATH checks current head-file hashes; --record writes the first
verification.json alongside this script. No feature cache or media is opened.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    require(abs(float(actual) - float(expected)) < 1e-10, f"Metric mismatch: {name}: {actual} != {expected}")


def probability(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def counts(labels, names):
    return {name: int(np.count_nonzero(labels == i)) for i, name in enumerate(names)}


def scores(truth, logits, names):
    p = probability(logits)
    predicted = np.argmax(p, axis=1)
    confidence = np.max(p, axis=1)
    correct = (predicted == truth).astype(float)
    maximum = logits.max(axis=1)
    log_normalizer = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
    bin_id = np.minimum(np.floor(confidence * 10).astype(int), 9)
    bins, ece = [], 0.0
    for b in range(10):
        subset = bin_id == b
        n = int(subset.sum())
        accuracy = float(correct[subset].mean()) if n else None
        mean_confidence = float(confidence[subset].mean()) if n else None
        if n:
            ece += n / len(truth) * abs(accuracy - mean_confidence)
        bins.append({"count": n, "accuracy": accuracy, "mean_confidence": mean_confidence})
    return {"count": len(truth), "accuracy": float(correct.mean()),
            "nll_nats": float((log_normalizer - logits[np.arange(len(truth)), truth]).mean()),
            "multiclass_brier": float(((p - np.eye(7)[truth]) ** 2).sum(axis=1).mean()),
            "top_label_ece_10_equal_width_bins": ece, "mean_confidence": float(confidence.mean()),
            "support": counts(truth, names), "predicted_support": counts(predicted, names), "reliability_bins": bins}


def check_metrics(truth, logits, reported, names):
    computed = scores(truth, logits, names)
    for key in ("count", "support", "predicted_support"):
        require(computed[key] == reported[key], f"Count/support mismatch: {key}")
    for key in ("accuracy", "nll_nats", "multiclass_brier", "top_label_ece_10_equal_width_bins", "mean_confidence"):
        close(computed[key], reported[key], key)
    for actual, expected in zip(computed["reliability_bins"], reported["reliability_bins"]):
        require(actual["count"] == expected["count"], "Reliability bin count mismatch")
        if actual["count"]:
            close(actual["accuracy"], expected["accuracy"], "bin accuracy")
            close(actual["mean_confidence"], expected["mean_confidence"], "bin confidence")


def check_population(ids, truth, logits, reported, names):
    check_metrics(truth, logits, reported["metrics"], names)
    confidence = probability(logits).max(axis=1)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: (-confidence[i], str(ids[i]))))
    require([x["requested_coverage"] for x in reported["fixed_coverage"]] == [1.0, .8, .5], "Coverage fractions differ")
    for row in reported["fixed_coverage"]:
        count = int(np.ceil(row["requested_coverage"] * len(ids)))
        selected = order[:count]
        retained_hash = hashlib.sha256("\n".join(ids[selected].tolist()).encode()).hexdigest()
        require(row["retained_ids_sha256"] == retained_hash, "Confidence ranking/tie-break mismatch")
        close(row["actual_coverage"], count / len(ids), "actual coverage")
        check_metrics(truth[selected], logits[selected], row, names)


def verify(heads=None):
    manifest, protocol, result, started = (read(name) for name in ("artifact-integrity.json", "protocol.json", "results.json", "run-started.json"))
    for row in manifest["copied_evidence"]:
        path = HERE / row["file"]
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], f"Copied evidence changed: {row['file']}")
    protocol_hash = digest(HERE / "protocol.json")
    require(result["protocol_sha256"] == started["protocol_sha256"] == protocol_hash, "Protocol hash references differ")
    require(digest(HERE / "source_snapshot.py") == protocol["study_source_sha256"], "Frozen source hash differs")
    require(result["active_head_sha256"] == protocol["active_head_sha256"], "Head references differ")
    require(result["cache_sha256"] == protocol["cache_sha256"], "Cache references differ")
    require(result["feature_identity"] == protocol["feature_identity"], "Feature identity differs")
    frozen, began, completed = (datetime.fromisoformat(x) for x in (protocol["frozen_at"], started["started_at"], result["completed_at"]))
    require(frozen < began <= completed, "Recorded protocol/run chronology is inconsistent")
    with np.load(HERE / "dev-confidence.npz", allow_pickle=False) as data:
        require(set(data.files) == {"ids", "labels", "routes", "text_logits", "operational_logits", "oof_scaled_logits", "fold_assignment", "oof_temperatures"}, "Unexpected per-example artifact fields")
        a = {name: data[name].copy() for name in data.files}
    ids, truth, routes = a["ids"], a["labels"], a["routes"]
    require(len(ids) == len(set(ids.tolist())) == result["development_count"] == 1109, "Unexpected development population")
    require(all(str(i).startswith("dev:") for i in ids), "Non-development ID found")
    require(set(routes.tolist()) == {"fusion", "text_fallback"}, "Unexpected route")
    require(set(a["fold_assignment"].tolist()) == set(range(5)), "Unexpected fold assignment")
    names = protocol["labels"]
    groups = np.array([str(i).rsplit(":", 1)[0] for i in ids])
    for group in set(groups.tolist()):
        require(len(set(a["fold_assignment"][groups == group].tolist())) == 1, "Dialogue crosses held folds")
    require(len(set(groups.tolist())) == result["development_dialogues"] == 114, "Dialogue count differs")
    for field in ("text_logits", "operational_logits", "oof_scaled_logits"):
        require(a[field].shape == (1109, 7) and np.isfinite(a[field]).all(), "Invalid saved logits")
    fallback = routes == "text_fallback"
    require(np.array_equal(a["operational_logits"][fallback], a["text_logits"][fallback]), "Fallback does not equal the text head")
    require(np.all((a["oof_temperatures"] >= .25) & (a["oof_temperatures"] <= 4)), "Out-of-bounds temperature")
    require(np.array_equal(a["operational_logits"] / a["oof_temperatures"][:, None], a["oof_scaled_logits"]), "Scaled logits mismatch")
    require(np.array_equal(a["operational_logits"].argmax(axis=1), a["oof_scaled_logits"].argmax(axis=1)), "Argmax changed")
    populations = {"all": np.ones(len(ids), dtype=bool), "fusion": ~fallback, "text_fallback": fallback}
    for key, field in (("raw_operational", "operational_logits"), ("crossfit_temperature_operational", "oof_scaled_logits")):
        for population, mask in populations.items():
            check_population(ids[mask], truth[mask], a[field][mask], result[key][population], names)
    for name, mask in (("all", populations["all"]), ("fusion_eligible", populations["fusion"])):
        check_population(ids[mask], truth[mask], a["text_logits"][mask], result["raw_text_diagnostics"][name], names)
    seen = set()
    for row in result["folds"]:
        key = (row["fold"], row["route"])
        require(key not in seen, "Repeated fold/route")
        seen.add(key)
        held = (a["fold_assignment"] == row["fold"]) & (routes == row["route"])
        fit = (a["fold_assignment"] != row["fold"]) & (routes == row["route"])
        require(not set(groups[fit]) & set(groups[held]), "Fit/held dialogue overlap")
        require(int(fit.sum()) == row["fit_count"] and int(held.sum()) == row["held_count"], "Fold row counts differ")
        require(counts(truth[fit], names) == row["fit_support"] and counts(truth[held], names) == row["held_support"], "Fold support differs")
        require(len(set(groups[fit].tolist())) == row["fit_dialogue_count"] and len(set(groups[held].tolist())) == row["held_dialogue_count"], "Fold dialogue counts differ")
        t = row["temperature"]
        require(np.all(a["oof_temperatures"][held] == t), "Fold temperature differs")
        close(scores(truth[fit], a["operational_logits"][fit], names)["nll_nats"], row["fit_nll_before"], "fit NLL before")
        close(scores(truth[fit], a["operational_logits"][fit] / t, names)["nll_nats"], row["fit_nll_after"], "fit NLL after")
        check_metrics(truth[held], a["operational_logits"][held], row["held_before"], names)
        check_metrics(truth[held], a["oof_scaled_logits"][held], row["held_after"], names)
    require(seen == {(fold, route) for fold in range(5) for route in ("fusion", "text_fallback")}, "Missing fold/route results")
    active_verified = None
    if heads is not None:
        active_verified = {stage: digest(heads / f"{stage}.pt") == expected for stage, expected in protocol["active_head_sha256"].items()}
        require(all(active_verified.values()), "Supplied head directory differs from recorded active heads")
    return {"verified_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
            "copied_evidence_byte_hashes_verified": True, "protocol_and_source_hashes_verified": True,
            "recorded_chronology_consistent": True, "development_rows": len(ids), "dialogues": len(set(groups.tolist())),
            "all_saved_metric_coverage_and_fold_values_recomputed": True,
            "held_dialogue_route_isolation_verified": True, "argmax_unchanged": True,
            "supplied_active_head_hashes_verified": active_verified,
            "original_model_inference_repeated": False, "original_feature_caches_read_or_included": False,
            "media_or_model_binaries_included": False,
            "scope": "Independent numeric and provenance checks on the saved development logits; not a new model evaluation, calibration deployment, or independent generalization study."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=Path)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    report = verify(args.heads)
    if args.record:
        with (HERE / "verification.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
