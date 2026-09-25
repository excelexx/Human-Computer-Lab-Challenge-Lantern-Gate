"""Frozen-protocol, CPU-only confidence diagnostics on MELD development data.

Run ``python -m checkin.confidence_study --home ... --output ... plan`` before
``... run``. No test data, encoder inference, generator, or production edits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import torch

from .models import DIMENSIONS, load_head
from .settings import LABELS
from .train import atomic_json, file_sha256, load_cache


FOLDS, SEED, BINS = 5, 20260925, 10
COVERAGES = (1.0, 0.8, 0.5)
TEMPERATURE_BOUNDS = (0.25, 4.0)
ROUTES = ("fusion", "text_fallback")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def supports(labels):
    return {label: int(count) for label, count in zip(LABELS, np.bincount(labels, minlength=7))}


def dialogue_groups(ids):
    parts = [str(identity).split(":") for identity in ids]
    if any(len(p) != 3 or p[0] != "dev" or not p[1].isdigit() or not p[2].isdigit() for p in parts):
        raise ValueError("Confidence study requires official development dialogue IDs.")
    return np.asarray([":".join(p[:2]) for p in parts])


def validate_arrays(labels, logits):
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or logits.shape != (len(labels), 7):
        raise ValueError("Expected integer labels and Nx7 logits.")
    if not np.isfinite(logits).all() or np.any((labels < 0) | (labels >= 7)):
        raise ValueError("Labels/logits are outside the finite seven-class contract.")
    return labels, logits


def negative_log_likelihood(labels, logits):
    labels, logits = validate_arrays(labels, logits)
    if not len(labels):
        raise ValueError("Cannot fit or score an empty population.")
    return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]))


def metrics(labels, logits):
    """Equal-width top-label ECE; multiclass Brier uses the sum over 7 labels."""
    labels, logits = validate_arrays(labels, logits)
    if not len(labels):
        return {"count": 0, "status": "unavailable_empty_population", "support": supports(labels)}
    probability = softmax(logits, axis=1)
    predicted = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = (predicted == labels).astype(np.float64)
    assignments = np.minimum((confidence * BINS).astype(np.int64), BINS - 1)
    bins, ece = [], 0.0
    for index in range(BINS):
        mask = assignments == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        average_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(accuracy - average_confidence)
        bins.append({"index": index, "lower_inclusive": index / BINS, "upper": (index + 1) / BINS,
                     "upper_inclusive": index == BINS - 1, "count": count,
                     "accuracy": accuracy, "mean_confidence": average_confidence})
    return {"count": len(labels), "status": "measured_on_development", "support": supports(labels),
            "predicted_support": supports(predicted), "accuracy": float(correct.mean()),
            "nll_nats": negative_log_likelihood(labels, logits),
            "multiclass_brier": float(np.square(probability - np.eye(7)[labels]).sum(axis=1).mean()),
            "top_label_ece_10_equal_width_bins": float(ece), "mean_confidence": float(confidence.mean()),
            "reliability_bins": bins}


def coverage_report(ids, labels, logits):
    """Fixed coverage fractions are diagnostics, not selected thresholds."""
    labels, logits = validate_arrays(labels, logits)
    ids = np.asarray(ids)
    if ids.shape != labels.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("Coverage requires aligned, unique IDs.")
    confidence = softmax(logits, axis=1).max(axis=1)
    order = np.lexsort((ids, -confidence))
    rows = []
    for fraction in COVERAGES:
        count = int(np.ceil(len(ids) * fraction))
        selected = order[:count]
        rows.append({"requested_coverage": fraction, "actual_coverage": count / len(ids) if len(ids) else None,
                     "retained_ids_sha256": hashlib.sha256("\n".join(ids[selected].tolist()).encode()).hexdigest(),
                     **metrics(labels[selected], logits[selected])})
    return rows


def fit_temperature(labels, logits):
    labels, logits = validate_arrays(labels, logits)
    objective = lambda temperature: negative_log_likelihood(labels, logits / temperature)
    optimized = minimize_scalar(objective, method="bounded", bounds=TEMPERATURE_BOUNDS,
                                options={"xatol": 1e-5, "maxiter": 200})
    if not optimized.success:
        raise RuntimeError(f"Temperature optimizer failed: {optimized.message}")
    # Include exact bounds and the uncalibrated T=1 reference in the declared fit.
    choices = (TEMPERATURE_BOUNDS[0], float(optimized.x), 1.0, TEMPERATURE_BOUNDS[1])
    temperature = min(choices, key=lambda value: (objective(value), abs(value - 1.0)))
    return temperature, {"temperature": temperature, "fit_count": len(labels), "fit_support": supports(labels),
                         "fit_nll_before": objective(1.0), "fit_nll_after": objective(temperature),
                         "optimizer_evaluations": int(optimized.nfev),
                         "at_fixed_bound": temperature in TEMPERATURE_BOUNDS}


def crossfit_temperatures(ids, labels, logits, routes):
    labels, logits = validate_arrays(labels, logits)
    ids, routes = np.asarray(ids), np.asarray(routes)
    if routes.shape != labels.shape or set(routes.tolist()) != set(ROUTES):
        raise ValueError("Both declared operational routes are required.")
    groups = dialogue_groups(ids)
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    assignment = np.full(len(labels), -1, dtype=np.int64)
    temperature = np.full(len(labels), np.nan, dtype=np.float64)
    rows = []
    for fold, (fit_rows, held_rows) in enumerate(splitter.split(logits, labels, groups)):
        if set(groups[fit_rows]) & set(groups[held_rows]):
            raise AssertionError("Dialogue leakage in cross-fitting.")
        assignment[held_rows] = fold
        for route in ROUTES:
            fit = fit_rows[routes[fit_rows] == route]
            held = held_rows[routes[held_rows] == route]
            if not len(fit) or not len(held):
                raise ValueError("Each fold needs fitting and held rows for both routes.")
            fitted, report = fit_temperature(labels[fit], logits[fit])
            temperature[held] = fitted
            rows.append({"fold": fold, "route": route, "fit_dialogue_count": len(set(groups[fit].tolist())),
                         "held_dialogue_count": len(set(groups[held].tolist())), "held_count": len(held),
                         "held_support": supports(labels[held]), **report,
                         "held_before": metrics(labels[held], logits[held]),
                         "held_after": metrics(labels[held], logits[held] / fitted)})
    if np.any(assignment < 0) or not np.isfinite(temperature).all():
        raise AssertionError("Every development row must receive one held-fold temperature.")
    scaled = logits / temperature[:, None]
    changed = int((logits.argmax(axis=1) != scaled.argmax(axis=1)).sum())
    if changed:
        raise AssertionError("Positive scalar temperatures must not change predicted labels.")
    return {"scaled_logits": scaled, "temperatures": temperature, "fold_assignment": assignment,
            "folds": rows, "changed_argmax_labels": changed}


def head_contract(home, plan=None):
    models, metadata, hashes = {}, {}, {}
    for stage, dimension in DIMENSIONS.items():
        path = home / "checkpoints" / f"{stage}.pt"
        digest = file_sha256(path)
        if plan and digest != plan["active_head_sha256"][stage]:
            raise ValueError("Active head changed after protocol freeze.")
        model, payload = load_head(path, device="cpu")
        if payload.get("stage") != stage or payload.get("input_dim") != dimension or payload.get("labels") != LABELS:
            raise ValueError("Active head label/stage/dimension contract differs.")
        models[stage], hashes[stage] = model, digest
        metadata[stage] = {"architecture": payload.get("architecture", "mlp"),
                           "feature_identity": payload.get("feature_identity"), "cache_sha256": payload.get("cache_sha256"),
                           "parameters": sum(parameter.numel() for parameter in model.parameters())}
    return models, metadata, hashes


def plan_study(home, output):
    home, output = Path(home).resolve(), Path(output).resolve()
    if output == home or home in output.parents:
        raise ValueError("Study output must remain outside the production artifact tree.")
    if (output / "protocol.json").exists():
        raise FileExistsError("Protocol already exists; do not rewrite a frozen study.")
    train, dev = load_cache(home, "train"), load_cache(home, "dev")
    if train["feature_identity"] != dev["feature_identity"]:
        raise ValueError("Training and development features differ.")
    _, head_metadata, head_hashes = head_contract(home)
    cache_hashes = {"train": train["sha256"], "dev": dev["sha256"]}
    if any(item["feature_identity"] != dev["feature_identity"] or item["cache_sha256"] != cache_hashes for item in head_metadata.values()):
        raise ValueError("Active heads do not match the current feature caches.")
    protocol = {
        "schema_version": 1, "frozen_at": utc_now(), "home": str(home), "labels": LABELS,
        "study_source_sha256": file_sha256(Path(__file__)), "feature_identity": dev["feature_identity"],
        "cache_sha256": cache_hashes, "active_head_sha256": head_hashes, "head_metadata": head_metadata,
        "population": "Official development split only; no test cache access.",
        "route": "fusion when cached quality availability is 1; active text head otherwise",
        "diagnostics": ["operational all", "fusion route", "text fallback route", "text-only all", "text-only on fusion-eligible rows"],
        "metrics": {"accuracy": "argmax", "nll": "natural log loss from stable logsumexp, in nats",
                    "brier": "Mean sum over seven classes of (p - onehot)^2; range 0..2",
                    "ece": "Top-label ECE, ten fixed equal-width confidence bins; last bin includes confidence 1",
                    "support": "All seven true-label and predicted-label counts retained, including zeros"},
        "fixed_coverages": list(COVERAGES), "coverage_rule": "Keep ceil(fraction*N) highest-confidence rows within each reported population; ties broken lexicographically by ID. No threshold optimization.",
        "controlled_experiment": {"method": "Five-fold stratified dialogue-group cross-fitting on dev",
                                  "folds": FOLDS, "seed": SEED, "routes": list(ROUTES),
                                  "fit": "One positive scalar temperature per route, using only the other four folds; minimize NLL",
                                  "bounds": list(TEMPERATURE_BOUNDS), "optimizer": "bounded minimize_scalar, xatol=1e-5, maxiter=200; also compare exact endpoints and T=1",
                                  "prediction": "Each row uses only its held-fold route temperature; report every fold/support/temperature and unchanged argmax",
                                  "no_final_fit": "Do not fit deployable full-dev temperatures or select a confidence threshold"},
        "limitations": ["Development data already influenced head selection/confirmation; this study is not independent generalization evidence.",
                        "Cross-fitting protects the temperature fit from its held dialogue labels, not from earlier head-selection bias.",
                        "ECE depends on binning and sample size; differences are descriptive without a significance claim.",
                        "Top-confidence subsets may disproportionately retain common/easier classes; report supports.",
                        "No deployment, production parameter change, threshold selection, model retraining, encoder inference, or GPU usage."],
        "resource": "CPU only; PyTorch and BLAS limited to two threads",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(output / "protocol.json", protocol)
    return protocol


def collect_logits(model, features):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(features), 512):
            value = model(torch.from_numpy(features[start:start + 512])).detach().cpu().numpy()
            rows.append(value.astype(np.float64))
    return np.concatenate(rows)


def population_reports(ids, labels, logits, routes):
    result = {}
    for population, mask in (("all", np.ones(len(labels), dtype=bool)), *[(name, routes == name) for name in ROUTES]):
        result[population] = {"metrics": metrics(labels[mask], logits[mask]),
                              "fixed_coverage": coverage_report(ids[mask], labels[mask], logits[mask])}
    return result


def run_study(home, output):
    home, output = Path(home).resolve(), Path(output).resolve()
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["study_source_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("Study source changed after protocol freeze; do not silently revise the protocol.")
    if protocol["home"] != str(home):
        raise ValueError("Protocol references another artifact directory.")
    for split, digest in protocol["cache_sha256"].items():
        if split not in {"train", "dev"} or file_sha256(home / "cache" / f"{split}.npz") != digest:
            raise ValueError("Train/dev feature bytes changed after protocol freeze.")
    models, _, _ = head_contract(home, protocol)
    with (output / "run-started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": utc_now(), "protocol_sha256": file_sha256(protocol_path)}, handle, indent=2)
    dev = load_cache(home, "dev")
    available = dev["quality"][:, 2] > 0
    routes = np.where(available, "fusion", "text_fallback")
    text_logits = collect_logits(models["text"], dev["text"])
    fusion_features = np.concatenate((dev["vision"], dev["text"], dev["quality"]), axis=1)
    fusion_logits = collect_logits(models["fusion"], fusion_features)
    operational = np.where(available[:, None], fusion_logits, text_logits)
    experiment = crossfit_temperatures(dev["ids"], dev["labels"], operational, routes)
    before = population_reports(dev["ids"], dev["labels"], operational, routes)
    after = population_reports(dev["ids"], dev["labels"], experiment["scaled_logits"], routes)
    report = {
        "schema_version": 1, "completed_at": utc_now(), "protocol_sha256": file_sha256(protocol_path),
        "active_head_sha256": protocol["active_head_sha256"], "feature_identity": protocol["feature_identity"],
        "cache_sha256": protocol["cache_sha256"], "development_count": len(dev["ids"]),
        "development_dialogues": len(set(dialogue_groups(dev["ids"]).tolist())),
        "raw_operational": before, "crossfit_temperature_operational": after,
        "raw_text_diagnostics": {
            "all": {"metrics": metrics(dev["labels"], text_logits), "fixed_coverage": coverage_report(dev["ids"], dev["labels"], text_logits)},
            "fusion_eligible": {"metrics": metrics(dev["labels"][available], text_logits[available]),
                                "fixed_coverage": coverage_report(dev["ids"][available], dev["labels"][available], text_logits[available])}},
        "folds": experiment["folds"], "changed_argmax_labels": experiment["changed_argmax_labels"],
        "interpretation": protocol["limitations"], "test_accessed": False,
        "production_changed": False, "threshold_selected": False, "deployable_temperature_fitted": False,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": "cpu", "threads": 2},
    }
    with (output / "dev-confidence.npz").open("wb") as handle:
        np.savez_compressed(handle, ids=dev["ids"], labels=dev["labels"], routes=routes, text_logits=text_logits,
                            operational_logits=operational, oof_scaled_logits=experiment["scaled_logits"],
                            fold_assignment=experiment["fold_assignment"], oof_temperatures=experiment["temperatures"])
    head_contract(home, protocol)  # Prove active checkpoint bytes stayed unchanged.
    atomic_json(output / "results.json", report)
    write_summary(output, report)
    return report


def write_summary(output, report):
    lines = ["# Development-only confidence study", "",
             "This is a diagnostic study, not independent generalization evidence: these development examples already influenced head selection or confirmation. No confidence threshold, deployable calibration, or production change was selected.", "",
             "The protocol was frozen before computing results. Five dialogue-group folds fit one bounded positive temperature per operational route on the other four folds only. NLL uses natural logs; multiclass Brier sums errors across seven classes; ECE uses ten fixed equal-width bins.", "",
             "| Population | N | Accuracy | Raw NLL | Cross-fit NLL | Raw Brier | Cross-fit Brier | Raw ECE | Cross-fit ECE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for population in ("all", *ROUTES):
        raw = report["raw_operational"][population]["metrics"]
        calibrated = report["crossfit_temperature_operational"][population]["metrics"]
        lines.append(f"| {population} | {raw['count']} | {raw['accuracy']:.4f} | {raw['nll_nats']:.4f} | {calibrated['nll_nats']:.4f} | {raw['multiclass_brier']:.4f} | {calibrated['multiclass_brier']:.4f} | {raw['top_label_ece_10_equal_width_bins']:.4f} | {calibrated['top_label_ece_10_equal_width_bins']:.4f} |")
    lines += ["", f"Changed argmax labels: **{report['changed_argmax_labels']}**. Temperatures change probability magnitudes, not predicted categories.", "",
              "## Temperatures and fold support", "", "| Fold | Route | Fit rows | Held rows | Temperature | Held NLL before | Held NLL after |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["folds"]:
        lines.append(f"| {row['fold'] + 1} | {row['route']} | {row['fit_count']} | {row['held_count']} | {row['temperature']:.4f} | {row['held_before']['nll_nats']:.4f} | {row['held_after']['nll_nats']:.4f} |")
    lines += ["", "## Fixed coverage diagnostics", "",
              "Each population keeps its highest-confidence ceil(fraction × N) rows, with ID-based tie-breaking. These fixed slices were not optimized into deployment thresholds. Class supports and reliability-bin counts for every slice are retained in results.json.", "",
              "| Population | Confidence method | Requested coverage | Retained rows | Accuracy | NLL | ECE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for population in ("all", *ROUTES):
        for method, key in (("raw", "raw_operational"), ("cross-fit temperature", "crossfit_temperature_operational")):
            for row in report[key][population]["fixed_coverage"]:
                lines.append(f"| {population} | {method} | {row['requested_coverage']:.0%} | {row['count']} | {row['accuracy']:.4f} | {row['nll_nats']:.4f} | {row['top_label_ece_10_equal_width_bins']:.4f} |")
    lines += ["", "Cross-fitting protects the temperature fit from its held dialogue labels; it does not remove earlier head-selection bias. Differences are descriptive, ECE is sensitive to binning and small supports, and high-confidence slices may favor common classes. There was no test-cache access, no GPU/encoder/generator inference, no final full-dev temperature fit, and no changes to runtime parameters.", "",
              "Reproduce with `python -m checkin.confidence_study --home <artifacts> --output <new-study-directory> plan`, then the same command ending in `run`. Exact train/dev cache bytes and active checkpoint hashes must match the frozen protocol. Existing protocols and runs are never overwritten."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("phase", choices=("plan", "run"))
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        result = {"plan": plan_study, "run": run_study}[args.phase](args.home, args.output)
    print(json.dumps({"phase": args.phase, "output": str(args.output.resolve()), "completed": True,
                      "timestamp": result.get("completed_at", result.get("frozen_at"))}, indent=2))


if __name__ == "__main__":
    main()
