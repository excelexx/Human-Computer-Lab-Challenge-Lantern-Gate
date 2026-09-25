"""Descriptive uncertainty and calibration checks for saved MELD predictions.

No weights are learned, no threshold is selected, and no model is promoted.
Bootstrap resampling groups utterances by dialogue rather than treating nearby
conversation turns as independent. The official test set has been viewed before.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
from .settings import LABELS
from .train import atomic_json, file_sha256


def _labels(values, name="labels"):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu" or np.any((array < 0) | (array >= 7)):
        raise ValueError(f"{name} must be a one-dimensional array of integer class indices 0..6")
    return array.astype(np.int64, copy=False)


def scores_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(7), where=denominator > 0)
    total = matrix.sum()
    return {"macro_f1": float(f1.mean()),
            "weighted_f1": float(np.dot(f1, matrix.sum(1)) / total) if total else 0.0,
            "accuracy": float(np.trace(matrix) / total) if total else 0.0}


def calibration(labels, probabilities, bins=10):
    labels = _labels(labels)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not len(labels) or probabilities.shape != (len(labels), 7) or not np.isfinite(probabilities).all():
        raise ValueError("Calibration requires aligned nonempty finite seven-class probabilities")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid probability distribution")
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    which = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        mask = which == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(accuracy - mean)
        rows.append({"interval": [index / bins, (index + 1) / bins], "n": count,
                     "accuracy": accuracy, "mean_top_class_score": mean})
    target = np.eye(7)[labels]
    return {"n": len(labels), "top_label_ece_equal_width": float(ece),
            "multiclass_brier_sum": float(np.square(probabilities - target).sum(1).mean()),
            "negative_log_likelihood": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean()),
            "bins": rows, "note": "Descriptive reliability check only; no fitted temperature or decision threshold."}


def clustered_difference(truth, prediction_a, prediction_b, groups, draws=2000, seed=20260925):
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    truth, prediction_a, prediction_b = (_labels(value, name) for value, name in
        ((truth, "truth"), (prediction_a, "prediction_a"), (prediction_b, "prediction_b")))
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("Dialogue groups must be one-dimensional")
    if not len(truth) or not all(len(x) == len(truth) for x in (prediction_a, prediction_b, groups)):
        raise ValueError("Bootstrap arrays must be aligned and nonempty")
    unique, group_index = np.unique(groups, return_inverse=True)
    matrices = []
    for predictions in (prediction_a, prediction_b):
        tensor = np.zeros((len(unique), 7, 7), dtype=np.int64)
        np.add.at(tensor, (group_index, truth, predictions), 1)
        matrices.append(tensor)
    actual = [scores_from_confusion(tensor.sum(0)) for tensor in matrices]
    differences = {key: [] for key in actual[0]}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        sampled = rng.integers(0, len(unique), size=len(unique))
        measured = [scores_from_confusion(tensor[sampled].sum(0)) for tensor in matrices]
        for key in differences:
            differences[key].append(measured[0][key] - measured[1][key])
    return {"utterances": len(truth), "dialogues": len(unique), "draws": draws, "seed": seed,
            "difference_definition": "model A minus model B; whole dialogues sampled with replacement",
            "model_a": actual[0], "model_b": actual[1],
            "differences": {key: {"observed": actual[0][key] - actual[1][key],
                                   "percentile_95_interval": np.quantile(values, [.025, .975]).tolist()}
                            for key, values in differences.items()}}


def run(predictions, output, draws=2000, seed=20260925):
    path = Path(predictions)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Saved predictions must have unique nonempty example identities")
    if any(not re.fullmatch(r"(?:train|dev|test):\d+:\d+", str(row["id"])) for row in rows):
        raise ValueError("Example IDs must use official split:dialogue:utterance identity syntax")
    splits = {row["id"].split(":")[0] for row in rows}
    if len(splits) != 1:
        raise ValueError("Compare one official split at a time, never a mixed population")
    split = next(iter(splits))
    results = {}
    for name, subset in (("full_split", rows), ("common_visual", [r for r in rows if r["vision_available"]])):
        if not subset:
            results[name] = {"status": "unavailable", "n": 0, "reason": "No visually eligible examples"}
            continue
        truth = _labels([row["true_label_id"] for row in subset])
        operational = np.array([row["operational_probabilities"] for row in subset])
        text = np.array([row["text_probabilities"] for row in subset])
        groups = np.array([row["id"].rsplit(":", 1)[0] for row in subset])
        results[name] = {"paired_cluster_bootstrap": clustered_difference(truth, operational.argmax(1), text.argmax(1), groups, draws, seed),
                         "operational_calibration": calibration(truth, operational), "text_calibration": calibration(truth, text)}
        for disagreement in (False, True):
            mask = np.array([row["modality_disagreement"] is disagreement and row["vision_available"] for row in subset])
            if mask.any():
                result = clustered_difference(truth[mask], operational[mask].argmax(1), text[mask].argmax(1), groups[mask], draws, seed)
                results[name]["disagreement" if disagreement else "agreement"] = result
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "prediction_file_sha256": file_sha256(path),
              "script_sha256": file_sha256(__file__), "rows": len(rows), "labels": LABELS, "results": results,
              "split": split,
              "test_status": "Previously inspected official test results; this is descriptive backtesting of a reused benchmark."
                    if split == "test" else f"Descriptive analysis of saved {split} predictions; no test access.",
              "limitations": ["No model, prompt or threshold selection uses these statistics.",
                              "Dialogue grouping addresses within-dialogue dependence but not shared speakers, scenes or episodes across dialogues.",
                              "Percentile bootstrap intervals are descriptive uncertainty estimates and not a fresh replication or causal claim.",
                              "ECE depends on binning and class distribution. A softmax probability is not certainty about someone's feelings."]}
    atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps({name: value.get("paired_cluster_bootstrap", value) for name, value in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
