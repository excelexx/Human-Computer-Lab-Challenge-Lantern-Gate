"""Dev-only modality interventions on existing MELD features and trained heads.

This command never trains, selects a checkpoint, or opens test features. The
zeroed/shuffled conditions are constructed diagnostics, not natural input pairs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import numpy as np
from sklearn.metrics import f1_score

from .data import read_manifest
from .models import DIMENSIONS, load_head
from .settings import LABELS, runtime_home
from .train import atomic_json, file_sha256, load_cache, predict_probabilities


def build_interventions(cache, seed=42):
    """Keep each text/label pair fixed; shuffle vision and its quality together."""
    eligible = np.flatnonzero(cache["quality"][:, 2] > 0)
    if len(eligible) == 0:
        raise ValueError("Modality diagnostics require eligible visual dev examples.")
    visual = cache["vision"][eligible].copy()
    textual = cache["text"][eligible].copy()
    quality = cache["quality"][eligible].copy()
    permutation = np.random.default_rng(seed).permutation(len(eligible))
    features = {
        "full_fusion": np.concatenate((visual, textual, quality), axis=1),
        "zero_vision": np.concatenate((np.zeros_like(visual), textual, np.zeros_like(quality)), axis=1),
        "shuffled_vision": np.concatenate((visual[permutation], textual, quality[permutation]), axis=1),
        "zero_text": np.concatenate((visual, np.zeros_like(textual), quality), axis=1),
    }
    return {
        "ids": cache["ids"][eligible].copy(),
        "labels": cache["labels"][eligible].copy(),
        "text": textual,
        "permutation": permutation,
        "shuffled_vision_source_ids": cache["ids"][eligible][permutation].copy(),
        "features": features,
    }


def _comparison(truth, probabilities, reference):
    predicted = probabilities.argmax(axis=1)
    reference_labels = reference.argmax(axis=1)
    changed = predicted != reference_labels
    distance = np.abs(probabilities - reference).sum(axis=1)
    return {
        "count": int(len(truth)),
        "macro_f1": float(f1_score(truth, predicted, labels=list(range(7)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=list(range(7)), average="weighted", zero_division=0)),
        "changed_labels_vs_full_fusion": int(changed.sum()),
        "changed_label_fraction_vs_full_fusion": float(changed.mean()),
        "mean_distribution_l1_vs_full_fusion": float(distance.mean()),
        "max_distribution_l1_vs_full_fusion": float(distance.max()),
    }


def run_ablation(home=None, seed=42, batch_size=512, device="cpu"):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    home = runtime_home(home)
    dev = load_cache(home, "dev")
    train = load_cache(home, "train")
    identity = dev["feature_identity"]
    if not identity or identity != train["feature_identity"]:
        raise ValueError("Train/dev feature provenance must be present and consistent.")

    # Refuse unrelated caches masquerading as official dev rows. The diagnostic
    # can use a recorded subset, but every row must resolve to MELD annotations.
    manifest = read_manifest(home, "dev")
    by_id = {row["id"]: row for row in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("The prepared dev manifest contains duplicate identities.")
    for row_id, label in zip(dev["ids"], dev["labels"]):
        row = by_id.get(str(row_id))
        if row is None or int(row["label"]) != int(label):
            raise ValueError(f"Dev feature/annotation mismatch: {row_id}")

    heads, checkpoints = {}, {}
    required_hashes = {"train": train["sha256"], "dev": dev["sha256"]}
    for stage in ("fusion", "text"):
        path = home / "checkpoints" / f"{stage}.pt"
        model, payload = load_head(path, device=device)
        if payload.get("stage") != stage or payload.get("input_dim") != DIMENSIONS[stage] or payload.get("labels") != LABELS:
            raise ValueError(f"Invalid {stage} checkpoint stage, dimensions or label order.")
        if payload.get("feature_identity") != identity or payload.get("cache_sha256") != required_hashes:
            raise ValueError(f"The {stage} checkpoint does not match these train/dev feature caches.")
        if not isinstance(payload.get("train_count"), int) or payload["train_count"] < 1:
            raise ValueError(f"The {stage} checkpoint has no recorded training examples.")
        heads[stage] = model
        checkpoints[stage] = {"sha256": file_sha256(path), "train_count": payload["train_count"],
                              "created_at": payload.get("created_at")}

    interventions = build_interventions(dev, seed)
    predictions = {name: predict_probabilities(heads["fusion"], values, batch_size=batch_size, device=device)
                   for name, values in interventions["features"].items()}
    predictions["text_head"] = predict_probabilities(heads["text"], interventions["text"], batch_size=batch_size, device=device)
    reference = predictions["full_fusion"]
    summaries = {name: _comparison(interventions["labels"], probabilities, reference)
                 for name, probabilities in predictions.items()}
    permuted = interventions["permutation"]
    moved_count = int(np.count_nonzero(permuted != np.arange(len(permuted))))
    warnings = []
    if len(dev["ids"]) != len(manifest):
        warnings.append("Cached dev examples are a subset of the prepared dev split.")
    if len(manifest) != 1109:
        warnings.append("The prepared dev manifest does not contain the official 1,109 examples.")
    if moved_count == 0:
        warnings.append("The seeded permutation moved no examples; shuffled-vision sensitivity cannot be assessed.")
    rows = []
    for index, row_id in enumerate(interventions["ids"]):
        rows.append({
            "id": str(row_id), "true_label": LABELS[int(interventions["labels"][index])],
            "shuffled_vision_source_id": str(interventions["shuffled_vision_source_ids"][index]),
            "conditions": {name: {"label": LABELS[int(values[index].argmax())],
                                   "probabilities": values[index].tolist()}
                           for name, values in predictions.items()},
        })
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured", "split": "dev", "seed": int(seed), "device": str(device),
        "purpose": "Post-training modality sensitivity diagnostics; no checkpoint selection or training performed.",
        "test_data_used": False, "labels": LABELS,
        "cohort": {"definition": "Identical dev utterances with eligible visual input in all five comparisons.",
                   "eligible_visual_count": len(rows), "dev_cache_count": len(dev["ids"]),
                   "prepared_dev_count": len(manifest), "coverage_in_dev_cache": len(rows) / len(dev["ids"])},
        "conditions": {
            "full_fusion": "Unmodified, correctly paired cached text, vision and quality.",
            "zero_vision": "Raw fusion head after zeroing vision AND all quality/mask values. Diagnostic only: deployed missing-vision turns use the text head.",
            "shuffled_vision": "Raw fusion head after a seeded permutation moves vision and its quality together; text, label and target ID stay fixed. Constructed mismatched pairs.",
            "zero_text": "Raw fusion head with a zero text vector and original vision/quality. Diagnostic outside the supported nonempty-text input contract.",
            "text_head": "Separately trained text-only head on the same eligible dev text; also the deployed missing-vision fallback.",
        },
        "comparisons": summaries,
        "permutation": {"eligible_positions": permuted.tolist(), "moved_examples": moved_count,
                        "fixed_examples": len(permuted) - moved_count, "quality_moves_with_vision": True},
        "cache_sha256": required_hashes, "feature_identity": identity, "checkpoints": checkpoints,
        "score_semantics": "uncalibrated_softmax",
        "distance_definition": "For each example sum absolute differences across the seven class scores, then average; range 0 to 2.",
        "warnings": warnings,
        "limitations": [
            "The dev split was already used for checkpoint selection; these are not held-out generalization results.",
            "Changed outputs establish sensitivity to the interventions, not an accuracy benefit or meaningful understanding.",
            "Zeroed and shuffled inputs can be out of distribution; their scores are not deployed fallback performance.",
            "A label can remain unchanged even when its distribution shifts; report both measures.",
            "These checks do not evaluate response generation, facial speaker attribution, or clinical effectiveness.",
        ],
        "examples": rows,
    }
    atomic_json(home / "reports/ablation-dev.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_ablation(**vars(args))
    print(json.dumps({"status": report["status"], "cohort": report["cohort"],
                      "comparisons": report["comparisons"], "warnings": report["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
