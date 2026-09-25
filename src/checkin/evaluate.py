"""Evaluate trained MELD classifiers with coverage and comparable subsets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .models import DIMENSIONS, load_head
from .settings import LABELS, runtime_home
from .train import atomic_json, file_sha256, load_cache, predict_probabilities, stage_arrays


def classification_metrics(truth, predicted):
    if len(truth) == 0:
        return {"count": 0, "status": "unavailable_no_eligible_examples"}
    classes = list(range(len(LABELS)))
    report = classification_report(truth, predicted, labels=classes, target_names=LABELS,
                                   zero_division=0, output_dict=True)
    return {
        "count": int(len(truth)), "status": "measured",
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=classes, average="weighted", zero_division=0)),
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
        "confusion_matrix_order": list(LABELS),
    }


def _save_confusion(path, metrics, title):
    if metrics["count"] == 0:
        Path(path).unlink(missing_ok=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(7.0, 6.0))
    shown = axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set(xticks=np.arange(7), yticks=np.arange(7), xticklabels=LABELS, yticklabels=LABELS,
             xlabel="Predicted label", ylabel="True MELD label", title=title)
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = float(matrix.max()) / 2
    for row in range(7):
        for column in range(7):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black", fontsize=9)
    fig.colorbar(shown, ax=axis, label="Utterances", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _atomic_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def exact_media_overlap(manifest_rows, test_ids):
    """Identify exact video overlaps without consulting emotion labels."""
    reference = defaultdict(list)
    for row in manifest_rows:
        if row.get("split") in {"train", "dev"} and row.get("media_sha256"):
            reference[row["media_sha256"]].append(row["id"])
    requested = set(str(identity) for identity in test_ids)
    test_rows = {row["id"]: row for row in manifest_rows
                 if row.get("split") == "test" and row.get("id") in requested}
    excluded = []
    known_media_count = 0
    for identity in sorted(requested):
        digest = test_rows.get(identity, {}).get("media_sha256")
        if digest:
            known_media_count += 1
        if digest in reference:
            excluded.append({"id": identity, "sha256": digest,
                             "overlaps_with": sorted(reference[digest])})
    return excluded, known_media_count


def evaluate(home=None, split="test", batch_size=512, device="cpu"):
    """Evaluate existing checkpoints; this function never trains or tunes models."""
    if split not in {"dev", "test"}:
        raise ValueError("Evaluation split must be dev or test.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    home = runtime_home(home)
    cache = load_cache(home, split)
    train = load_cache(home, "train")
    dev = cache if split == "dev" else load_cache(home, "dev")
    if len({item["feature_identity"] for item in (train, dev, cache)}) != 1:
        raise ValueError("Split caches use inconsistent encoder/preprocessing identities.")
    required_hashes = {"train": train["sha256"], "dev": dev["sha256"]}
    probabilities, checkpoint_metadata = {}, {}
    eligible = cache["quality"][:, 2] > 0
    for stage in ("vision", "text", "fusion"):
        path = home / "checkpoints" / f"{stage}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Train the {stage} stage before evaluation: {path}")
        model, payload = load_head(path, device=device)
        if payload.get("stage") != stage or payload.get("input_dim") != DIMENSIONS[stage] or payload.get("labels") != LABELS:
            raise ValueError(f"The {stage} checkpoint has an incompatible stage, dimension or label order.")
        if payload.get("cache_sha256") != required_hashes:
            raise ValueError(f"The {stage} checkpoint does not match the current train/dev feature caches.")
        if payload.get("feature_identity") != cache["feature_identity"]:
            raise ValueError(f"The {stage} checkpoint and evaluation features have different identities.")
        features, _ = stage_arrays(cache, stage)
        probabilities[stage] = predict_probabilities(model, features, batch_size=batch_size, device=device)
        checkpoint_metadata[stage] = {
            "sha256": file_sha256(path), "train_count": payload["train_count"],
            "best_dev_macro_f1": payload["best_dev_macro_f1"],
            "selected_loss": payload["selected_loss"],
        }
    truth = cache["labels"]
    text_predictions = probabilities["text"].argmax(axis=1)
    fusion_predictions = probabilities["fusion"].argmax(axis=1)
    vision_predictions = probabilities["vision"].argmax(axis=1)
    operational_probabilities = probabilities["fusion"].copy()
    operational_probabilities[~eligible] = probabilities["text"][~eligible]
    operational_predictions = operational_probabilities.argmax(axis=1)
    train_counts = np.bincount(train["labels"], minlength=len(LABELS))
    majority_label = int(train_counts.argmax())
    majority_predictions = np.full(len(truth), majority_label, dtype=np.int64)
    comparisons = {
        "common_visual_subset": {
            "count": int(eligible.sum()),
            "definition": "The identical examples with reliable visual input, selected before classification.",
            "models": {
                "majority": classification_metrics(truth[eligible], majority_predictions[eligible]),
                "vision": classification_metrics(truth[eligible], vision_predictions),
                "text": classification_metrics(truth[eligible], text_predictions[eligible]),
                "fusion": classification_metrics(truth[eligible], fusion_predictions[eligible]),
            },
        },
        "full_split": {
            "count": int(len(truth)),
            "definition": "Every cached utterance; operational fusion uses the text head when vision is unavailable.",
            "models": {
                "majority": classification_metrics(truth, majority_predictions),
                "text": classification_metrics(truth, text_predictions),
                "operational_fusion": classification_metrics(truth, operational_predictions),
            },
        },
    }
    coverage_by_class = {}
    for index, label in enumerate(LABELS):
        class_mask = truth == index
        count = int(class_mask.sum())
        accepted = int((class_mask & eligible).sum())
        coverage_by_class[label] = {"count": count, "visual_eligible": accepted,
                                    "coverage": accepted / count if count else None}
    source_manifest = home / "manifests" / "meld.jsonl"
    manifest_split_count = None
    manifest_id_match = None
    all_manifest_rows = []
    if source_manifest.is_file():
        with source_manifest.open(encoding="utf-8") as handle:
            all_manifest_rows = [json.loads(line) for line in handle if line.strip()]
        manifest_rows = [row for row in all_manifest_rows if row.get("split") == split]
        manifest_split_count = len(manifest_rows)
        by_id = {row["id"]: row for row in manifest_rows}
        manifest_id_match = len(by_id) == manifest_split_count and set(by_id) == set(cache["ids"].tolist())
        for index, identity in enumerate(cache["ids"]):
            row = by_id.get(str(identity))
            if row is not None and "label" in row and int(row["label"]) != int(truth[index]):
                raise ValueError(f"Cache label differs from the prepared MELD annotation: {identity}")
    warnings = []
    if manifest_split_count is None:
        warnings.append("Original manifest not available: completeness against the original MELD split is unverified.")
    elif not manifest_id_match:
        warnings.append("This cache is a subset of the prepared split; these results are not full-split MELD metrics.")
    official_count = {"dev": 1109, "test": 2610}[split]
    if len(truth) != official_count:
        warnings.append(f"Only {len(truth)} cached examples were evaluated; the official {split} split contains {official_count}.")
    if cache["feature_identity"] is None:
        warnings.append("Encoder/preprocessing identity is unavailable; only train/dev cache hashes were verified.")
    if not eligible.any():
        warnings.append("There are no eligible visual examples; multimodal evaluation is incomplete.")
    secondary = {
        "status": "not_applicable" if split != "test" else "unavailable_without_media_hashes",
        "definition": "Secondary test diagnostic excluding exact video SHA256 overlaps with train/dev. This does not detect all text or semantic overlap.",
        "used_for_training_or_selection": False,
    }
    excluded_ids = set()
    if split == "test" and all_manifest_rows:
        excluded, known_media_count = exact_media_overlap(all_manifest_rows, cache["ids"])
        reference_rows = [row for row in all_manifest_rows if row.get("split") in {"train", "dev"}]
        known_reference_count = sum(bool(row.get("media_sha256")) for row in reference_rows)
        secondary.update(test_rows_with_media_sha256=known_media_count,
                         test_rows_without_media_sha256=int(len(truth) - known_media_count),
                         reference_rows_with_media_sha256=known_reference_count,
                         reference_rows_without_media_sha256=len(reference_rows) - known_reference_count)
        if known_media_count and known_reference_count:
            excluded_ids = {item["id"] for item in excluded}
            retained = np.array([str(identity) not in excluded_ids for identity in cache["ids"]], dtype=bool)
            common_retained = retained & eligible
            secondary.update(
                status="measured", excluded_count=len(excluded), excluded=excluded,
                retained_count=int(retained.sum()),
                comparisons={
                    "common_visual_subset": {
                        "count": int(common_retained.sum()),
                        "models": {
                            "majority": classification_metrics(truth[common_retained], majority_predictions[common_retained]),
                            "vision": classification_metrics(truth[common_retained], vision_predictions[retained[eligible]]),
                            "text": classification_metrics(truth[common_retained], text_predictions[common_retained]),
                            "fusion": classification_metrics(truth[common_retained], fusion_predictions[common_retained]),
                        },
                    },
                    "full_split_after_exclusion": {
                        "count": int(retained.sum()),
                        "models": {
                            "majority": classification_metrics(truth[retained], majority_predictions[retained]),
                            "text": classification_metrics(truth[retained], text_predictions[retained]),
                            "operational_fusion": classification_metrics(truth[retained], operational_predictions[retained]),
                        },
                    },
                },
            )
            if excluded:
                warnings.append(f"Official test data contains {len(excluded)} exact video overlaps with train/dev. Primary metrics retain them; see the clearly separate secondary diagnostic.")
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split, "labels": list(LABELS), "device": str(device),
        "score_semantics": "uncalibrated_softmax", "zero_division": 0,
        "count": int(len(truth)), "prepared_manifest_split_count": manifest_split_count,
        "evaluated_entire_prepared_split": manifest_id_match,
        "official_split_count": official_count,
        "visual_eligible_count": int(eligible.sum()), "visual_coverage": float(eligible.mean()),
        "visual_unavailable_count": int((~eligible).sum()), "coverage_by_class": coverage_by_class,
        "majority_baseline": {"label": LABELS[majority_label], "source": "all training-cache labels", "class_counts": train_counts.tolist()},
        "comparisons": comparisons, "checkpoints": checkpoint_metadata,
        "cache_sha256": {"train": train["sha256"], "dev": dev["sha256"], split: cache["sha256"]},
        "feature_identity": cache["feature_identity"], "warnings": warnings,
        "secondary_duplicate_free_diagnostic": secondary,
        "limitations": ["Utterance emotion labels are not verified facial-expression labels.",
                        "Face eligibility is a heuristic, not verified active-speaker identification.",
                        "These classification scores do not measure generated-response quality or therapeutic benefit."],
    }
    rows = []
    vision_index = 0
    for index, identity in enumerate(cache["ids"]):
        visual_prediction = int(vision_predictions[vision_index]) if eligible[index] else None
        visual_scores = probabilities["vision"][vision_index].tolist() if eligible[index] else None
        if eligible[index]:
            vision_index += 1
        rows.append({
            "id": str(identity), "split": split, "true_label": LABELS[int(truth[index])],
            "true_label_id": int(truth[index]), "vision_available": bool(eligible[index]),
            "vision_label": LABELS[visual_prediction] if visual_prediction is not None else None,
            "vision_probabilities": visual_scores,
            "text_label": LABELS[int(text_predictions[index])], "text_probabilities": probabilities["text"][index].tolist(),
            "fusion_label": LABELS[int(fusion_predictions[index])] if eligible[index] else None,
            "operational_label": LABELS[int(operational_predictions[index])],
            "operational_probabilities": operational_probabilities[index].tolist(),
            "emotion_source": "fusion" if eligible[index] else "text_fallback",
            "modality_disagreement": visual_prediction != int(text_predictions[index]) if visual_prediction is not None else None,
            "quality": cache["quality"][index].tolist(), "score_semantics": "uncalibrated_softmax",
            "excluded_from_secondary_media_diagnostic": str(identity) in excluded_ids,
        })
    reports = home / "reports"
    _atomic_jsonl(reports / f"predictions-{split}.jsonl", rows)
    for comparison, values in comparisons.items():
        for name, metrics in values["models"].items():
            _save_confusion(reports / f"confusion-{split}-{comparison}-{name}.png", metrics,
                            f"{split}: {name}\n{comparison.replace('_', ' ')} (n={metrics['count']})")
    atomic_json(reports / f"evaluation-{split}.json", report)
    for comparison, values in comparisons.items():
        for name, metrics in values["models"].items():
            if metrics["count"]:
                print(f"{comparison}/{name}: n={metrics['count']} macro F1={metrics['macro_f1']:.4f}, weighted F1={metrics['weighted_f1']:.4f}", flush=True)
    if secondary["status"] == "measured":
        print(f"Secondary exact-media-overlap diagnostic: excluded {secondary['excluded_count']}; retained {secondary['retained_count']}. Primary official metrics are unchanged.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
