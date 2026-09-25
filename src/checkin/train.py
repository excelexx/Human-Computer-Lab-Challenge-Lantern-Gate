"""Train small MELD heads from frozen, normalized local encoder features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import DIMENSIONS, EmotionHead
from .settings import LABELS, runtime_home


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cache(home, split):
    """Validate cache contracts before trusting cached labels and features."""
    if split not in {"train", "dev", "test"}:
        raise ValueError("Expected an official MELD split: train, dev or test.")
    path = Path(home) / "cache" / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache is missing; run extraction first: {path}")
    with np.load(path, allow_pickle=False) as source:
        missing = {"ids", "labels", "vision", "text", "quality"} - set(source.files)
        if missing:
            raise ValueError(f"{path.name} is missing fields: {sorted(missing)}")
        result = {name: source[name].copy() for name in ("ids", "labels", "vision", "text", "quality")}
        identity = str(source["feature_identity"].item()) if "feature_identity" in source.files else None
    ids, labels = result["ids"], result["labels"]
    if ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) == 0:
        raise ValueError("Cache IDs must be a nonempty one-dimensional Unicode array.")
    if len(set(ids.tolist())) != len(ids) or any(not str(value).startswith(split + ":") for value in ids):
        raise ValueError("Cache IDs must be unique and carry the correct split prefix.")
    if labels.shape != (len(ids),) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= len(LABELS))):
        raise ValueError("Cache labels must be integer MELD class indices in 0..6.")
    for name, dimensions in (("vision", 1408), ("text", 1024), ("quality", 3)):
        values = result[name]
        if values.shape != (len(ids), dimensions) or values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} cache: expected finite Nx{dimensions} floating values.")
        result[name] = values.astype(np.float32, copy=False)
    quality = result["quality"]
    if np.any((quality < 0) | (quality > 1)) or not np.isin(quality[:, 2], [0.0, 1.0]).all():
        raise ValueError("Visual quality must be in [0,1] with a binary availability flag.")
    if np.any(result["vision"][quality[:, 2] == 0] != 0):
        raise ValueError("Unavailable visual features must be zero, never a neutral-emotion substitute.")
    for name, mask in (("vision", quality[:, 2] > 0), ("text", np.ones(len(ids), dtype=bool))):
        norms = np.linalg.norm(result[name][mask], axis=1)
        if not np.allclose(norms, 1.0, atol=2e-3):
            raise ValueError(f"{name} cache features must be L2-normalized before training.")
    result["labels"] = labels.astype(np.int64, copy=False)
    metadata_path = path.with_suffix(".metadata.json")
    if identity is None and metadata_path.is_file():
        identity = json.loads(metadata_path.read_text(encoding="utf-8")).get("feature_identity")
    if identity is not None and (not isinstance(identity, str) or not identity):
        raise ValueError("feature_identity must be a nonempty string when provided.")
    result.update(sha256=file_sha256(path), feature_identity=identity, split=split)
    return result


def stage_arrays(cache, stage):
    if stage == "vision":
        mask = cache["quality"][:, 2] > 0
        return cache["vision"][mask], cache["labels"][mask]
    if stage == "text":
        return cache["text"], cache["labels"]
    if stage == "fusion":
        return np.concatenate((cache["vision"], cache["text"], cache["quality"]), axis=1), cache["labels"]
    raise ValueError(f"Unknown stage: {stage}")


def predict_probabilities(model, features, batch_size=512, device="cpu"):
    if len(features) == 0:
        return np.empty((0, len(LABELS)), dtype=np.float32)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Classifier produced nonfinite logits.")
            outputs.append(logits.softmax(dim=-1).float().cpu().numpy())
    return np.concatenate(outputs)


def _class_weights(labels, weighted):
    counts = np.bincount(labels, minlength=len(LABELS))
    if not weighted:
        return None, counts
    weights = np.zeros(len(LABELS), dtype=np.float32)
    present = counts > 0
    weights[present] = 1 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    return weights, counts


def train_stage(home=None, stage="vision", loss="both", epochs=30, patience=5,
                batch_size=128, learning_rate=1e-3, weight_decay=1e-4,
                seed=42, device="cpu"):
    """Select a checkpoint using dev macro F1; never opens the test cache."""
    if stage not in DIMENSIONS or loss not in {"both", "unweighted", "weighted"}:
        raise ValueError("Invalid training stage or loss option.")
    if min(epochs, patience, batch_size) < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Training hyperparameters must be positive (weight decay may be zero).")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    home = runtime_home(home)
    train, dev = load_cache(home, "train"), load_cache(home, "dev")
    if train["feature_identity"] != dev["feature_identity"]:
        raise ValueError("Train/dev features come from different encoders or preprocessing.")
    x_train, y_train = stage_arrays(train, stage)
    x_dev, y_dev = stage_arrays(dev, stage)
    if stage == "fusion":
        # Deployment uses the separately trained text head for missing vision.
        # Select fusion on the visual cases where its output is actually consumed.
        # Training still includes all rows and the existing visual dropout.
        eligible_dev = dev["quality"][:, 2] > 0
        x_dev, y_dev = x_dev[eligible_dev], y_dev[eligible_dev]
    selection_population = "eligible_visual_dev" if stage in {"vision", "fusion"} else "all_dev"
    if len(y_train) == 0 or len(y_dev) == 0:
        raise ValueError(f"{stage} training requires usable train and dev examples.")
    candidates = ["unweighted", "weighted"] if loss == "both" else [loss]
    best_overall = None
    candidate_reports = []
    started = time.perf_counter()
    for candidate in candidates:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = EmotionHead(DIMENSIONS[stage]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        weights, counts = _class_weights(y_train, candidate == "weighted")
        weight_tensor = torch.from_numpy(weights).to(device) if weights is not None else None
        loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                            batch_size=batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
        history, best_score, best_state, best_epoch, stale = [], -1.0, None, 0, 0
        candidate_started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total_mass = 0.0, 0.0
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                if stage == "fusion":
                    features = features.clone()
                    missing = torch.rand(len(features), device=device) < 0.15
                    features[missing, :1408] = 0
                    features[missing, -3:] = 0
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                objective = F.cross_entropy(logits, labels, weight=weight_tensor)
                if not torch.isfinite(objective):
                    raise RuntimeError("Training produced nonfinite loss; no checkpoint was published.")
                objective.backward()
                optimizer.step()
                mass = float(weight_tensor[labels].sum().item()) if weight_tensor is not None else len(labels)
                total_loss += float(objective.item()) * mass
                total_mass += mass
            probabilities = predict_probabilities(model, x_dev, device=device)
            predictions = probabilities.argmax(axis=1)
            score = float(f1_score(y_dev, predictions, labels=list(range(7)), average="macro", zero_division=0))
            weighted_f1 = float(f1_score(y_dev, predictions, labels=list(range(7)), average="weighted", zero_division=0))
            history.append({"epoch": epoch, "train_loss": total_loss / total_mass,
                            "dev_macro_f1": score, "dev_weighted_f1": weighted_f1})
            if score > best_score + 1e-12:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if epoch == 1 or epoch % 5 == 0 or stale >= patience:
                print(f"{stage}/{candidate} epoch={epoch} dev_macro_f1={score:.4f} best={best_score:.4f}", flush=True)
            if stale >= patience:
                break
        candidate_report = {
            "loss": candidate, "best_dev_macro_f1": best_score, "best_epoch": best_epoch,
            "epochs_run": len(history), "history": history,
            "training_seconds": time.perf_counter() - candidate_started,
            "class_weights": weights.tolist() if weights is not None else None,
        }
        candidate_reports.append(candidate_report)
        if best_overall is None or best_score > best_overall["best_dev_macro_f1"] + 1e-12:
            best_overall = {**candidate_report, "state_dict": best_state}
    finished_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1, "stage": stage, "created_at": finished_at,
        "input_dim": DIMENSIONS[stage], "seed": seed, "device": str(device),
        "train_count": int(len(y_train)), "dev_count": int(len(y_dev)),
        "dev_cache_count": int(len(dev["labels"])), "selection_population": selection_population,
        "train_class_counts": {label: int(counts[i]) for i, label in enumerate(LABELS)},
        "missing_train_classes": [LABELS[i] for i in range(7) if counts[i] == 0],
        "selected_loss": best_overall["loss"], "best_dev_macro_f1": best_overall["best_dev_macro_f1"],
        "best_epoch": best_overall["best_epoch"], "candidates": candidate_reports,
        "selection_metric": f"seven-class macro F1 on {selection_population}; zero_division=0",
        "test_used_for_selection": False,
        "cache_sha256": {"train": train["sha256"], "dev": dev["sha256"]},
        "feature_identity": train["feature_identity"],
        "feature_identity_verified": train["feature_identity"] is not None,
        "training_seconds": time.perf_counter() - started,
        "hyperparameters": {"batch_size": batch_size, "learning_rate": learning_rate,
                            "weight_decay": weight_decay, "maximum_epochs": epochs, "patience": patience,
                            "dropout": 0.2, "visual_dropout": 0.15 if stage == "fusion" else 0},
        "labels": list(LABELS), "encoders_frozen": True,
    }
    checkpoint = {
        "schema_version": 1, "stage": stage, "input_dim": DIMENSIONS[stage],
        "hidden_dim": 128, "state_dict": best_overall["state_dict"],
        "history": best_overall["history"], "best_dev_macro_f1": best_overall["best_dev_macro_f1"],
        "best_epoch": best_overall["best_epoch"], "train_count": int(len(y_train)),
        "dev_count": int(len(y_dev)), "selection_population": selection_population,
        "selected_loss": best_overall["loss"], "labels": list(LABELS), "seed": seed,
        "cache_sha256": report["cache_sha256"], "feature_identity": train["feature_identity"],
        "created_at": finished_at,
    }
    atomic_checkpoint(home / "checkpoints" / f"{stage}.pt", checkpoint)
    atomic_json(home / "reports" / f"training-{stage}.json", report)
    print(f"Saved {stage}: {report['selected_loss']}, dev macro F1={report['best_dev_macro_f1']:.4f}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home")
    parser.add_argument("--stage", choices=["vision", "text", "fusion", "all"], required=True)
    parser.add_argument("--loss", choices=["both", "unweighted", "weighted"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    options = vars(args)
    options.pop("threads")
    stage = options.pop("stage")
    for selected in ["vision", "text", "fusion"] if stage == "all" else [stage]:
        train_stage(stage=selected, **options)


if __name__ == "__main__":
    main()
