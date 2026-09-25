"""Regression checks for deployment selection and checkpoint provenance.

Small synthetic tensors verify contracts only; these are not MELD scores.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from checkin.models import DIMENSIONS
from checkin.settings import LABELS


@pytest.mark.parametrize("field,value", [
    ("labels", list(reversed(LABELS))),
    ("stage", "fusion"),
    ("input_dim", DIMENSIONS["vision"] + 1),
])
def test_inference_rejects_head_metadata_mismatch_even_with_matching_features(tmp_path, monkeypatch, field, value):
    import checkin.pipeline as pipeline
    import checkin.features as features

    # No pretrained weights or hardware are needed to exercise acceptance rules.
    monkeypatch.setattr(pipeline, "VideoProcessor", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "VisionEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "TextEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(features, "feature_metadata", lambda *args: {})
    monkeypatch.setattr(features, "feature_identity", lambda *args: "matching-feature-identity")

    def checkpoint(path, device):
        stage = Path(path).stem
        payload = {"stage": stage, "input_dim": DIMENSIONS[stage], "labels": list(LABELS),
                   "feature_identity": "matching-feature-identity"}
        if stage == "vision":
            payload[field] = value
        return object(), payload

    monkeypatch.setattr(pipeline, "load_head", checkpoint)
    instance = pipeline.CheckInPipeline(tmp_path, device="cpu")
    with pytest.raises(ValueError, match="incompatible stage, dimension or label order"):
        instance._load()
    assert not instance._loaded


def _cache(path, split, labels, available):
    count = len(labels)
    text = np.zeros((count, 1024), dtype=np.float32)
    text[:, 0] = 1
    vision = np.zeros((count, 1408), dtype=np.float32)
    vision[np.asarray(available, dtype=bool), 0] = 1
    quality = np.zeros((count, 3), dtype=np.float32)
    quality[np.asarray(available, dtype=bool)] = [1, .95, 1]
    np.savez(path / f"{split}.npz", ids=np.array([f"{split}:0:{i}" for i in range(count)]),
             labels=np.array(labels, dtype=np.int64), text=text, vision=vision, quality=quality,
             feature_identity=np.array("synthetic-review-fixture"))


def test_fusion_selects_on_consumed_visual_dev_path_but_trains_all_rows(tmp_path, monkeypatch):
    import checkin.train as training

    cache = tmp_path / "cache"
    cache.mkdir()
    _cache(cache, "train", [0, 1, 2, 3, 4, 5, 6, 0], [1, 1, 0, 1, 1, 0, 1, 0])
    _cache(cache, "dev", [4, 6, 0, 0], [1, 1, 0, 0])
    original_predict = training.predict_probabilities
    consumed = []

    def inspect_selection(model, features, **kwargs):
        consumed.append(features.copy())
        return original_predict(model, features, **kwargs)

    monkeypatch.setattr(training, "predict_probabilities", inspect_selection)
    report = training.train_stage(tmp_path, stage="fusion", loss="unweighted", epochs=1,
                                  patience=1, batch_size=8, device="cpu")
    assert report["train_count"] == 8
    assert report["dev_count"] == 2
    assert report["dev_cache_count"] == 4
    assert report["selection_population"] == "eligible_visual_dev"
    assert len(consumed) == 1
    assert consumed[0].shape == (2, DIMENSIONS["fusion"])
    assert np.all(consumed[0][:, -1] == 1)
    payload = torch.load(tmp_path / "checkpoints/fusion.pt", weights_only=True)
    assert payload["dev_count"] == 2
    assert payload["selection_population"] == "eligible_visual_dev"


def test_fusion_refuses_selection_without_visual_dev_examples(tmp_path):
    from checkin.train import train_stage

    cache = tmp_path / "cache"
    cache.mkdir()
    _cache(cache, "train", [0, 1], [1, 1])
    _cache(cache, "dev", [0, 1], [0, 0])
    with pytest.raises(ValueError, match="usable train and dev examples"):
        train_stage(tmp_path, stage="fusion", loss="unweighted", epochs=1)
    assert not (tmp_path / "checkpoints/fusion.pt").exists()
