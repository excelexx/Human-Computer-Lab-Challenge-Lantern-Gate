"""Small fixtures verify fallback-only replay and preservation of failed rows."""
import numpy as np
import pytest
import torch

from checkin.models import LinearEmotionHead
from checkin.robustness import VARIANTS, operational_predictions
from checkin.robustness_compare import _verified_head, compare_predictions, summarize_conditions
from checkin.settings import LABELS
from checkin.train import file_sha256


def constant_head(dimensions, selected):
    head = LinearEmotionHead(dimensions).eval()
    with torch.no_grad():
        head.net.weight.zero_()
        head.net.bias.zero_()
        head.net.bias[selected] = 3
    return head


def paired_fixture():
    heads = {"text": constant_head(1024, 0), "fusion": constant_head(2435, 4)}
    visual, textual = np.zeros((2, 1408), np.float32), np.zeros((2, 1024), np.float32)
    visual[0, 0] = 1
    textual[:, 0] = 1
    quality = np.array([[1, .95, 1], [0, 0, 0]], np.float32)
    return heads, visual, textual, quality


def test_candidate_can_change_only_missing_vision_and_reproduces_recorded_baseline():
    heads, visual, textual, quality = paired_fixture()
    recorded = operational_predictions(heads, visual, textual, quality)
    baseline, candidate, error = compare_predictions(heads, constant_head(1024, 3), visual, textual, quality, recorded)
    assert error == 0
    assert np.array_equal(candidate[0], baseline[0])
    assert baseline.argmax(1).tolist() == [4, 0]
    assert candidate.argmax(1).tolist() == [4, 3]


def test_replay_rejects_probabilities_from_a_different_baseline():
    heads, visual, textual, quality = paired_fixture()
    stale = operational_predictions(heads, visual, textual, quality)[:, ::-1].copy()
    with pytest.raises(ValueError, match="does not reproduce"):
        compare_predictions(heads, constant_head(1024, 3), visual, textual, quality, stale)


def test_head_verification_rejects_hash_or_label_order_drift(tmp_path):
    path = tmp_path / "text.pt"
    head = constant_head(1024, 0)
    payload = {"architecture": "linear", "stage": "text", "input_dim": 1024,
               "labels": list(reversed(LABELS)), "state_dict": head.state_dict(),
               "feature_identity": "fixture", "cache_sha256": {"train": "a", "dev": "b"}, "train_count": 7}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="hash differs"):
        _verified_head(path, "text", "stale", "fixture", payload["cache_sha256"])
    with pytest.raises(ValueError, match="metadata does not match"):
        _verified_head(path, "text", file_sha256(path), "fixture", payload["cache_sha256"])


def test_failed_conditions_are_not_counted_as_fallback_and_transitions_are_paired():
    probabilities = np.eye(7)[0].tolist()
    available = {"vision_available": True, "quality_reason": "usable", "source": "fusion",
                 "baseline_probabilities": probabilities, "candidate_probabilities": probabilities}
    missing = {**available, "vision_available": False, "quality_reason": "no_face", "source": "text_fallback"}
    first = {"id": "dev:0:0", "true_label_id": 0, "conditions": {name: dict(available) for name in VARIANTS}}
    first["conditions"]["black_control"] = missing
    second = {"id": "dev:0:1", "true_label_id": 0, "conditions": {name: {"error": "decoder_failure"} for name in VARIANTS}}
    summary = summarize_conditions([first, second])["black_control"]
    assert summary["requested_n"] == 2 and summary["n"] == 1
    assert summary["fallback_count"] == 1
    assert summary["availability_transitions"]["eligible_to_fallback"]["n"] == 1
    assert summary["availability_transitions"]["fallback_to_fallback"]["n"] == 0
    assert summary["errors"] == [{"id": "dev:0:1", "error": "decoder_failure"}]
