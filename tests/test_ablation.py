"""Constructed fixtures verify intervention alignment, not model quality."""
import numpy as np
import pytest

from checkin.ablation import build_interventions, _comparison


def _fixture():
    # Deliberately recognizable values make accidental pairing changes visible.
    return {
        "ids": np.array([f"dev:0:{i}" for i in range(7)]),
        "labels": np.arange(7),
        "vision": np.arange(14, dtype=np.float32).reshape(7, 2),
        "text": np.arange(21, dtype=np.float32).reshape(7, 3) + 100,
        "quality": np.array([[.6 + i * .02, .8 + i * .01, 0 if i == 2 else 1]
                             for i in range(7)], dtype=np.float32),
    }


def test_shuffling_keeps_vision_quality_paired_and_text_labels_fixed():
    cache = _fixture()
    original = {key: values.copy() for key, values in cache.items()}
    result = build_interventions(cache, seed=42)
    eligible = np.array([0, 1, 3, 4, 5, 6])
    order = result["permutation"]
    shuffled = result["features"]["shuffled_vision"]
    np.testing.assert_array_equal(result["ids"], cache["ids"][eligible])
    np.testing.assert_array_equal(result["labels"], cache["labels"][eligible])
    np.testing.assert_array_equal(shuffled[:, :2], cache["vision"][eligible][order])
    np.testing.assert_array_equal(shuffled[:, 2:5], cache["text"][eligible])
    np.testing.assert_array_equal(shuffled[:, -3:], cache["quality"][eligible][order])
    np.testing.assert_array_equal(result["shuffled_vision_source_ids"], cache["ids"][eligible][order])
    assert set(order.tolist()) == set(range(len(eligible)))
    assert np.any(order != np.arange(len(eligible)))
    np.testing.assert_array_equal(build_interventions(cache, seed=42)["permutation"], order)
    for key in cache:
        np.testing.assert_array_equal(cache[key], original[key])


def test_zero_conditions_change_only_the_intended_modalities():
    result = build_interventions(_fixture())
    full = result["features"]["full_fusion"]
    zero_visual = result["features"]["zero_vision"]
    zero_text = result["features"]["zero_text"]
    assert np.count_nonzero(zero_visual[:, :2]) == 0
    assert np.count_nonzero(zero_visual[:, -3:]) == 0
    np.testing.assert_array_equal(zero_visual[:, 2:5], full[:, 2:5])
    assert np.count_nonzero(zero_text[:, 2:5]) == 0
    np.testing.assert_array_equal(zero_text[:, :2], full[:, :2])
    np.testing.assert_array_equal(zero_text[:, -3:], full[:, -3:])


def test_diagnostics_reject_empty_visual_cohort():
    fixture = _fixture()
    fixture["quality"][:, 2] = 0
    with pytest.raises(ValueError, match="eligible visual dev"):
        build_interventions(fixture)


def test_distribution_shift_is_reported_when_label_does_not_change():
    reference = np.array([[.8, .2, 0, 0, 0, 0, 0]], dtype=np.float32)
    changed_scores = np.array([[.6, .4, 0, 0, 0, 0, 0]], dtype=np.float32)
    result = _comparison(np.array([0]), changed_scores, reference)
    assert result["changed_labels_vs_full_fusion"] == 0
    assert result["mean_distribution_l1_vs_full_fusion"] == pytest.approx(.4)
