"""Numeric definitions, held-fold isolation, and dev-only artifact integrity."""
import hashlib
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
import torch

from checkin import confidence_study as study
from checkin.models import DIMENSIONS, EmotionHead, LinearEmotionHead
from checkin.settings import LABELS


def test_uniform_probabilities_have_known_seven_class_scores():
    result = study.metrics(np.arange(7), np.zeros((7, 7)))
    assert result["accuracy"] == pytest.approx(1 / 7)
    assert result["nll_nats"] == pytest.approx(np.log(7))
    assert result["multiclass_brier"] == pytest.approx(6 / 7)
    assert result["top_label_ece_10_equal_width_bins"] == pytest.approx(0)
    assert sum(row["count"] for row in result["reliability_bins"]) == 7


def test_nll_stays_finite_for_extreme_logits_and_confidence_one_is_in_last_bin():
    logits = np.zeros((1, 7))
    logits[0, 0] = 10000
    result = study.metrics(np.array([1]), logits)
    assert result["nll_nats"] == pytest.approx(10000)
    assert result["multiclass_brier"] == pytest.approx(2)
    assert result["top_label_ece_10_equal_width_bins"] == pytest.approx(1)
    assert result["reliability_bins"][-1]["count"] == 1
    assert result["support"]["surprise"] == 1


def test_coverage_uses_fixed_ceiling_counts_and_stable_id_ties():
    ids = np.array([f"dev:0:{i}" for i in [6, 1, 4, 0, 5, 2, 3]])
    rows = study.coverage_report(ids, np.arange(7), np.zeros((7, 7)))
    assert [row["count"] for row in rows] == [7, 6, 4]
    expected = hashlib.sha256("\n".join(sorted(ids.tolist())[:4]).encode()).hexdigest()
    assert rows[-1]["retained_ids_sha256"] == expected
    assert sum(rows[-1]["support"].values()) == 4


def test_temperature_is_positive_bounded_and_preserves_argmax():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(35, 7)) * 8
    labels = np.tile(np.arange(7), 5)
    temperature, report = study.fit_temperature(labels, logits)
    assert .25 <= temperature <= 4
    assert report["fit_nll_after"] <= report["fit_nll_before"]
    np.testing.assert_array_equal(logits.argmax(axis=1), (logits / temperature).argmax(axis=1))


def test_flat_temperature_objective_chooses_uncalibrated_reference():
    temperature, _ = study.fit_temperature(np.arange(7), np.zeros((7, 7)))
    assert temperature == 1


def test_crossfit_never_uses_a_held_dialogue_or_other_route_for_fitting(monkeypatch):
    ids = np.array([f"dev:{i // 7}:{i % 7}" for i in range(70)])
    labels = np.tile(np.arange(7), 10)
    routes = np.where(np.arange(70) % 2 == 0, "fusion", "text_fallback")
    logits = np.zeros((70, 7))
    logits[:, 0] = np.arange(70)
    fitted_rows = []

    def capture_fit(y, x):
        fitted_rows.append(x[:, 0].astype(int))
        return 1.0, {"temperature": 1.0, "fit_count": len(y), "fit_support": study.supports(y)}

    monkeypatch.setattr(study, "fit_temperature", capture_fit)
    result = study.crossfit_temperatures(ids, labels, logits, routes)
    groups = study.dialogue_groups(ids)
    for report, fit in zip(result["folds"], fitted_rows):
        held = np.flatnonzero((result["fold_assignment"] == report["fold"]) & (routes == report["route"]))
        assert set(fit) == set(np.flatnonzero((result["fold_assignment"] != report["fold"]) & (routes == report["route"])))
        assert not set(groups[fit]) & set(groups[held])
    assert result["changed_argmax_labels"] == 0
    assert len(result["folds"]) == 10


def fixture_artifacts(tmp_path):
    home, output = tmp_path / "runtime", tmp_path / "study"
    (home / "cache").mkdir(parents=True)
    (home / "checkpoints").mkdir()
    random = np.random.default_rng(42)
    for split in ("train", "dev"):
        text = random.normal(size=(70, 1024)).astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        vision = random.normal(size=(70, 1408)).astype(np.float32)
        vision /= np.linalg.norm(vision, axis=1, keepdims=True)
        quality = np.ones((70, 3), dtype=np.float32)
        quality[::2] = 0
        vision[::2] = 0
        np.savez_compressed(home / "cache" / f"{split}.npz", ids=np.array([f"{split}:{i // 7}:{i % 7}" for i in range(70)]),
                            labels=np.tile(np.arange(7), 10), text=text, vision=vision, quality=quality,
                            feature_identity="confidence-fixture")
    hashes = {split: study.file_sha256(home / "cache" / f"{split}.npz") for split in ("train", "dev")}
    for stage, dimension in DIMENSIONS.items():
        model = LinearEmotionHead(dimension) if stage == "text" else EmotionHead(dimension)
        torch.save({"stage": stage, "input_dim": dimension, "architecture": "linear" if stage == "text" else "mlp",
                    "labels": LABELS, "state_dict": model.state_dict(), "feature_identity": "confidence-fixture",
                    "cache_sha256": hashes}, home / "checkpoints" / f"{stage}.pt")
    return home, output


def test_full_protocol_runs_only_dev_metrics_and_never_changes_heads(tmp_path, monkeypatch):
    home, output = fixture_artifacts(tmp_path)
    before = {path: path.read_bytes() for path in (home / "checkpoints").glob("*.pt")}
    opened = []
    loader = study.load_cache

    def record_load(home, split):
        assert split in {"train", "dev"}
        opened.append(split)
        return loader(home, split)

    monkeypatch.setattr(study, "load_cache", record_load)
    with threadpool_limits(limits=2):
        protocol = study.plan_study(home, output)
        frozen = (output / "protocol.json").read_bytes()
        result = study.run_study(home, output)
    assert opened == ["train", "dev", "dev"]
    assert (output / "protocol.json").read_bytes() == frozen
    assert study.file_sha256(output / "source_snapshot.py") == protocol["study_source_sha256"]
    assert all(path.read_bytes() == original for path, original in before.items())
    assert result["test_accessed"] is result["production_changed"] is result["threshold_selected"] is False
    assert result["changed_argmax_labels"] == 0
    assert len(result["folds"]) == 10
    assert result["raw_operational"]["all"]["metrics"]["count"] == 70
    assert (output / "SUMMARY.md").is_file()
    with pytest.raises(FileExistsError):
        study.run_study(home, output)


def test_changed_head_is_rejected_before_evaluation(tmp_path):
    home, output = fixture_artifacts(tmp_path)
    study.plan_study(home, output)
    (home / "checkpoints/text.pt").write_bytes(b"Changed after freeze")
    with pytest.raises(ValueError, match="Active head changed"):
        study.run_study(home, output)
    assert not (output / "run-started.json").exists()


def test_protocol_cannot_be_overwritten_or_created_inside_production(tmp_path):
    home, output = fixture_artifacts(tmp_path)
    study.plan_study(home, output)
    with pytest.raises(FileExistsError, match="Protocol already exists"):
        study.plan_study(home, output)
    with pytest.raises(ValueError, match="outside the production"):
        study.plan_study(home, home / "confidence")


def test_changed_cache_is_rejected_before_evaluation(tmp_path):
    home, output = fixture_artifacts(tmp_path)
    study.plan_study(home, output)
    with (home / "cache/dev.npz").open("ab") as handle:
        handle.write(b"Changed after freeze")
    with pytest.raises(ValueError, match="feature bytes changed"):
        study.run_study(home, output)
    assert not (output / "run-started.json").exists()
