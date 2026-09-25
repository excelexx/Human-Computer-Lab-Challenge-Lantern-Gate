"""Study integrity and numeric contracts, using small CPU-only fixtures."""

import json

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from checkin import model_study as study
from checkin.models import EmotionHead, LinearEmotionHead, load_head
from checkin.settings import LABELS


def test_dialogue_groups_keep_utterances_together():
    ids = np.array([f"train:{dialogue}:{utterance}" for dialogue in range(21) for utterance in range(3)])
    groups = study.groups_from_ids(ids)
    labels = np.tile(np.arange(7), 9)
    for train, validation in StratifiedGroupKFold(3, shuffle=True, random_state=42).split(ids, labels, groups):
        assert not set(groups[train]) & set(groups[validation])
    with pytest.raises(ValueError, match="Malformed"):
        study.groups_from_ids(["train:wrong"])


def test_frequency_weights_use_only_provided_training_labels():
    labels = np.array([0, 0, 0, 0, 1])
    weight = study.weights_for(labels, "sqrt_balanced")
    assert weight.mean() == pytest.approx(1)
    assert weight[-1] / weight[0] == pytest.approx(2)
    assert study.weights_for(labels, "unweighted") is None


def test_standardizer_folding_preserves_probabilities_with_constant_column():
    rng = np.random.default_rng(2)
    labels = np.tile(np.arange(7), 10)
    x = rng.normal(size=(70, 12)).astype(np.float32) + labels[:, None]
    x[:, -1] = 3.0
    with threadpool_limits(limits=2):
        model, report = study.fit_linear(x, labels, {"C": 0.01, "weighting": "sqrt_balanced"})
    probability = study.predict_linear(model, x)
    assert probability.shape == (70, 7)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    assert report["learned_inference_parameters"] == 7 * 12 + 7
    assert set(model) == {"coefficient", "intercept"}


def test_late_fusion_never_uses_unavailable_vision():
    text = np.array([[0.2, 0.8], [0.9, 0.1]])
    vision = np.array([[0.8, 0.2], [np.nan, np.nan]])
    result = study.mix_probabilities(text, vision, np.array([True, False]), 0.2)
    np.testing.assert_allclose(result[0], [0.32, 0.68])
    np.testing.assert_array_equal(result[1], text[1])
    np.testing.assert_array_equal(study.mix_probabilities(text, vision, np.array([True, False]), 0), text)


def test_confusion_macro_matches_seven_class_sklearn_definition():
    truth = np.array([0, 0, 1, 2, 6])
    prediction = np.array([0, 1, 1, 0, 6])
    matrix = np.bincount(truth * 7 + prediction, minlength=49).reshape(7, 7)
    assert study.macro_from_confusion(matrix) == pytest.approx(f1_score(truth, prediction, labels=list(range(7)), average="macro", zero_division=0))


def test_paired_bootstrap_is_reproducible_and_identical_models_have_zero_delta():
    truth = np.tile(np.arange(7), 4)
    predicted = truth.copy()
    groups = np.array([f"test:{n // 4}" for n in range(len(truth))])
    report = study.paired_dialogue_interval(truth, predicted, predicted, groups, replicates=30)
    assert report["paired_delta_macro_f1"] == 0
    assert report["percentile_95_interval"] == [0, 0]
    assert report == study.paired_dialogue_interval(truth, predicted, predicted, groups, replicates=30)


def test_evaluation_rejects_modified_selection_before_opening_test(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text("{}", encoding="utf-8")
    (tmp_path / "selection.json").write_text(json.dumps({"plan_sha256": "wrong"}), encoding="utf-8")
    monkeypatch.setattr(study, "read_plan", lambda *args: {})
    monkeypatch.setattr(study, "load_cache", lambda *args: pytest.fail("Test was opened before selection verification"))
    with pytest.raises(ValueError, match="Frozen selection"):
        study.evaluate_once(tmp_path / "home", tmp_path)


def test_evaluation_is_single_use_before_opening_test(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text("{}", encoding="utf-8")
    (tmp_path / "selection.json").write_text(json.dumps({
        "plan_sha256": study.file_sha256(tmp_path / "plan.json"), "frozen_model_sha256": {},
    }), encoding="utf-8")
    (tmp_path / "test-evaluation-started.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(study, "read_plan", lambda *args: {})
    monkeypatch.setattr(study, "load_cache", lambda *args: pytest.fail("Test cache was opened a second time"))
    with pytest.raises(FileExistsError):
        study.evaluate_once(tmp_path / "home", tmp_path)


def test_linear_head_roundtrip_preserves_folded_probabilities_and_parameter_count(tmp_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(70, 12)).astype(np.float32)
    labels = np.tile(np.arange(7), 10)
    with threadpool_limits(limits=2):
        fitted, _ = study.fit_linear(features, labels, {"C": 0.01, "weighting": "sqrt_balanced"})
    head = LinearEmotionHead(12)
    state = {"net.weight": torch.from_numpy(fitted["coefficient"].astype(np.float32)),
             "net.bias": torch.from_numpy(fitted["intercept"].astype(np.float32))}
    path = tmp_path / "linear.pt"
    torch.save({"architecture": "linear", "input_dim": 12, "state_dict": state}, path)
    loaded, payload = load_head(path)
    assert payload["architecture"] == "linear"
    assert sum(parameter.numel() for parameter in loaded.parameters()) == 7 * 12 + 7
    with torch.inference_mode():
        probability = loaded(torch.from_numpy(features)).softmax(dim=1).numpy()
    np.testing.assert_allclose(probability, study.predict_linear(fitted, features), atol=1e-6, rtol=1e-5)


def test_legacy_mlp_defaults_and_explicit_hidden_width_are_honored(tmp_path):
    head = EmotionHead(5, hidden_dim=17).eval()
    path = tmp_path / "legacy.pt"
    torch.save({"input_dim": 5, "hidden_dim": 17, "state_dict": head.state_dict()}, path)
    restored, _ = load_head(path)
    sample = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    torch.testing.assert_close(restored(sample), head(sample))


@pytest.mark.parametrize("architecture", ["unknown", None, "transformer"])
def test_loader_rejects_unknown_architecture(tmp_path, architecture):
    path = tmp_path / "wrong.pt"
    torch.save({"architecture": architecture, "input_dim": 5, "state_dict": {}}, path)
    with pytest.raises(ValueError, match="Unknown classifier architecture"):
        load_head(path)


def test_linear_loader_rejects_mismatched_weight_shape(tmp_path):
    path = tmp_path / "wrong-shape.pt"
    head = LinearEmotionHead(5)
    torch.save({"architecture": "linear", "input_dim": 6, "state_dict": head.state_dict()}, path)
    with pytest.raises(ValueError, match="architecture/input_dim"):
        load_head(path)


def test_linear_loader_rejects_nonfinite_weights(tmp_path):
    path = tmp_path / "nonfinite.pt"
    head = LinearEmotionHead(5)
    with torch.no_grad():
        head.net.weight[0, 0] = float("nan")
    torch.save({"architecture": "linear", "input_dim": 5, "state_dict": head.state_dict()}, path)
    with pytest.raises(ValueError, match="finite floating-point"):
        load_head(path)


def _frozen_preparation_fixture(tmp_path):
    """Small genuine caches/head files; no test split exists in this fixture."""
    home, output, baseline = tmp_path / "runtime", tmp_path / "study", tmp_path / "original-heads"
    (home / "cache").mkdir(parents=True)
    (home / "checkpoints").mkdir()
    (output / "frozen-models").mkdir(parents=True)
    baseline.mkdir()
    rng = np.random.default_rng(52)
    for split in ("train", "dev"):
        labels = np.tile(np.arange(7), 2)
        text = rng.normal(size=(14, 1024)).astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        vision = rng.normal(size=(14, 1408)).astype(np.float32)
        vision /= np.linalg.norm(vision, axis=1, keepdims=True)
        quality = np.ones((14, 3), dtype=np.float32)
        quality[::2] = 0
        vision[::2] = 0
        np.savez_compressed(home / "cache" / f"{split}.npz", ids=np.array([f"{split}:{i // 7}:{i % 7}" for i in range(14)]),
                            labels=labels, text=text, vision=vision, quality=quality, feature_identity="fixture-features")
    hashes = {split: study.file_sha256(home / "cache" / f"{split}.npz") for split in ("train", "dev")}
    for stage, dimension in study.DIMENSIONS.items():
        head = EmotionHead(dimension)
        path = baseline / f"{stage}.pt"
        torch.save({"stage": stage, "input_dim": dimension, "labels": LABELS, "state_dict": head.state_dict(),
                    "feature_identity": "fixture-features", "cache_sha256": hashes}, path)
        (home / "checkpoints" / path.name).write_bytes(path.read_bytes())
    plan = {"schema_version": 1, "home": str(home), "study_source_sha256": "0" * 64,
            "feature_identity": "fixture-features", "cache_sha256": hashes,
            "baseline_head_sha256": {stage: study.file_sha256(baseline / f"{stage}.pt") for stage in study.DIMENSIONS},
            "linear": {"C": [.01], "weighting": ["sqrt_balanced"], "solver": "lbfgs", "max_iter": 500, "tolerance": 1e-4},
            "cv": {"seed": 42}}
    study.atomic_json(output / "plan.json", plan)
    study.save_arrays(output / "frozen-models/text.npz", coefficient=rng.normal(0, .01, size=(7, 1024)), intercept=np.zeros(7))
    selection = {"schema_version": 1, "plan_sha256": study.file_sha256(output / "plan.json"),
                 "feature_identity": plan["feature_identity"], "cache_sha256": hashes,
                 "frozen_model_sha256": {"text": study.file_sha256(output / "frozen-models/text.npz")},
                 "linear_winners": {"text": {"config": {"stage": "text", "C": .01, "weighting": "sqrt_balanced"}}}}
    study.atomic_json(output / "selection.json", selection)
    return home, output, baseline


def test_preparation_uses_preserved_baseline_after_production_changes_without_test_access(tmp_path, monkeypatch):
    home, output, baseline = _frozen_preparation_fixture(tmp_path)
    (home / "checkpoints/text.pt").write_bytes(b"Changed production text checkpoint")
    before = {path: path.read_bytes() for path in [output / "plan.json", output / "selection.json", home / "checkpoints/text.pt"]}
    original_loader = study.load_cache
    opened = []

    def load_train_dev(home, split):
        assert split in {"train", "dev"}, "Preparation must never open official test data"
        opened.append(split)
        return original_loader(home, split)

    monkeypatch.setattr(study, "load_cache", load_train_dev)
    with threadpool_limits(limits=2):
        result = study.prepare_hybrid(home, output, baseline, tmp_path / "candidate")
    assert opened == ["train", "dev"]
    assert result["production_activated"] is False
    assert result["test_accessed"] is False
    assert result["parameters"]["runtime_heads"]["text"] == 7175
    assert result["parameters"]["complete_total"] == 4464745375
    assert result["original_study_source_sha256"] == "0" * 64
    assert result["preparation_source_sha256"] == study.file_sha256(study.__file__)
    model, payload = load_head(tmp_path / "candidate/text.pt")
    assert payload["architecture"] == "linear"
    assert payload["labels"] == LABELS
    assert payload["train_count"] == payload["dev_count"] == 14
    assert payload["cache_sha256"] == json.loads((output / "plan.json").read_text())["cache_sha256"]
    assert all(path.read_bytes() == data for path, data in before.items())
    arrays = study.load_arrays(tmp_path / "candidate/hybrid-dev-predictions.npz")
    np.testing.assert_array_equal(arrays["hybrid"][arrays["available"]], arrays["baseline_operational"][arrays["available"]])


def test_preparation_rejects_changed_baseline_without_explicit_original_path(tmp_path):
    home, output, _ = _frozen_preparation_fixture(tmp_path)
    (home / "checkpoints/text.pt").write_bytes(b"Different checkpoint")
    with pytest.raises(ValueError, match="--baseline-heads"):
        study.prepare_hybrid(home, output)
    assert not (output / "deployment").exists()


def test_preparation_rejects_modified_frozen_text_before_loading_caches(tmp_path, monkeypatch):
    home, output, baseline = _frozen_preparation_fixture(tmp_path)
    (output / "frozen-models/text.npz").write_bytes(b"Changed after freeze")
    monkeypatch.setattr(study, "load_cache", lambda *args: pytest.fail("Cache opened before frozen artifact verification"))
    with pytest.raises(ValueError, match="Frozen text model bytes changed"):
        study.prepare_hybrid(home, output, baseline)


def test_preparation_refuses_existing_candidate_or_runtime_destination(tmp_path):
    home, output, baseline = _frozen_preparation_fixture(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "text.pt").write_bytes(b"Existing candidate")
    with pytest.raises(FileExistsError, match="nothing was overwritten"):
        study.prepare_hybrid(home, output, baseline, candidate)
    assert (candidate / "text.pt").read_bytes() == b"Existing candidate"
    with pytest.raises(ValueError, match="outside the production artifact tree"):
        study.prepare_hybrid(home, output, baseline, home / "checkpoints")
    with pytest.raises(ValueError, match="separate from preserved baseline heads"):
        study.prepare_hybrid(home, output, baseline, baseline)


def test_prepare_cli_forwards_baseline_and_candidate_paths(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(study, "prepare_hybrid", lambda *args: calls.append(args) or {"frozen_at": "fixture"})
    home, output = tmp_path / "runtime", tmp_path / "study"
    baseline, candidate = tmp_path / "heads", tmp_path / "candidate"
    study.main(["--home", str(home), "--output", str(output), "--baseline-heads", str(baseline),
                "--candidate-output", str(candidate), "prepare"])
    assert calls == [(home.resolve(), output.resolve(), baseline, candidate)]
    assert '"phase": "prepare"' in capsys.readouterr().out


def test_plan_baseline_override_is_persisted_and_source_guard_remains_strict(tmp_path):
    home, output, baseline = _frozen_preparation_fixture(tmp_path)
    fresh = tmp_path / "new-study"
    (home / "checkpoints/text.pt").write_bytes(b"Changed current production text")
    plan = study.make_plan(home, fresh, baseline)
    assert plan["baseline_heads"] == str(baseline.resolve())
    assert study.read_plan(home, fresh)["baseline_head_sha256"] == plan["baseline_head_sha256"]
    with pytest.raises(ValueError, match="source changed after preregistration"):
        study.read_plan(home, output, baseline)


@pytest.mark.parametrize("linear_text, expected_text, expected_total", [
    (False, 132103, 4464870303), (True, 7175, 4464745375),
])
def test_baseline_inventory_counts_actual_verified_mlp_or_linear_head(tmp_path, linear_text, expected_text, expected_total):
    home, output, baseline = _frozen_preparation_fixture(tmp_path)
    plan = json.loads((output / "plan.json").read_text())
    if linear_text:
        path = baseline / "text.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload.update(architecture="linear", state_dict=LinearEmotionHead(1024).state_dict())
        torch.save(payload, path)
        plan["baseline_head_sha256"]["text"] = study.file_sha256(path)
    # Current production need not be the frozen comparison baseline.
    (home / "checkpoints/text.pt").write_bytes(b"Different production state")
    report = study.baseline_parameter_inventory(home, plan, baseline)
    assert report["baseline_heads"] == {"vision": 181255, "text": expected_text, "fusion": 312711}
    assert report["baseline_complete_total"] == expected_total
    assert "may differ from current production" in report["count_scope"]
    assert report["baseline_head_sha256"] == plan["baseline_head_sha256"]


def test_baseline_inventory_refuses_a_checkpoint_changed_after_freeze(tmp_path):
    home, output, baseline = _frozen_preparation_fixture(tmp_path)
    plan = json.loads((output / "plan.json").read_text())
    (baseline / "text.pt").write_bytes(b"Not the frozen checkpoint")
    with pytest.raises(ValueError, match="frozen baseline text checkpoint"):
        study.baseline_parameter_inventory(home, plan, baseline)
