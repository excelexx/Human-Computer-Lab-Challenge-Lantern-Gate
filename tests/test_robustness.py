"""Integrity tests for interventions; synthetic fixtures are not ML evidence."""
import numpy as np
import pytest
from checkin.robustness import ffmpeg_command, metric_summary, operational_predictions, sample_indices


@pytest.fixture
def saved_example(tmp_path):
    from checkin.robustness import empty_example, save_example
    row = {"id": "dev:4:2", "label": 0, "text": "An example", "media_sha256": "a" * 64}
    plan = {"protocol_identity": "b" * 64, "feature_identity": "c" * 64}
    text = np.zeros((1, 1024), np.float32)
    text[0, 0] = 1
    sample, arrays = empty_example(row, text, plan)
    sample["conditions"]["original"] = {"probabilities": np.eye(7)[0].tolist(), "label": "neutral",
        "quality": {"available": True, "reason": "ok", "valid_frame_fraction": 1., "mean_detection_score": .9},
        "source": "fusion"}
    arrays["vision"][0, 0] = 1
    arrays["quality"][0] = [1, .9, 1]
    arrays["valid"][0] = True
    result = tmp_path / "examples/dev-4-2.json"
    save_example(tmp_path, result, sample, arrays)
    return tmp_path, result, row, text, plan, sample, arrays


def test_sampling_is_repeatable_stratified_and_rejects_unavailable_faces():
    cache = {"labels": np.repeat(np.arange(7), 5), "quality": np.ones((35, 3))}
    cache["quality"][::5, 2] = 0
    selected = sample_indices(cache, per_class=3)
    assert np.array_equal(selected, sample_indices(cache, per_class=3))
    assert len(selected) == 21
    assert np.all(cache["quality"][selected, 2] == 1)
    assert np.all(np.bincount(cache["labels"][selected], minlength=7) == 3)


def test_stream_copy_command_never_contains_a_video_filter_or_reencoder():
    command = ffmpeg_command("input clip.mp4", "output clip.mp4", "silent_copy")
    assert command[command.index("-c:v") + 1] == "copy"
    assert "-vf" not in command and "-an" in command
    assert command[-1] == "output clip.mp4"


def test_operational_corruption_falls_back_to_text_instead_of_zero_fusion(monkeypatch):
    import checkin.robustness as module
    calls = []
    def predict(model, features):
        calls.append((model, features.copy()))
        return np.tile(np.eye(7, dtype=np.float32)[0 if model == "text" else 4], (len(features), 1))
    monkeypatch.setattr(module, "predict_probabilities", predict)
    probabilities = operational_predictions({"text": "text", "fusion": "fusion"}, np.zeros((2, 1408)),
                                            np.ones((2, 1024)), np.array([[0, 0, 0], [1, .9, 1]]))
    assert probabilities.argmax(1).tolist() == [0, 4]
    assert calls[1][1].shape == (1, 2435)


def test_metrics_use_identical_examples_and_report_distribution_changes():
    reference = np.eye(7)[[0, 1]]
    perturbed = reference * .8 + .2 / 7
    summary = metric_summary([0, 1], perturbed, reference)
    assert summary["label_flips"] == 0
    assert summary["mean_distribution_l1"] > 0
    with pytest.raises(ValueError):
        metric_summary([0, 1], perturbed, reference[:1])


def test_feature_snapshot_roundtrip_preserves_order_fixed_text_and_error_mask(saved_example):
    from checkin.robustness import VARIANTS, load_example
    output, result, row, text, plan, sample, arrays = saved_example
    resumed, stored = load_example(output, result, row, text, plan)
    assert resumed == sample
    assert stored["condition_names"].tolist() == list(VARIANTS)
    assert stored["vision"].shape == (8, 1408)
    assert stored["quality"].shape == (8, 3)
    assert np.array_equal(stored["text"], text)
    assert stored["valid"].tolist() == [True] + [False] * 7
    assert all("error" in resumed["conditions"][name] for name in list(VARIANTS)[1:])


@pytest.mark.parametrize("key,value", [("id", "dev:0:0"), ("label", "joy"),
    ("source_media_sha256", "e" * 64), ("protocol_identity", "f" * 64), ("text", "changed")])
def test_resume_rejects_stale_example_identity(saved_example, key, value):
    from checkin.robustness import load_example
    from checkin.train import atomic_json
    output, result, row, text, plan, sample, _ = saved_example
    sample[key] = value
    atomic_json(result, sample)
    with pytest.raises(ValueError, match="mismatched"):
        load_example(output, result, row, text, plan)


def test_resume_rejects_mutated_npz_bytes(saved_example):
    from checkin.robustness import load_example
    output, result, row, text, plan, sample, _ = saved_example
    (output / sample["features"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash changed"):
        load_example(output, result, row, text, plan)


@pytest.mark.parametrize("damage", ["vision_shape", "quality_range", "missing_variant", "bad_probability", "false_valid", "nan_feature", "changed_text"])
def test_resume_rejects_bad_shapes_predictions_and_availability(saved_example, damage):
    from checkin.robustness import load_example, save_example
    output, result, row, text, plan, sample, arrays = saved_example
    if damage == "vision_shape":
        arrays["vision"] = arrays["vision"][:, :-1]
    elif damage == "quality_range":
        arrays["quality"][0, 0] = 2
    elif damage == "missing_variant":
        del sample["conditions"]["silent_copy"]
    elif damage == "bad_probability":
        sample["conditions"]["original"]["probabilities"] = [0] * 7
    elif damage == "false_valid":
        arrays["valid"][1] = True
    elif damage == "nan_feature":
        arrays["vision"][0, 0] = np.nan
    else:
        arrays["text"][0, 0] = .5
    save_example(output, result, sample, arrays)
    with pytest.raises(ValueError):
        load_example(output, result, row, text, plan)


def test_interrupted_feature_publish_leaves_previous_json_snapshot_valid(saved_example, monkeypatch):
    import checkin.robustness as module
    output, result, row, text, plan, sample, arrays = saved_example
    previous = result.read_bytes()
    arrays["vision"][0, 0], arrays["vision"][0, 1] = 0, 1
    def fail(*args):
        raise OSError("simulated interruption before JSON commit")
    monkeypatch.setattr(module, "atomic_json", fail)
    with pytest.raises(OSError):
        module.save_example(output, result, sample, arrays)
    assert result.read_bytes() == previous
    _, intact = module.load_example(output, result, row, text, plan)
    assert intact["vision"][0, 0] == 1


def test_selected_manifest_contract_binds_text_and_actual_media(tmp_path):
    from checkin.robustness import identity, selected_manifest_rows
    from checkin.train import file_sha256
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture media")
    cache = {"ids": np.array(["dev:4:2"]), "labels": np.array([0])}
    row = {"id": "dev:4:2", "split": "dev", "label": 0, "text": "before", "video_path": str(media), "media_sha256": file_sha256(media)}
    original = selected_manifest_rows(cache, [0], [row])
    changed = selected_manifest_rows(cache, [0], [{**row, "text": "after"}])
    assert identity(original) != identity(changed)
    media.write_bytes(b"changed media")
    with pytest.raises(ValueError, match="Source media changed"):
        selected_manifest_rows(cache, [0], [row])
    changed = selected_manifest_rows(cache, [0], [{**row, "media_sha256": file_sha256(media)}])
    assert identity(original) != identity(changed)


def test_live_feature_contract_rejects_code_metadata_and_detector_changes(tmp_path, monkeypatch):
    import hashlib
    import inspect
    import checkin.robustness as module
    from checkin.train import atomic_json, file_sha256
    (tmp_path / "cache").mkdir()
    (tmp_path / "models/vision").mkdir(parents=True)
    detector = tmp_path / "models/vision/yunet.onnx"
    detector.write_bytes(b"fixture detector")
    monkeypatch.setattr(module, "YUNET_SHA256", file_sha256(detector))
    code = "\n".join(inspect.getsource(obj) for obj in (module.VideoProcessor, module.l2_normalize, module.VisionEncoder, module.TextEncoder))
    contract = {"implementation_sha256": hashlib.sha256(code.encode()).hexdigest()}
    feature_id = module.feature_identity(contract)
    atomic_json(tmp_path / "cache/dev.metadata.json", {"feature_metadata": contract, "feature_identity": feature_id})
    assert module.verified_feature_contract(tmp_path, {"feature_identity": feature_id})["yunet_sha256"] == file_sha256(detector)
    with pytest.raises(ValueError, match="metadata"):
        module.verified_feature_contract(tmp_path, {"feature_identity": "different"})
    monkeypatch.setattr(module.inspect, "getsource", lambda _: "changed source")
    with pytest.raises(ValueError, match="implementation"):
        module.verified_feature_contract(tmp_path, {"feature_identity": feature_id})
    monkeypatch.undo()
    monkeypatch.setattr(module, "YUNET_SHA256", "wrong")
    with pytest.raises(ValueError, match="detector"):
        module.verified_feature_contract(tmp_path, {"feature_identity": feature_id})


def test_resume_retries_only_failed_variants_and_keeps_success_features(tmp_path, monkeypatch):
    """Execute both invocations with explicit non-ML fixtures, never a GPU."""
    import types
    from pathlib import Path
    import checkin.robustness as module
    from checkin.train import file_sha256
    home = tmp_path / "home"
    (home / "checkpoints").mkdir(parents=True)
    for stage in ("vision", "text", "fusion"):
        (home / f"checkpoints/{stage}.pt").write_bytes(stage.encode())
    source = home / "dev.mp4"
    source.write_bytes(b"fixture media")
    row = {"id": "dev:4:2", "split": "dev", "label": 0, "text": "fixture", "video_path": str(source), "media_sha256": file_sha256(source)}
    vision = np.zeros((1, 1408), np.float32)
    textual = np.zeros((1, 1024), np.float32)
    vision[0, 0] = textual[0, 0] = 1
    cache = {"ids": np.array([row["id"]]), "labels": np.array([0]), "quality": np.array([[1., .9, 1.]], np.float32),
             "vision": vision, "text": textual, "feature_identity": "fixture-contract", "sha256": "fixture-cache"}
    monkeypatch.setattr(module, "load_cache", lambda *a: cache)
    monkeypatch.setattr(module, "read_manifest", lambda *a: [row])
    contract = {"feature_metadata": {"vision": {}, "processor": "fixture", "frames": 8, "min_valid": 6}}
    monkeypatch.setattr(module, "verified_feature_contract", lambda *a: contract)
    monkeypatch.setattr(module, "load_head", lambda p: (p.stem, {"stage": p.stem, "feature_identity": "fixture-contract", "labels": module.LABELS, "input_dim": module.DIMENSIONS[p.stem]}))
    monkeypatch.setattr(module, "predict_probabilities", lambda model, features: np.tile(np.eye(7, dtype=np.float32)[0], (len(features), 1)))
    class Processor:
        version, frames, min_valid = "fixture", 8, 6
        def __init__(self, *a): pass
        def process(self, path):
            return [np.zeros((2, 2, 3), np.uint8)], {"available": True, "reason": "ok", "valid_frame_fraction": 1., "mean_detection_score": .9}
    class Encoder:
        def __init__(self, *a, **kw): pass
        def metadata(self): return {}
        def encode_faces(self, crops): return vision.copy()
    monkeypatch.setattr(module, "VideoProcessor", Processor)
    monkeypatch.setattr(module, "VisionEncoder", Encoder)
    commands, fail_blur = [], [True]
    def command(args, **kwargs):
        if args == ["ffmpeg", "-version"]:
            return types.SimpleNamespace(stdout="fixture FFmpeg\n")
        commands.append(args[-1])
        if "gaussian_blur" in args[-1] and fail_blur[0]:
            raise RuntimeError("transient fixture failure")
        Path(args[-1]).write_bytes(b"converted fixture")
        return types.SimpleNamespace(stdout="")
    monkeypatch.setattr(module.subprocess, "run", command)
    output = tmp_path / "report"
    first = module.run(home, output, per_class=1, device="cpu", repeats=1)
    assert first["status"] == "complete_with_errors"
    assert len(commands) == 7
    first_path = first["examples"][0]["features"]["path"]
    commands.clear()
    fail_blur[0] = False
    second = module.run(home, output, per_class=1, device="cpu", repeats=1)
    assert second["status"] == "complete"
    assert len(commands) == 1 and "gaussian_blur" in commands[0]
    assert (output / first_path).is_file()
    assert second["examples"][0]["features"]["path"] != first_path
    with np.load(output / second["examples"][0]["features"]["path"], allow_pickle=False) as stored:
        assert stored["valid"].all()
        assert np.array_equal(stored["text"], textual)
    with pytest.raises(ValueError, match="different protocol"):
        module.run(home, output, per_class=1, device="cuda", repeats=1)
