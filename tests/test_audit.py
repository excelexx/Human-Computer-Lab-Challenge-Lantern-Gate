"""Parameter inventory contracts using tiny local classifiers, never GPU models."""

import json

import pytest
import torch

from checkin.audit import audit_parameters, runtime_inventory
from checkin.downloads import model_manifest
from checkin.models import DIMENSIONS, EmotionHead, LinearEmotionHead
from checkin.settings import LABELS


BASELINE_TOTAL = 4_464_870_303
MIXED_TOTAL = 4_464_745_375


def write_heads(home, text_architecture="linear", text_hidden=128):
    directory = home / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    modules = {}
    for stage, dimension in DIMENSIONS.items():
        architecture = text_architecture if stage == "text" else "mlp"
        hidden = text_hidden if stage == "text" else 128
        model = LinearEmotionHead(dimension) if architecture == "linear" else EmotionHead(dimension, hidden)
        # The baseline files omitted architecture/hidden_dim; retain that case.
        payload = {"stage": stage, "input_dim": dimension, "labels": LABELS, "state_dict": model.state_dict()}
        if architecture == "linear":
            payload["architecture"] = "linear"
        elif hidden != 128:
            payload.update(architecture="mlp", hidden_dim=hidden)
        torch.save(payload, directory / f"{stage}.pt")
        modules[stage] = model
    return modules


def test_fresh_default_manifest_describes_shipped_mixed_stack():
    manifest = model_manifest()
    text = next(row for row in manifest["components"] if row["name"] == "MELD text head")
    assert text["architecture_kind"] == "linear"
    assert text["architecture"] == "1024 -> 7"
    assert text["parameters"] == 7175
    assert manifest["total_parameter_upper_bound"] == MIXED_TOTAL
    assert sum(row["parameters"] for row in manifest["components"]) == MIXED_TOTAL


@pytest.mark.parametrize("text_architecture,total,description", [("linear", MIXED_TOTAL, "1024 -> 7"), ("mlp", BASELINE_TOTAL, "1024 -> 128 -> 7")])
def test_audit_report_and_runtime_manifest_agree_on_loaded_heads(tmp_path, monkeypatch, text_architecture, total, description):
    import checkin.audit as module
    modules = write_heads(tmp_path, text_architecture)
    # Pinned large artifacts are outside this focused CPU fixture.
    monkeypatch.setattr(module, "artifacts", lambda **kw: [])
    (tmp_path / "reports").mkdir()
    historical = tmp_path / "reports/parameters-baseline.json"
    historical.write_bytes(b"historical evidence remains unchanged")
    report = audit_parameters(tmp_path)
    manifest = json.loads((tmp_path / "manifests/models.json").read_text())
    assert report["status"] == "verified"
    assert report["total_parameter_upper_bound"] == manifest["total_parameter_upper_bound"] == total
    assert report["components"] == manifest["components"]
    assert report["learned_head_parameters"] == sum(p.numel() for model in modules.values() for p in model.parameters())
    text = next(row for row in report["head_checks"] if row["name"] == "text")
    assert text["architecture"] == description
    assert text["count_method"] == "loaded checkpoint"
    assert historical.read_bytes() == b"historical evidence remains unchanged"


def test_nondefault_mlp_width_is_counted_from_the_actual_module(tmp_path):
    heads = write_heads(tmp_path, "mlp", text_hidden=64)
    manifest, checks = runtime_inventory(tmp_path)
    text = next(row for row in checks if row["name"] == "text")
    actual = sum(p.numel() for p in heads["text"].parameters())
    assert text["architecture"] == "1024 -> 64 -> 7"
    assert text["parameters"] == actual == 66_055
    assert manifest["total_parameter_upper_bound"] == BASELINE_TOTAL - 132_103 + actual


def test_missing_heads_are_explicitly_planned_with_shipped_architectures(tmp_path):
    manifest, checks = runtime_inventory(tmp_path)
    assert manifest["total_parameter_upper_bound"] == MIXED_TOTAL
    assert all(row["status"] == "planned_missing_checkpoint" for row in checks)
    assert all("plan" in row["count_method"] for row in checks)


def test_invalid_head_contract_fails_audit_instead_of_becoming_verified(tmp_path, monkeypatch):
    import checkin.audit as module
    write_heads(tmp_path)
    path = tmp_path / "checkpoints/text.pt"
    payload = torch.load(path, weights_only=True)
    payload["labels"] = list(reversed(LABELS))
    torch.save(payload, path)
    monkeypatch.setattr(module, "artifacts", lambda **kw: [])
    report = audit_parameters(tmp_path)
    assert report["status"] == "failed"
    text = next(row for row in report["head_checks"] if row["name"] == "text")
    assert text["status"] == "invalid"
    assert "not verified" in text["count_method"]


def test_pipeline_counts_loaded_modules_even_if_disk_checkpoint_changes(tmp_path, monkeypatch):
    from checkin.pipeline import CheckInPipeline
    baseline = write_heads(tmp_path, "mlp")
    pipe = CheckInPipeline(tmp_path, device="cpu")
    monkeypatch.setattr(pipe.generator, "status", lambda: {"available": True})
    before = pipe.status()
    assert before["parameter_budget"] == BASELINE_TOTAL
    assert before["parameter_accounting_status"] == "checkpoints_counted"
    pipe.heads, pipe._loaded = baseline, True
    write_heads(tmp_path, "linear")
    loaded = pipe.status()
    assert loaded["parameter_budget"] == BASELINE_TOTAL
    assert loaded["parameter_accounting_status"] == "loaded_runtime"
    assert all(row["count_method"] == "loaded inference module" for row in loaded["head_inventory"])
    pipe._loaded = False
    assert pipe.status()["parameter_budget"] == MIXED_TOTAL


def test_pipeline_status_reports_invalid_classifier_before_loading_encoders(tmp_path, monkeypatch):
    from checkin.pipeline import CheckInPipeline
    write_heads(tmp_path)
    (tmp_path / "checkpoints/text.pt").write_bytes(b"broken")
    pipe = CheckInPipeline(tmp_path, device="cpu")
    monkeypatch.setattr(pipe.generator, "status", lambda: {"available": True})
    status = pipe.status()
    assert not status["ready"] and not status["classification_ready"]
    assert status["parameter_accounting_status"] == "invalid"
    assert any("Invalid text classifier" in error for error in status["errors"])


def test_setup_rerun_preserves_actual_legacy_inventory_instead_of_new_default(tmp_path, monkeypatch):
    import checkin.downloads as module
    write_heads(tmp_path, "mlp")
    native = tmp_path / "vendor/llama"
    native.mkdir(parents=True)
    (native / "llama-server.exe").write_bytes(b"fixture runtime; never executed")
    monkeypatch.setattr(module, "_download", lambda artifact, home: {"name": artifact["name"], "status": "fixture"})
    monkeypatch.setattr(module, "_extract_runtime", lambda *args: None)
    module.download_all(tmp_path, include_meld=False)
    manifest = json.loads((tmp_path / "manifests/models.json").read_text())
    assert manifest["total_parameter_upper_bound"] == BASELINE_TOTAL
    assert next(row for row in manifest["components"] if row["name"] == "MELD text head")["architecture_kind"] == "mlp"
