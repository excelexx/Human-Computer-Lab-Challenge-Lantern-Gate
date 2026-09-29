"""CPU-only coverage of benchmark failures and resume dependencies."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("dialogue_matrix_runner_under_test", scripts / "evaluate-dialogue-matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.generator, "_fit_context", lambda client, base, messages, deadline: messages)
    return module


class FakeClient:
    def __init__(self, finish="stop", content="We can wait here."):
        self.finish, self.content = finish, content

    def post(self, *args, **kwargs):
        return self

    def raise_for_status(self):
        return self

    def json(self):
        return {"choices": [{"message": {"content": self.content}, "finish_reason": self.finish}]}


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_non_stop_completion_preserves_partial_text_and_fails(runner, finish):
    item = runner.cases.matrix_cases()[0]
    result = runner.generate(FakeClient(finish, "A partial reply"), item)
    assert result["response"] == "A partial reply"
    assert result["error"] and "did not complete normally" in result["error"]
    assert not runner.accepted_response(result)


@pytest.mark.parametrize("content", ["", "   ", None, ["invalid"]])
def test_empty_or_malformed_content_never_advances(runner, content):
    result = runner.generate(FakeClient(content=content), runner.cases.matrix_cases()[0])
    assert result["error"]
    assert not runner.accepted_response(result)


def test_complete_text_is_accepted(runner):
    result = runner.generate(FakeClient(), runner.cases.matrix_cases()[0])
    assert result["error"] is None
    assert runner.accepted_response(result)


@pytest.mark.parametrize("field", ["history", "game_before", "text", "game_after"])
def test_resume_rejects_changed_journey_dependencies(runner, field):
    item = runner.cases.matrix_cases()[0]
    result = runner.generate(FakeClient(), item)
    runner.validate_journey_record(result, item)
    changed = {**result, field: "wrong dependency"}
    with pytest.raises(RuntimeError, match="dependency mismatch"):
        runner.validate_journey_record(changed, item)


def test_interrupted_jsonl_preserves_tail_and_resumes_complete_records(runner, tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_bytes(b'{"id":"done"}\n{"id":"unfinished')
    assert list(runner.read_jsonl(path)) == ["done"]
    assert path.read_bytes() == b'{"id":"done"}\n'
    archive = list(tmp_path.glob("*.interrupted-*"))
    assert len(archive) == 1 and archive[0].read_bytes() == b'{"id":"unfinished'
    assert list(runner.read_jsonl(path)) == ["done"]


def test_resume_rejects_duplicate_ids(runner, tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text('{"id":"same"}\n{"id":"same"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Duplicate case"):
        runner.read_jsonl(path)


def test_resume_does_not_discard_corrupt_middle_record(runner, tmp_path):
    path = tmp_path / "responses.jsonl"
    original = b'{"id":"first"}\nbroken\n{"id":"last"}\n'
    path.write_bytes(original)
    with pytest.raises(json.JSONDecodeError):
        runner.read_jsonl(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("field", ["text", "stage", "history", "evidence", "game_before", "game_after", "state", "context"])
def test_normal_matrix_resume_rejects_mutated_fixture_with_same_id(runner, field):
    item = runner.cases.matrix_cases()[0]
    result = runner.generate(FakeClient(), item)
    runner.validate_cached_matrix({result["id"]: result}, [item])
    mutated = {**result, field: "changed without changing ID"}
    with pytest.raises(RuntimeError, match="mismatch"):
        runner.validate_cached_matrix({mutated["id"]: mutated}, [item])


def test_normal_matrix_resume_rejects_unknown_record_ids(runner):
    with pytest.raises(RuntimeError, match="Unknown saved case IDs"):
        runner.validate_cached_matrix({"unknown": {"id": "unknown"}}, runner.cases.matrix_cases())


def test_partial_checks_and_resume_report_truthful_completeness(runner, monkeypatch, tmp_path):
    planned = [{"id": "one", "kind": "preset_path", "violations": []},
               {"id": "two", "kind": "authored_policy", "violations": []}]
    monkeypatch.setattr(runner, "deterministic_cases", lambda: iter(planned))
    runner.run_checks(tmp_path, 1)
    summary = json.loads((tmp_path / "check-summary.json").read_text())
    assert not summary["coverage_complete"]
    assert summary["expected_cases"] == 2 and summary["unique_cases"] == 1
    assert summary["missing_ids"] == ["two"]
    runner.run_checks(tmp_path, None)
    summary = json.loads((tmp_path / "check-summary.json").read_text())
    assert summary["coverage_complete"] and summary["unique_cases"] == 2
    assert summary["new_cases"] == 1 and summary["missing_ids"] == []


def test_checks_reject_changed_saved_result_before_appending(runner, monkeypatch, tmp_path):
    planned = [{"id": "one", "kind": "preset_path", "violations": []}]
    monkeypatch.setattr(runner, "deterministic_cases", lambda: iter(planned))
    path = tmp_path / "checks.jsonl"
    original = json.dumps({**planned[0], "violations": ["corrupted"]}) + "\n"
    path.write_text(original)
    with pytest.raises(RuntimeError, match="Saved deterministic check mismatch"):
        runner.run_checks(tmp_path, None)
    assert path.read_text() == original


def test_requested_section_completion_is_separate_from_full_coverage(runner):
    report = runner.coverage({"preset": {}}, {"preset": "presets", "custom": "customs"}, {"presets"})
    assert report["requested_sections_complete"]
    assert not report["coverage_complete"]
    assert report["requested_expected_cases"] == 1 and report["expected_cases"] == 2
