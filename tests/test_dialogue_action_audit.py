"""Ensure action audit coverage cannot be manufactured from incomplete fixtures."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audit_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("dialogue_action_audit_under_test", scripts / "audit-dialogue-actions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def successful_rows(module):
    return [{**item, "game_after": module.expected_after(item), "response": "We can wait here.",
             "finish_reason": "stop", "error": None}
            for item in module.cases.matrix_cases() if item["section"] == "customs"]


def test_empty_and_partial_audits_never_report_full_coverage(audit_module):
    empty = audit_module.audit([])
    assert empty["failed"] == 0 and empty["checked"] == 0
    assert empty["expected_cases"] == 924 and not empty["coverage_complete"]
    partial = audit_module.audit(successful_rows(audit_module)[:1])
    assert partial["unique_cases"] == 1 and len(partial["missing_ids"]) == 923
    assert not partial["coverage_complete"] and not partial["generation_complete"]


def test_complete_fixed_fixture_coverage_is_reported(audit_module):
    result = audit_module.audit(successful_rows(audit_module))
    assert result["coverage_complete"] and result["generation_complete"]
    assert result["checked"] == result["unique_cases"] == result["expected_cases"] == 924
    assert result["unique_text_stage_pairs"] == 132


def test_duplicate_rows_cannot_inflate_action_coverage(audit_module):
    row = successful_rows(audit_module)[0]
    with pytest.raises(ValueError, match="Duplicate case"):
        audit_module.audit([row, row])


@pytest.mark.parametrize("field", ["text", "history", "game_before", "evidence", "stage", "section"])
def test_changed_fixture_cannot_keep_a_trusted_action_case_id(audit_module, field):
    row = successful_rows(audit_module)[0]
    with pytest.raises(ValueError, match="fixture"):
        audit_module.audit([{**row, field: "wrong fixture"}])


def test_unknown_custom_slug_does_not_default_to_no_action(audit_module):
    row = successful_rows(audit_module)[0]
    with pytest.raises(ValueError, match="Unknown custom case"):
        audit_module.audit([{**row, "id": "custom/opening/unknown/neutral"}])


def test_failed_generation_is_not_presented_as_completed_interaction(audit_module):
    rows = successful_rows(audit_module)
    rows[0] = {**rows[0], "error": "truncated", "finish_reason": "length"}
    result = audit_module.audit(rows)
    assert result["coverage_complete"] and result["failed"] == 0
    assert not result["generation_complete"]
    assert result["generation_unsuccessful_ids"] == [rows[0]["id"]]


def test_action_mismatch_is_retained_as_failure(audit_module):
    row = successful_rows(audit_module)[0]
    row = {**row, "game_after": {"completed": 1, "route": "bridge", "phase": "depart"}}
    result = audit_module.audit([row])
    assert result["failed"] == 1
    assert result["failures"][0]["id"] == row["id"]
