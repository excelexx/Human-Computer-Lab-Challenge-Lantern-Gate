"""Study-design checks; constructed inputs are never model-quality evidence."""
import json

import pytest

from checkin.response_backtest import authored_cases, canonical_hash, pattern_flags, read_protocol, run_stage


def test_authored_splits_do_not_overlap_and_pair_text_only_once_per_case():
    cases = authored_cases()
    assert len([case for case in cases if case["split"] == "dev"]) == 10
    assert len([case for case in cases if case["split"] == "check"]) == 6
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["text"] for case in cases}) == len(cases)
    for case in cases:
        assert len(case["conditions"]) == 3
        missing = case["conditions"][-1]["state"]
        assert missing["vision"]["available"] is False
        assert missing["modalities"]["vision_label"] is None
        assert missing["emotion"]["source"] == "text_fallback"
        assert all("text" not in condition for condition in case["conditions"])


def test_declared_protocol_rejects_modification(tmp_path):
    protocol = {"cases": [{"text": "original"}]}
    protocol["protocol_sha256"] = canonical_hash(protocol)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    assert read_protocol(tmp_path) == protocol
    protocol["cases"][0]["text"] = "changed after declaration"
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        read_protocol(tmp_path)


def test_reserved_stage_requires_review_of_exact_dev_output_before_network(tmp_path):
    protocol = {"prompts": {"baseline": "fixed"}}
    protocol["protocol_sha256"] = canonical_hash(protocol)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    selection = {"protocol_sha256": protocol["protocol_sha256"], "selected_prompt": "baseline",
                 "development_review": "Review completed", "dev_responses_sha256": "different"}
    (tmp_path / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (tmp_path / "responses-dev.jsonl").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="exact completed development"):
        run_stage(tmp_path, "check")
    assert not (tmp_path / "responses-check.jsonl").exists()


def test_pattern_flags_preserve_narrow_lexical_evidence_without_quality_verdict():
    # Even negated matches remain visible for qualitative review, never silently
    # converted to a quality or clinical verdict by this deliberately narrow check.
    flags = pattern_flags("I cannot say this is normal.", ["medical_assertion"])
    assert flags == [{"check": "medical_assertion", "matches": ["normal"]}]
    assert pattern_flags("How does that feel for you?", ["medical_assertion"]) == []
