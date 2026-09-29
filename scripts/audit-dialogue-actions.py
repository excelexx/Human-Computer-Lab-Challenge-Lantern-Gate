"""Audit saved custom-case actions against separately authored expectations.

This reads transcripts; it performs no inference and never judges dialogue prose.
"""
import argparse
import hashlib
import json
from pathlib import Path

import dialogue_matrix_cases as cases


def expected_after(item):
    slug = item["id"].split("/")[2]
    before = item["game_before"]
    route = before["route"]
    phase = "talking"
    if slug in ("bridge", "stairs"):
        route = slug
        if before["completed"] >= 2 or (before["completed"] >= 1 and before["route"]):
            phase = "depart"
    elif slug in ("change", "exclude_bridge"):
        route = "stairs"  # Preference selection; permission to leave is separate.
    elif slug in ("negated_bridge", "curly_negation", "will_not") and route == "bridge":
        route = None
    elif slug == "other_route" and route:
        route = "stairs" if route == "bridge" else "bridge"
    elif slug == "dont_worry" and before["route"] and before["completed"] >= 1:
        phase = "depart"
    return {"completed": min(3, before["completed"] + 1), "route": route, "phase": phase}


def audit(rows):
    """Check fixed fixture identity before applying independently authored expectations."""
    expected_cases = {item["id"]: item for item in cases.matrix_cases() if item["section"] == "customs"}
    seen = set()
    checks, unsuccessful = [], []
    for row in rows:
        case_id = row["id"]
        if case_id in seen:
            raise ValueError(f"Duplicate case ID: {case_id}")
        seen.add(case_id)
        if row.get("section") != "customs":
            if case_id in expected_cases:
                raise ValueError(f"Custom fixture has wrong section: {case_id}")
            continue
        fixture = expected_cases.get(case_id)
        if fixture is None:
            raise ValueError(f"Unknown custom case ID: {case_id}")
        for field, value in fixture.items():
            if row.get(field) != value:
                raise ValueError(f"Custom fixture mismatch for {case_id}: {field}")
        checks.append({"id": case_id, "text": row["text"], "expected": expected_after(row),
                       "actual": row["game_after"]})
        if (row.get("error") or row.get("finish_reason") != "stop"
                or not isinstance(row.get("response"), str) or not row["response"].strip()):
            unsuccessful.append(case_id)
    present = {item["id"] for item in checks}
    missing = sorted(set(expected_cases) - present)
    result = {"scope": "Separately authored expectations for proposed quest transitions in the fixed custom corpus. This does not judge prose or prove that the app committed movement. coverage_complete identifies partial runs.",
              "integrity_version": 2, "case_definition_sha256": hashlib.sha256(Path(cases.__file__).read_bytes()).hexdigest(),
              "checked": len(checks), "expected_cases": len(expected_cases), "unique_cases": len(present),
              "coverage_complete": not missing, "missing_ids": missing,
              "unique_text_stage_pairs": len({row["id"].rsplit("/", 1)[0] for row in checks}),
              "generation_unsuccessful_ids": unsuccessful, "generation_complete": not missing and not unsuccessful,
              "failures": [row for row in checks if row["expected"] != row["actual"]]}
    result["failed"] = len(result["failures"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines()]
    result = audit(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("failures", "missing_ids")}))


if __name__ == "__main__":
    main()
