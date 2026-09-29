"""Resumable local dialogue development matrix and exhaustive finite policy checks.

Examples (from the repository with the environment installed):
  python scripts/evaluate-dialogue-matrix.py --output reports/run --kind matrix
  python scripts/evaluate-dialogue-matrix.py --output reports/run --kind checks
  python scripts/evaluate-dialogue-matrix.py --output reports/run --sections presets --limit 5

Set PYTHONPATH to an archived source tree to benchmark that production revision.
This script's sibling case definitions stay fixed. Existing JSONL is appended only
when the code/case/sampling fingerprint matches. Errors are recorded, not retried
silently; choose a new output directory for a different implementation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import re
import time

import httpx

import checkin.character as character
import checkin.generator as generator
import checkin.quest as quest
import checkin.scene as scene
import dialogue_matrix_cases as cases

SAMPLING = {"model": "checkin-qwen", "stream": False, "max_tokens": generator.MAX_OUTPUT_TOKENS,
            "temperature": getattr(generator, "GENERATION_TEMPERATURE", .5), "top_p": .8, "top_k": 20, "min_p": 0., "seed": 314}
BASE_URL = "http://127.0.0.1:8081"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def metadata(output, label):
    files = {name: Path(module.__file__) for name, module in
             (("character", character), ("generator", generator), ("quest", quest),
              ("scene", scene), ("cases", cases))}
    files["runner"] = Path(__file__)
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    plan = cases.matrix_cases()
    protocol = {
        "version": 3, "label": label, "sampling": SAMPLING, "source_sha256": source_hashes,
        "runner_integrity": "Version 3 adds cached-record validation and explicit coverage accounting. This integrity hardening does not change generation inputs, the corpus, or sampling.",
        "case_sha256": digest(plan), "journey_sha256": digest(cases.journey_cases()),
        "base_url": BASE_URL,
        "counts": dict(Counter(item["section"] for item in plan)),
        "preset_paths": 4 ** 3 * 7 ** 3, "raw_authored_policy_cases": 12 * 7 * 7 * 2 * 2,
        "interpretation": "Synthetic evidence interventions through local Qwen. Authored canonical histories except journeys. Not webcam/emotion accuracy or an exhaustive set of arbitrary custom text, generated histories, or random seeds.",
    }
    fingerprint = digest(protocol)
    path = output / "metadata.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise RuntimeError("Fingerprint changed. Preserve the previous run and choose a new --output directory.")
        return prior
    existing = [item.name for item in output.iterdir() if item.name != ".runner.lock"]
    if existing:
        raise RuntimeError("The output directory has files without metadata; choose an empty directory.")
    value = {**protocol, "fingerprint": fingerprint, "created_at": utcnow(),
             "source_paths": {name: str(path.resolve()) for name, path in files.items()},
             "system_prompt": character.SYSTEM_PROMPT,
             "system_prompt_sha256": hashlib.sha256(character.SYSTEM_PROMPT.encode()).hexdigest()}
    atomic_json(path, value)
    return value


def read_jsonl(path):
    records = {}
    if not path.exists():
        return records
    # A interrupted final write is preserved for inspection, then discarded only
    # from the append log so the incomplete case can be generated again.
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        tail = raw[boundary:]
        archive = path.with_name(path.name + f".interrupted-{time.time_ns()}")
        archive.write_bytes(tail)
        with path.open("r+b") as handle:
            handle.truncate(boundary)
        raw = raw[:boundary]
    for line in raw.splitlines():
        item = json.loads(line)
        if item["id"] in records:
            raise RuntimeError(f"Duplicate case ID in {path}: {item['id']}")
        records[item["id"]] = item
    return records


def append_record(handle, records, item):
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    handle.flush()
    records[item["id"]] = item


def review_flags(item):
    """Cheap review candidates only; no claim that regex judges coherence."""
    text = item.get("response", "")
    flags = []
    if item.get("error"):
        flags.append("generation_error")
    if not text.strip():
        flags.append("empty")
    if item.get("finish_reason") != "stop":
        flags.append("non_stop_finish")
    if "*" in text or "```" in text:
        flags.append("markdown")
    if text.count("?") > 1:
        flags.append("multiple_questions")
    if re.search(r"\b(safe|safer|safest|guaranteed|dangerous)\b", text, re.I):
        flags.append("review_safety_word")
    if re.search(r"\b(your face|your smile|you look|I can see you|I see your)\b", text, re.I):
        flags.append("review_visual_claim")
    if len(text.split()) > 45:
        flags.append("long_response")
    return flags


def generate(client, item):
    game = item["game_before"]
    state = cases.state_for(item["text"], game, **item["evidence"])
    result = {**item, "started_at": utcnow(), "state": state,
              "context": generator._turn_context(item["text"], state),
              "game_after": quest.preview(game, item["text"]), "error": None}
    start = time.perf_counter()
    try:
        messages = generator._messages(item["text"], state, item["history"])
        result["messages_before_fit"] = messages
        messages = generator._fit_context(client, BASE_URL, messages, time.monotonic() + 90)
        result["messages"] = messages
        response = client.post(BASE_URL + "/v1/chat/completions",
                               json={**SAMPLING, "messages": messages}).raise_for_status().json()
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise RuntimeError("The local generator returned malformed choices.")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        result.update(response=content if isinstance(content, str) else "", finish_reason=choice.get("finish_reason"),
                      usage=response.get("usage"), model=response.get("model"))
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The local generator returned no response text.")
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(f"The local generator did not complete normally ({choice.get('finish_reason')!r}); partial text is preserved.")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result.setdefault("response", "")
    result["seconds"] = time.perf_counter() - start
    result["review_flags"] = review_flags(result)
    return result


def validate_journey_record(saved, expected):
    """A cached turn must belong to this exact reconstructed dependency chain."""
    for field, value in expected.items():
        if saved.get(field) != value:
            raise RuntimeError(f"Saved journey dependency mismatch for {expected['id']}: {field}")
    after = quest.preview(expected["game_before"], expected["text"])
    if saved.get("game_after") != after:
        raise RuntimeError(f"Saved journey dependency mismatch for {expected['id']}: game_after")


def accepted_response(saved):
    content = saved.get("response")
    return (not saved.get("error") and saved.get("finish_reason") == "stop"
            and isinstance(content, str) and bool(content.strip()))


def journey_item(name, index, text, emotion, game, history):
    return {"id": f"journey/{name}/{index}", "section": "journeys", "stage": f"turn_{index}",
            "text": text, "emotion_intervention": emotion,
            "history_kind": "actual_generated_preceding_turns",
            "review_concern": "Continue actual generated history; preserve route, consent and facts.",
            "game_before": game, "history": history,
            "evidence": {"fused": emotion, "visual": emotion}}


def validate_cached_record(saved, expected):
    validate_journey_record(saved, expected)
    state = cases.state_for(expected["text"], expected["game_before"], **expected["evidence"])
    if saved.get("state") != state:
        raise RuntimeError(f"Saved case fixture mismatch for {expected['id']}: state")
    if saved.get("context") != generator._turn_context(expected["text"], state):
        raise RuntimeError(f"Saved case fixture mismatch for {expected['id']}: context")


def validate_cached_matrix(records, plan):
    """Validate every cached input, including cases outside the requested section."""
    expected = {item["id"]: item for item in plan}
    journey_ids = {f"journey/{name}/{index}" for name, turns in cases.journey_cases().items()
                   for index in range(len(turns))}
    unknown = set(records) - set(expected) - journey_ids
    if unknown:
        raise RuntimeError(f"Unknown saved case IDs: {sorted(unknown)}")
    for case_id, item in expected.items():
        if case_id in records:
            validate_cached_record(records[case_id], item)
    for name, turns in cases.journey_cases().items():
        game, history = quest.initial_quest(), []
        for index, (text, emotion) in enumerate(turns):
            case_id = f"journey/{name}/{index}"
            later_saved = any(f"journey/{name}/{later}" in records for later in range(index + 1, len(turns)))
            if case_id not in records:
                if later_saved:
                    raise RuntimeError(f"Saved journey has a missing dependency: {case_id}")
                break
            saved = records[case_id]
            validate_cached_record(saved, journey_item(name, index, text, emotion, game, history))
            if not accepted_response(saved):
                if later_saved:
                    raise RuntimeError(f"Saved journey continues after a failed response: {case_id}")
                break
            game = saved["game_after"]
            history = [*history, {"role": "user", "content": text},
                       {"role": "assistant", "content": saved["response"]}]


def coverage(records, expected_sections, requested_sections=None):
    missing = sorted(set(expected_sections) - set(records))
    requested = set(expected_sections) if requested_sections is None else {
        case_id for case_id, section in expected_sections.items() if section in requested_sections}
    return {"expected_cases": len(expected_sections), "unique_cases": len(records),
            "coverage_complete": not missing, "missing_ids": missing,
            "expected_by_section": dict(Counter(expected_sections.values())),
            "requested_expected_cases": len(requested),
            "requested_sections_complete": requested <= set(records)}


def run_matrix(output, sections, limit):
    path = output / "responses.jsonl"
    records = read_jsonl(path)
    plan = cases.matrix_cases()
    validate_cached_matrix(records, plan)
    added = 0
    with httpx.Client(trust_env=False, timeout=90) as client, path.open("a", encoding="utf-8") as handle:
        if limit != 0:
            props = client.get(BASE_URL + "/props").raise_for_status().json()
            models = client.get(BASE_URL + "/v1/models").raise_for_status().json()
            identity = {"model_ids": [item.get("id") for item in models.get("data", [])],
                        "model_path": props.get("model_path"),
                        "context": props.get("default_generation_settings", {}).get("n_ctx"),
                        "chat_template_sha256": digest(props.get("chat_template")),
                        "evidence_limit": "Server-reported identity; weight integrity is recorded in the project's model manifests, not rehashed for every matrix."}
            identity_path = output / "server-identity.json"
            if identity_path.exists():
                if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                    raise RuntimeError("Local server identity changed; choose a new output directory.")
            else:
                atomic_json(identity_path, identity)
        for item in plan:
            if item["section"] not in sections or item["id"] in records:
                continue
            if limit is not None and added >= limit:
                break
            append_record(handle, records, generate(client, item))
            added += 1
            if added % 25 == 0:
                print(json.dumps({"new_cases": added, "saved_cases": len(records), "last_id": item["id"]}), flush=True)
        if "journeys" in sections:
            for name, turns in cases.journey_cases().items():
                game, history = quest.initial_quest(), []
                for index, (text, emotion) in enumerate(turns):
                    case_id = f"journey/{name}/{index}"
                    item = journey_item(name, index, text, emotion, game, history)
                    if case_id not in records:
                        if any(f"journey/{name}/{later}" in records for later in range(index + 1, len(turns))):
                            raise RuntimeError(f"Saved journey has a missing dependency: {case_id}")
                        if limit is not None and added >= limit:
                            break
                        append_record(handle, records, generate(client, item))
                        added += 1
                    saved = records[case_id]
                    validate_journey_record(saved, item)
                    if not accepted_response(saved):
                        if any(f"journey/{name}/{later}" in records for later in range(index + 1, len(turns))):
                            raise RuntimeError(f"Saved journey continues after a failed response: {case_id}")
                        break  # A failed response must never silently advance a journey.
                    game = saved["game_after"]
                    history = [*history, {"role": "user", "content": text},
                               {"role": "assistant", "content": saved["response"]}]
    expected_sections = {item["id"]: item["section"] for item in plan}
    expected_sections.update({f"journey/{name}/{index}": "journeys"
                              for name, turns in cases.journey_cases().items() for index in range(len(turns))})
    summary = {"updated_at": utcnow(), "new_cases": added, "saved_cases": len(records),
               **coverage(records, expected_sections, sections),
               "sections": dict(Counter(item["section"] for item in records.values())),
               "finish_reasons": dict(Counter(str(item.get("finish_reason")) for item in records.values())),
               "review_flags": dict(Counter(flag for item in records.values() for flag in item["review_flags"])),
               "total_generation_seconds": sum(item["seconds"] for item in records.values()),
               "unsuccessful_ids": [key for key, item in records.items() if not accepted_response(item)],
               "note": "Flags are review candidates; absence of flags does not establish semantic correctness."}
    summary["generation_complete"] = summary["coverage_complete"] and not summary["unsuccessful_ids"]
    atomic_json(output / "response-summary.json", summary)
    print(json.dumps(summary), flush=True)


def deterministic_cases():
    for choices in product(range(4), repeat=3):
        for emotions in product(cases.EMOTIONS, repeat=3):
            game, trace, violations = quest.initial_quest(), [], []
            for step, (index, emotion) in enumerate(zip(choices, emotions)):
                text = quest.options(game)[index]
                before = dict(game)
                state = cases.state_for(text, game, emotion, emotion)
                direction = generator._turn_context(text, state)["npc_direction"]
                game = quest.preview(game, text)
                trace.append({"text": text, "emotion": emotion, "before": before, "after": game,
                              "game_context": state["game"], "direction": direction})
                if direction["cue_emotion"] != emotion or direction["direction_source"] != "ambiguous_demo_visual_cue":
                    violations.append(f"turn_{step}: authored visual cue was not retained")
            expected = {"completed": 3, "route": "bridge" if choices[1] in (0, 2) else "stairs",
                        "phase": "depart" if choices[2] < 3 else "talking"}
            if game != expected:
                violations.append(f"final_state: expected {expected}, got {game}")
            yield {"id": f"path/{''.join(map(str, choices))}/{'-'.join(emotions)}", "kind": "preset_path",
                   "choices": choices, "emotions": emotions, "trace": trace, "violations": violations}
    for index, text in enumerate(quest.ALL_EXAMPLES):
        stage = "opening" if index < 4 else "route" if index < 8 else "ready_bridge"
        game, _ = cases.fixture(stage)
        for fused, visual, available, disagreement in product(cases.EMOTIONS, cases.EMOTIONS, (False, True), (False, True)):
            state = cases.state_for(text, game, fused, visual, available=available, disagreement=disagreement)
            result = generator._turn_context(text, state)
            expected = visual if available else fused
            expected_source = "ambiguous_demo_visual_cue" if available else "combined_estimate"
            direction = result["npc_direction"]
            violations = []
            if direction["cue_emotion"] != expected or direction["direction_source"] != expected_source:
                violations.append("authored cue priority differs from expected")
            if not available and result["emotion_evidence"]["vision_emotion"] is not None:
                violations.append("unavailable vision leaked into evidence")
            yield {"id": f"policy/{index}/{fused}/{visual}/{int(available)}/{int(disagreement)}",
                   "kind": "authored_policy", "text": text, "state": state, "context": result,
                   "violations": violations}


def run_checks(output, limit):
    path = output / "checks.jsonl"
    records = read_jsonl(path)
    expected = {item["id"]: item for item in deterministic_cases()}
    unknown = set(records) - set(expected)
    if unknown:
        raise RuntimeError(f"Unknown saved check IDs: {sorted(unknown)}")
    for case_id, saved in records.items():
        if digest(saved) != digest(expected[case_id]):
            raise RuntimeError(f"Saved deterministic check mismatch: {case_id}")
    added = 0
    with path.open("a", encoding="utf-8") as handle:
        for item in expected.values():
            if item["id"] in records:
                continue
            if limit is not None and added >= limit:
                break
            append_record(handle, records, item)
            added += 1
    failed = [item["id"] for item in records.values() if item["violations"]]
    summary = {"updated_at": utcnow(), "new_cases": added, "saved_cases": len(records),
               **coverage(records, {key: item["kind"] for key, item in expected.items()}),
               "counts": dict(Counter(item["kind"] for item in records.values())),
               "violation_cases": len(failed), "violating_ids": failed,
               "scope": "Finite three-turn preset paths and raw authored cue policy configurations; coverage_complete records whether every planned case is present. No language generations or unlimited refusal loops in these checks."}
    atomic_json(output / "check-summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="development")
    parser.add_argument("--kind", choices=("matrix", "checks", "journeys", "all"), default="matrix")
    parser.add_argument("--sections", default="presets,customs,policy,exploratory,journeys")
    parser.add_argument("--limit", type=int, help="Maximum NEW records per kind; a resumed run skips saved cases.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    sections = set(args.sections.split(","))
    if not sections <= {"presets", "customs", "policy", "exploratory", "journeys"}:
        parser.error("Unknown section")
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two GPU runners sharing a run directory. Delete a
    # stale lock manually only after checking no runner process still owns it.
    lock = args.output / ".runner.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_at": utcnow()}))
    try:
        value = metadata(args.output, args.label)
        print(json.dumps({"fingerprint": value["fingerprint"], "counts": value["counts"]}), flush=True)
        if args.kind in ("checks", "all"):
            run_checks(args.output, args.limit)
        if args.kind in ("matrix", "all"):
            run_matrix(args.output, sections, args.limit)
        elif args.kind == "journeys":
            run_matrix(args.output, {"journeys"}, args.limit)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
