"""Predeclare and run six two-turn local interaction checks, not accuracy tests.

Use ``--home ... --output ... prepare`` before ``... run``. Preparation only
reads/hashes local artifacts and CPU caches. Execution uses CheckInPipeline
unchanged, including its current generator sampler; outputs are not deterministic.
Runs are not resumed or overwritten: another attempt needs another declared
output directory so failed/partial sessions remain visible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .data import read_manifest
from .downloads import LLAMA_REVISION
from .generator import SYSTEM_PROMPT
from .models import DIMENSIONS, load_head
from .robustness import identity, verified_feature_contract
from .schema import CheckInState
from .settings import GGUF_NAME, GGUF_SHA256, LABELS
from .train import atomic_json, file_sha256, load_cache


FOLLOWUPS = (
    "I want to talk a little more about how that felt.",
    "I am still sorting out my feelings. Could we slow down and talk about them?",
)


def select_sessions(rows, cache):
    """Use manifest order and cached eligibility, never predictions or labels."""
    eligible = {str(row_id) for row_id, q in zip(cache["ids"], cache["quality"]) if q[2] > 0}
    chosen, seen_ids, seen_media = [], set(), set()
    for row in rows:
        if row.get("split") != "dev" or row.get("id") not in eligible:
            continue
        if row["id"] in seen_ids or row.get("media_sha256") in seen_media:
            continue
        if not row.get("text") or not row.get("video_path") or not row.get("media_sha256"):
            raise ValueError("An eligible development utterance has incomplete source metadata")
        seen_ids.add(row["id"])
        seen_media.add(row["media_sha256"])
        chosen.append(row)
        if len(chosen) == 12:
            break
    if len(chosen) < 12:
        raise ValueError("This protocol requires 12 distinct eligible development clips")

    def real_turn(text_row, video_row, turn_id, kind):
        return {"turn_id": str(turn_id), "role": "user", "input_kind": kind,
                "text": text_row["text"].strip(), "text_source_id": text_row["id"],
                "video_source_id": video_row["id"], "video_path": video_row["video_path"],
                "media_sha256": video_row["media_sha256"], "expected_source": "fusion",
                "true_emotion_label": None, "history_roles": [] if turn_id == 1 else ["user", "assistant"]}

    sessions = []
    for index in range(6):
        mode = "missing_vision" if index < 2 else "fresh_real_pair" if index < 4 else "synthetic_mismatch"
        first = real_turn(chosen[index], chosen[index], 1, "real_dev_pair")
        if index < 2:
            second = {"turn_id": "2", "role": "user", "input_kind": "authored_followup_no_video",
                      "text": FOLLOWUPS[index], "text_source_id": None, "video_source_id": None,
                      "video_path": None, "media_sha256": None, "expected_source": "text_fallback",
                      "true_emotion_label": None, "history_roles": ["user", "assistant"]}
        elif index < 4:
            second = real_turn(chosen[index + 4], chosen[index + 4], 2, "fresh_real_dev_pair")
        else:
            second = real_turn(chosen[index + 4], chosen[index + 6], 2, "synthetic_mismatched_dev_text_and_video")
        sessions.append({"session_id": f"interaction-dev-{index+1:02d}", "mode": mode, "turns": [first, second]})
    return sessions


def fingerprints(home, cache):
    """Pin the complete input contract without constructing encoders or calling a model."""
    home = Path(home)
    contract = verified_feature_contract(home, cache)
    heads = {}
    for stage, dimension in DIMENSIONS.items():
        path = home / "checkpoints" / f"{stage}.pt"
        model, payload = load_head(path, device="cpu")
        if (payload.get("stage") != stage or payload.get("input_dim") != dimension
                or payload.get("labels") != LABELS or payload.get("feature_identity") != cache["feature_identity"]):
            raise ValueError("Interaction head does not match the declared feature/label contract")
        heads[stage] = {"sha256": file_sha256(path), "architecture": payload.get("architecture", "mlp"),
                        "parameters": sum(parameter.numel() for parameter in model.parameters())}
    model_path = home / "models/qwen" / GGUF_NAME
    generator_sha = file_sha256(model_path)
    if generator_sha != GGUF_SHA256:
        raise ValueError("Generator bytes differ from the pinned local Qwen artifact")
    package = Path(__file__).parent
    return {"feature_identity": cache["feature_identity"], "dev_cache_sha256": cache["sha256"],
            "feature_contract": contract, "head_checkpoints": heads,
            "generator": {"artifact": GGUF_NAME, "sha256": generator_sha, "llama_revision": LLAMA_REVISION,
                          "server_binary_sha256": file_sha256(home / "vendor/llama/llama-server.exe"),
                          "base_url": "http://127.0.0.1:8081", "sampler": "Unchanged LocalGenerator.stream implementation; no seed override"},
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "source_sha256": {name: file_sha256(package / name) for name in
                              ("interaction_backtest.py", "pipeline.py", "generator.py", "schema.py", "models.py")}}


def build_protocol(home):
    home = Path(home).resolve()
    cache = load_cache(home, "dev")
    sessions = select_sessions(read_manifest(home, "dev"), cache)
    for session in sessions:
        for turn in session["turns"]:
            if turn["video_path"] and file_sha256(turn["video_path"]) != turn["media_sha256"]:
                raise ValueError("Selected development media bytes differ from the manifest")
    return {"schema_version": 1, "split": "dev", "session_count": 6, "turn_count": 12,
            "fingerprints": fingerprints(home, cache), "sessions": sessions,
            "selection": "First twelve distinct cached-eligible development clips in manifest order; no prediction or label selection.",
            "history_protocol": "Reset to an empty list for each session. Turn two receives exactly the first user message and its actual completed assistant response. Prior messages never cross sessions.",
            "checks": ["event and state IDs", "validated state schema", "one completed nonempty response per turn",
                       "state before text and one done event", "streamed text equals final response", "expected fusion/text fallback",
                       "fresh or absent second-turn media", "declared history roles", "finite backend timings"],
            "scope": "Full local pipeline interaction and modality transitions. No browser, capture, therapeutic-quality, or emotion-accuracy score.",
            "input_notes": "MELD turns are replay fragments, not a newly recorded coherent human conversation. Authored follow-ups and synthetic mismatches have no ground-truth emotion label. Real-pair reference labels are intentionally not passed or scored.",
            "run_policy": "No hidden warmup, sampler override, automatic retry, or result overwrite. First-turn model loading remains visible. A fresh declared directory is required for another attempt."}


def prepare(home, output):
    output = Path(output).resolve()
    candidate = build_protocol(home)
    path = output / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        saved = {k: v for k, v in existing.items() if k not in {"created_at", "protocol_identity"}}
        if candidate != saved or existing.get("protocol_identity") != identity(saved):
            raise ValueError("Existing interaction protocol differs; choose a new output directory")
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepare the interaction protocol in an empty output directory")
    candidate["protocol_identity"] = identity(candidate)
    candidate["created_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, candidate)
    return candidate


def _validate_events(events, session_id, turn):
    errors, states, final_states, deltas = [], [], [], []
    seen_state = False
    for event in events:
        if event.get("session_id") != session_id or event.get("turn_id") != turn["turn_id"]:
            errors.append("Event session/turn identity differs from the protocol")
        kind = event.get("type")
        if final_states:
            errors.append("Event received after the done event")
        if kind in {"state", "done"}:
            try:
                state = CheckInState.model_validate(event.get("state")).model_dump()
                states.append(state)
                if state["session_id"] != session_id or state["turn_id"] != turn["turn_id"]:
                    errors.append("State session/turn identity differs from the protocol")
                if state["input"].get("text") != turn["text"] or state["input"].get("video_present") is not bool(turn["video_path"]):
                    errors.append("State input differs from the declared message/media availability")
                if kind == "state":
                    seen_state = True
                else:
                    final_states.append(state)
            except (ValueError, TypeError) as error:
                errors.append(f"Invalid state schema: {error}")
        elif kind == "text_delta":
            if not seen_state:
                errors.append("Response text arrived before the structured state")
            if not isinstance(event.get("text"), str):
                errors.append("Text delta is not a string")
            else:
                deltas.append(event["text"])
        elif kind == "error":
            errors.append(str(event.get("error") or "Pipeline error"))
        else:
            errors.append(f"Unknown event type: {kind}")
    if len(final_states) != 1:
        errors.append("Expected exactly one completed state")
    final = final_states[-1] if final_states else None
    if final is not None:
        if final["response"]["status"] != "complete" or not final["response"]["text"].strip():
            errors.append("Response did not complete with nonempty text")
        if not deltas or "".join(deltas) != final["response"]["text"]:
            errors.append("Streamed text does not match the final response")
        if final["emotion"]["source"] != turn["expected_source"]:
            errors.append("Emotion source differs from expected availability/fallback")
        if final["vision"].get("available") is not (turn["expected_source"] == "fusion"):
            errors.append("Visual availability differs from the planned media transition")
        for name in ("classification_ms", "first_token_ms", "completion_ms", "model_load_ms"):
            value = final["timing"].get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                errors.append(f"Missing or invalid backend timing: {name}")
    return {"passed": not errors, "errors": errors, "final_state": final, "streamed_text": "".join(deltas)}


def run(home, output, pipeline=None):
    """Run the already-declared protocol; an injected pipeline is for CPU tests only."""
    home, output = Path(home).resolve(), Path(output).resolve()
    protocol_path = output / "protocol.json"
    if not protocol_path.is_file():
        raise ValueError("Prepare the interaction protocol before running it")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    current = build_protocol(home)
    saved = {k: v for k, v in protocol.items() if k not in {"created_at", "protocol_identity"}}
    if current != saved or protocol.get("protocol_identity") != identity(saved):
        raise ValueError("Model, prompt, media or code changed after preparation; declare a new run")
    if (output / "report.json").exists() or (output / "run-started.json").exists():
        raise ValueError("This interaction run has already started; preserve it and declare another directory")
    # Exclusive creation prevents concurrent callers sharing session identities.
    with (output / "run-started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": datetime.now(timezone.utc).isoformat(), "protocol_identity": protocol["protocol_identity"]}, handle)
    if pipeline is None:
        from .pipeline import CheckInPipeline
        pipeline = CheckInPipeline(home)
    report = {"schema_version": 1, "protocol_identity": protocol["protocol_identity"],
              "protocol_created_at": protocol["created_at"], "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "sessions": [], "completed_turns": 0, "attempted_turns": 0,
              "scope": protocol["scope"], "quality_score": None}
    started = time.perf_counter()
    try:
        for declared in protocol["sessions"]:
            history = []
            session = {"session_id": declared["session_id"], "mode": declared["mode"], "turns": []}
            report["sessions"].append(session)
            for turn in declared["turns"]:
                if [message["role"] for message in history] != turn["history_roles"]:
                    raise RuntimeError("History roles differ from the predeclared session reset contract")
                if turn["turn_id"] == "2" and turn["video_path"] == declared["turns"][0]["video_path"]:
                    raise RuntimeError("Second turn unexpectedly reuses the first clip")
                if turn["video_path"] and file_sha256(turn["video_path"]) != turn["media_sha256"]:
                    raise RuntimeError("Media changed during the interaction run")
                result = {"input": turn, "history_sent": json.loads(json.dumps(history)), "events": []}
                session["turns"].append(result)
                report["attempted_turns"] += 1
                before = time.perf_counter()
                stream = None
                try:
                    stream = pipeline.stream(turn["text"], turn["video_path"], history.copy(), declared["session_id"], turn["turn_id"])
                    for event in stream:
                        result["events"].append(json.loads(json.dumps(event, allow_nan=False)))
                except Exception as error:
                    result["runner_error"] = f"{type(error).__name__}: {error}"
                finally:
                    if stream is not None and hasattr(stream, "close"):
                        stream.close()
                result["wall_ms"] = (time.perf_counter() - before) * 1000
                result["checks"] = _validate_events(result["events"], declared["session_id"], turn)
                if "runner_error" in result:
                    result["checks"]["passed"] = False
                    result["checks"]["errors"].append(result["runner_error"])
                atomic_json(output / "report.json", report)
                if not result["checks"]["passed"]:
                    raise RuntimeError("Interaction contract failed; retained trace shows the failure")
                response = result["checks"]["final_state"]["response"]["text"]
                history.extend([{"role": "user", "content": turn["text"]}, {"role": "assistant", "content": response}])
                report["completed_turns"] += 1
                print(f"interaction {report['completed_turns']}/12 {declared['session_id']} turn {turn['turn_id']}", flush=True)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare(args.home, args.output) if args.action == "prepare" else run(args.home, args.output)
    print(json.dumps({key: result[key] for key in ("protocol_identity", "status", "completed_turns", "attempted_turns") if key in result}, indent=2))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
