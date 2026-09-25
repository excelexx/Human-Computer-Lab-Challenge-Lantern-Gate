"""Auditable total learned-parameter accounting for the local inference path."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .downloads import artifacts, model_manifest, sha256_file
from .models import DIMENSIONS, EmotionHead, load_head
from .settings import runtime_home


def audit_parameters(home: str | Path) -> dict:
    """Verify pinned inference artifacts and count every actual trained head.

    Large pretrained counts come from their recorded architecture/checkpoint
    audits, linked to local files by SHA256. This does not claim to reload and
    recount the quantized generator or perform an inference benchmark.
    """
    home = runtime_home(home)
    manifest = model_manifest()
    inventory = [dict(component) for component in manifest["components"]]
    checks = []
    for artifact in artifacts(include_meld=False):
        if artifact["kind"] != "inference_model":
            continue
        path = home / artifact["path"]
        check = {"name": artifact["name"], "path": str(path), "expected_sha256": artifact["sha256"]}
        if not path.is_file():
            check["status"] = "missing"
        else:
            check["actual_sha256"] = sha256_file(path)
            check["status"] = "verified" if check["actual_sha256"] == check["expected_sha256"] else "checksum_mismatch"
        checks.append(check)
    head_checks = []
    actual_head_total = 0
    for name, dimension in DIMENSIONS.items():
        path = home / "checkpoints" / f"{name}.pt"
        record = {"name": name, "path": str(path)}
        if path.is_file():
            try:
                model, payload = load_head(path, device="cpu")
                if payload["input_dim"] != dimension:
                    raise ValueError(f"Expected {dimension} input features, got {payload['input_dim']}.")
                record.update(status="verified", sha256=sha256_file(path))
            except Exception as error:
                model = EmotionHead(dimension)
                record.update(status="invalid", error=str(error))
        else:
            model = EmotionHead(dimension)
            record["status"] = "planned_missing_checkpoint"
        count = sum(parameter.numel() for parameter in model.parameters())
        record["parameters"] = count
        record["count_method"] = "loaded checkpoint" if record["status"] == "verified" else "untrained architecture; checkpoint not verified"
        actual_head_total += count
        head_checks.append(record)
        for component in inventory:
            if component["name"] == f"MELD {name} head":
                component["parameters"] = count
                component["local_count_status"] = record["status"]
    total = sum(component["parameters"] for component in inventory)
    errors = any(check["status"] == "checksum_mismatch" for check in checks) or any(check["status"] == "invalid" for check in head_checks)
    complete = all(check["status"] == "verified" for check in checks + head_checks)
    within_limit = total <= 6_000_000_000
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if errors or not within_limit else "verified" if complete else "planned",
        "within_parameter_limit": within_limit,
        "parameter_limit": 6_000_000_000,
        "total_parameter_upper_bound": total,
        "remaining_parameter_budget": 6_000_000_000 - total,
        "learned_head_parameters": actual_head_total,
        "components": inventory,
        "artifact_checks": checks,
        "head_checks": head_checks,
        "verification_scope": "Pinned inference artifact SHA256 checks and loaded trained-head parameter counts. Pretrained parameter counts use recorded architecture/checkpoint evidence, not bytes on disk. Missing heads remain explicitly planned.",
        "notes": [
            "The YuNet allowance includes 17 nonlearned constants, so the overall value is a conservative upper bound.",
            "All three runtime heads and the original AffectNet classifier are included.",
            "Qwen embeddings shared with the output layer count once; quantization does not alter learned count.",
            "No extra learned projector, tracker, adapter, response grader, or auxiliary generator is required.",
            "A parameter audit does not establish task accuracy, memory fit, or successful inference.",
        ],
    }
    (home / "reports/parameters.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (home / "manifests/models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report
