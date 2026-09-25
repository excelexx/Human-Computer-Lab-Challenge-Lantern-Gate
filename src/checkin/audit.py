"""Auditable total learned-parameter accounting for the local inference path."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .downloads import artifacts, model_manifest, sha256_file
from .models import DIMENSIONS, EmotionHead, LinearEmotionHead, load_head
from .settings import LABELS, runtime_home


def describe_head(stage, model):
    """Count the actual module, including frozen parameters and learned biases."""
    if isinstance(model, LinearEmotionHead):
        dimensions = [model.net.in_features, model.net.out_features]
        kind = "linear"
    elif isinstance(model, EmotionHead):
        dimensions = [model.net[0].in_features, model.net[0].out_features, model.net[-1].out_features]
        kind = "mlp"
    else:
        raise ValueError(f"Unsupported runtime head module: {type(model).__name__}")
    if dimensions[0] != DIMENSIONS[stage] or dimensions[-1] != len(LABELS):
        raise ValueError(f"The {stage} head has incompatible input/output dimensions")
    return {"name": f"MELD {stage} head", "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "architecture_kind": kind, "architecture": " -> ".join(map(str, dimensions))}


def runtime_inventory(home, loaded_heads=None, verify_hashes=True):
    """Resolve runtime heads on CPU; loaded modules take precedence over disk.

    Missing/invalid checkpoints use the shipped architecture solely as an
    explicitly planned count. Such an inventory is never a verified audit.
    """
    home = Path(home)
    planned = {stage: next(c for c in model_manifest()["components"] if c["name"] == f"MELD {stage} head") for stage in DIMENSIONS}
    components, checks = {}, []
    for stage, dimension in DIMENSIONS.items():
        path = home / "checkpoints" / f"{stage}.pt"
        record = {"name": stage, "path": str(path)}
        description = None
        try:
            if loaded_heads is not None:
                description = describe_head(stage, loaded_heads[stage])
                record.update(status="verified", count_method="loaded inference module")
            elif path.is_file():
                model, payload = load_head(path, device="cpu")
                if payload.get("stage") != stage or payload.get("input_dim") != dimension or payload.get("labels") != LABELS:
                    raise ValueError("Checkpoint stage, input dimension or label order differs from the runtime contract")
                description = describe_head(stage, model)
                record.update(status="verified", count_method="loaded checkpoint")
                if verify_hashes:
                    record["sha256"] = sha256_file(path)
            else:
                record["status"] = "planned_missing_checkpoint"
        except Exception as error:
            record.update(status="invalid", error=str(error))
        if description is None:
            description = {key: planned[stage][key] for key in ("name", "parameters", "architecture_kind", "architecture")}
            record["count_method"] = "shipped architecture plan; actual checkpoint not verified"
        record.update({key: value for key, value in description.items() if key != "name"})
        components[stage] = {**description, "local_count_status": record["status"], "count_method": record["count_method"],
                             "count_kind": "exact loaded architecture" if record["status"] == "verified" else "planned architecture"}
        if "sha256" in record:
            components[stage]["checkpoint_sha256"] = record["sha256"]
        checks.append(record)
    return model_manifest(components), checks


def audit_parameters(home: str | Path) -> dict:
    """Verify pinned inference artifacts and count every actual trained head.

    Large pretrained counts come from their recorded architecture/checkpoint
    audits, linked to local files by SHA256. This does not claim to reload and
    recount the quantized generator or perform an inference benchmark.
    """
    home = runtime_home(home)
    manifest, head_checks = runtime_inventory(home)
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
    actual_head_total = sum(record["parameters"] for record in head_checks)
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
