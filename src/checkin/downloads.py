"""Pinned, checksum-verified setup downloads; never invoked during inference."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from . import settings


TEXT_HASHES = {
    "config.json": "ddec8b81d079d218ce9e54fc0af5d1d5937d6d53b5d42e70c1f251a1cebc830d",
    "tokenizer_config.json": "3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "pytorch_model.bin": "dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5",
}
LLAMA_REVISION = "b11146"
LLAMA_ARCHIVES = (
    ("llama-bin.zip", "llama-b11146-bin-win-cuda-12.4-x64.zip", "3c806a6ceccc3dae1c743ceb1a1fb2cce5b76f40bfbd4c6b7b8afb6ef45a5807"),
    ("llama-cudart.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifacts(include_meld: bool = True) -> list[dict[str, str]]:
    items = [
        {"name": "yunet", "path": "models/vision/yunet.onnx", "url": settings.YUNET_URL, "sha256": settings.YUNET_SHA256, "kind": "inference_model"},
        {"name": "affectnet_efficientnet_b2", "path": "models/vision/enet_b2_7.pt", "url": settings.VISION_URL, "sha256": settings.VISION_SHA256, "kind": "inference_model"},
        {"name": "qwen3_4b_q5", "path": f"models/qwen/{settings.GGUF_NAME}", "url": settings.GGUF_URL, "sha256": settings.GGUF_SHA256, "kind": "inference_model"},
    ]
    for filename, digest in TEXT_HASHES.items():
        items.append({"name": f"deberta_{filename}", "path": f"models/text/{filename}",
                      "url": f"https://huggingface.co/{settings.TEXT_MODEL}/resolve/{settings.TEXT_REVISION}/{filename}",
                      "sha256": digest, "kind": "inference_model"})
    for filename, remote_name, digest in LLAMA_ARCHIVES:
        items.append({"name": filename, "path": f"downloads/{filename}",
                      "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_REVISION}/{remote_name}",
                      "sha256": digest, "kind": "runtime"})
    if include_meld:
        items.append({"name": "MELD.Raw.tar.gz", "path": "downloads/MELD.Raw.tar.gz", "url": settings.MELD_URL,
                      "sha256": settings.MELD_SHA256, "kind": "training_dataset"})
    return items


def model_manifest(head_components: dict[str, dict] | None = None) -> dict:
    """Describe the shipped stack, or replace head entries with actual counts.

    The default is an architecture plan. Runtime auditing supplies loaded head
    entries so older all-MLP checkpoints remain accurately accounted for.
    """
    manifest = {
        "schema_version": 1,
        "parameter_limit": 6_000_000_000,
        "parameter_policy": "All required unique learned weights, frozen or trainable; quantization does not reduce counts. YuNet count conservatively includes 17 constants.",
        "components": [
            {"name": "YuNet face detector", "parameters": 53_121, "count_kind": "conservative upper bound",
             "nonlearned_values_included": 17, "license": "MIT", "revision": "f12e12798e8314f7c074a6656816c048dcc95b7a",
             "source": "https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet",
             "count_evidence": "Sum of ONNX initializer elements; includes 17 nonlearned values."},
            {"name": "EmotiEffLib AffectNet EfficientNet-B2 including original FER head", "parameters": 7_710_857,
             "count_kind": "exact", "license": "Apache-2.0 repository; document AffectNet data terms separately",
             "revision": "520a051c64cd191521e5934655314e769a319684",
             "source": "https://github.com/sb-ai-lab/EmotiEffLib/tree/520a051c64cd191521e5934655314e769a319684",
             "count_evidence": "Loaded pinned author's module and counted unique learned parameters, retaining original classifier."},
            {"name": "DeBERTa-v3-large text encoder", "parameters": 434_012_160, "count_kind": "exact",
             "license": "MIT", "model_id": settings.TEXT_MODEL, "revision": settings.TEXT_REVISION,
             "source": f"https://huggingface.co/{settings.TEXT_MODEL}/tree/{settings.TEXT_REVISION}",
             "count_evidence": "Instantiated AutoModel encoder; excludes pretraining-only heads unused by local inference."},
            {"name": "MELD vision head", "parameters": 181_255, "count_kind": "exact architecture",
             "source": "src/checkin/models.py", "architecture_kind": "mlp", "architecture": "1408 -> 128 -> 7"},
            {"name": "MELD text head", "parameters": 7_175, "count_kind": "exact architecture",
             "source": "src/checkin/models.py", "architecture_kind": "linear", "architecture": "1024 -> 7"},
            {"name": "MELD fusion head", "parameters": 312_711, "count_kind": "exact architecture",
             "source": "src/checkin/models.py", "architecture_kind": "mlp", "architecture": "2435 -> 128 -> 7"},
            {"name": "Qwen3-4B-Instruct-2507", "parameters": 4_022_468_096, "count_kind": "exact",
             "license": "Apache-2.0", "model_id": "Qwen/Qwen3-4B-Instruct-2507",
             "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
             "source": "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json",
             "count_evidence": "Config reconstruction and original safetensors metadata agree; tied embeddings counted once. Q5 preserves this learned count.",
             "quantized_by": "bartowski", "quantization_revision": "ae44f08e1392f39c0e474af10c3ff8355c8b6688",
             "quantization_source": "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/blob/ae44f08e1392f39c0e474af10c3ff8355c8b6688/README.md"},
        ],
        "head_inventory_source": "shipped architecture plan; local checkpoints require auditing",
        "zero_parameter_operations": ["tokenizers", "geometric face selection", "mean pooling", "normalization", "browser UI", "llama.cpp runtime"],
        "training_only_learned_models": [],
        "evaluation_only_learned_models": [],
        "artifacts": artifacts(),
        "runtime": {"name": "llama.cpp", "revision": LLAMA_REVISION, "platform": "Windows x64 CUDA 12.4", "license": "MIT; accompanying CUDA libraries have NVIDIA license terms"},
        "dataset": {"name": "MELD", "source": "https://github.com/declare-lab/MELD", "note": "Acquisition instructions only. Do not redistribute television clips; record dataset/media terms separately."},
    }
    if head_components is not None:
        for stage, resolved in head_components.items():
            expected_name = f"MELD {stage} head"
            component = next((item for item in manifest["components"] if item["name"] == expected_name), None)
            if component is None or resolved.get("name", expected_name) != expected_name:
                raise ValueError(f"Unknown head inventory entry: {stage}")
            component.update(resolved)
        manifest["head_inventory_source"] = "runtime head inspection; see each component's local_count_status"
    manifest["total_parameter_upper_bound"] = sum(item["parameters"] for item in manifest["components"])
    return manifest


def _download(artifact: dict[str, str], home: Path) -> dict:
    destination = home / artifact["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = artifact["sha256"]
    if destination.exists():
        actual = sha256_file(destination)
        if actual != expected:
            raise ValueError(f"Existing {destination} has the wrong checksum. It may still be downloading; do not overwrite an active download. Expected {expected}, got {actual}.")
        return {"name": artifact["name"], "path": str(destination), "status": "already_verified", "sha256": actual}
    partial = destination.with_name(destination.name + ".part")
    lock = destination.with_name(destination.name + ".download.lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Download lock exists at {lock}; another setup may be running. Remove only a confirmed stale lock.") from error
    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        offset = partial.stat().st_size if partial.exists() else 0
        if offset and sha256_file(partial) == expected:
            partial.replace(destination)
            return {"name": artifact["name"], "path": str(destination), "status": "verified_partial", "sha256": expected}
        request = urllib.request.Request(artifact["url"], headers={"User-Agent": "checkin-prototype/1.0", **({"Range": f"bytes={offset}-"} if offset else {})})
        print(f"Downloading {artifact['name']}" + (f" (resuming at {offset:,} bytes)" if offset else ""), flush=True)
        with urllib.request.urlopen(request, timeout=60) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Server returned an inconsistent resume range.")
            # A server may ignore Range and return 200: restart the partial file.
            with partial.open("ab" if resumed else "wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        actual = sha256_file(partial)
        if actual != expected:
            raise ValueError(f"Downloaded {artifact['name']} failed SHA256 verification. Expected {expected}, got {actual}. Partial file retained for inspection.")
        if destination.exists():
            raise RuntimeError(f"Destination appeared during download: {destination}; refusing to overwrite it.")
        partial.replace(destination)
        return {"name": artifact["name"], "path": str(destination), "status": "downloaded_verified", "sha256": actual}
    finally:
        lock.unlink(missing_ok=True)


def _extract_runtime(archive: Path, target: Path) -> None:
    """Flatten trusted native releases while rejecting unsafe archive members."""
    target.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            normalized = member.filename.replace("\\", "/")
            relative = PurePosixPath(normalized)
            if relative.is_absolute() or ".." in relative.parts or ":" in normalized or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe native runtime archive member: {member.filename}")
            if member.is_dir():
                continue
            name = relative.name
            if name.casefold() in seen:
                raise ValueError(f"Duplicate flattened runtime filename: {name}")
            seen.add(name.casefold())
            destination = target / name
            temporary = target / (name + ".extracting")
            with bundle.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if destination.exists() and sha256_file(destination) == sha256_file(temporary):
                temporary.unlink()
            else:
                temporary.replace(destination)


def download_all(home: str | Path, include_meld: bool = True) -> dict:
    """Download setup artifacts, verify content, then unpack native binaries."""
    home = settings.runtime_home(home)
    records = [_download(artifact, home) for artifact in artifacts(include_meld)]
    for local_name, _, _ in LLAMA_ARCHIVES:
        _extract_runtime(home / "downloads" / local_name, home / "vendor/llama")
    if not (home / "vendor/llama/llama-server.exe").is_file():
        raise RuntimeError("Native runtime extraction did not produce llama-server.exe.")
    # Setup can be rerun against either the shipped stack or older checkpoints.
    from .audit import runtime_inventory
    manifest, _ = runtime_inventory(home)
    (home / "manifests/models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"status": "verified", "artifacts": records, "runtime": str(home / "vendor/llama")}
    (home / "reports/downloads.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
