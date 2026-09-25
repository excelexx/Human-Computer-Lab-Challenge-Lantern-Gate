"""Install supplied trained heads without replacing differing local checkpoints.

Uses only the Python standard library. It copies serialized bytes and never
unpickles weights. The inference loader validates checkpoint/model compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


STAGES = ("vision", "text", "fusion")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(source_dir: Path, artifacts_root: Path) -> dict:
    source_dir = source_dir.expanduser().resolve()
    destination_dir = artifacts_root.expanduser().resolve() / "checkpoints"
    plans = []
    # Check every conflict before copying any file.
    for stage in STAGES:
        source = source_dir / f"{stage}.pt"
        target = destination_dir / source.name
        if not source.is_file():
            raise FileNotFoundError(f"Supplied trained head is missing: {source}. Obtain the trained-heads handoff or run the MELD training steps.")
        digest = sha256(source)
        if target.exists() and (not target.is_file() or sha256(target) != digest):
            raise FileExistsError(f"Existing checkpoint differs: {target}. No differing checkpoint was overwritten; choose another artifacts directory.")
        plans.append((source, target, digest))

    destination_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source, target, digest in plans:
        status = "already_identical"
        if not target.exists():
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".installing", dir=destination_dir)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as output, source.open("rb") as handle:
                    shutil.copyfileobj(handle, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if sha256(temporary) != digest:
                    raise ValueError(f"The source changed during installation: {source}")
                # The fully written file becomes visible atomically. Unlike
                # replace/Unix rename, link fails if the target already exists.
                try:
                    os.link(temporary, target)
                    status = "installed"
                except FileExistsError:
                    if not target.is_file() or sha256(target) != digest:
                        raise FileExistsError(f"A differing checkpoint appeared during installation: {target}")
                except OSError as error:
                    raise OSError(f"Atomic head installation requires a filesystem supporting hard links (for example NTFS): {destination_dir}") from error
            finally:
                temporary.unlink(missing_ok=True)
        elif not target.is_file() or sha256(target) != digest:
            raise FileExistsError(f"Checkpoint changed after preflight: {target}")
        records.append({"stage": target.stem, "path": str(target), "sha256": digest, "status": status})
    return {"artifacts_root": str(destination_dir.parent), "heads": records, "note": "Bytes copied and verified. Model compatibility is checked when the local pipeline loads."}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo / "trained-heads")
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("CHECKIN_HOME") or repo / ".artifacts"))
    args = parser.parse_args()
    print(json.dumps(install(args.source, args.home), indent=2))


if __name__ == "__main__":
    main()
