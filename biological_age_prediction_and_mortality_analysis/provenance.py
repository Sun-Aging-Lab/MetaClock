"""Small, reusable provenance helpers for normalized analysis inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Describe one immutable workflow input."""
    resolved = path.resolve(strict=True)
    return {"path": str(path), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def software_versions(packages: Iterable[str]) -> dict[str, str]:
    """Record Python and requested installed distribution versions."""
    versions = {"python": platform.python_version()}
    for package in packages:
        versions[package] = importlib.metadata.version(package)
    return versions


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
