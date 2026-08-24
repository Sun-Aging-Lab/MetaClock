"""Small, reusable provenance helpers for normalized analysis inputs."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Iterable


def file_record(path: Path) -> dict[str, Any]:
    """Describe one immutable workflow input."""
    resolved = path.resolve(strict=True)
    return {"path": str(path), "bytes": resolved.stat().st_size}


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
