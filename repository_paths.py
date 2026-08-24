"""Validated repository-local path handling shared by all workflows."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_repo_path(path: str | Path) -> Path:
    """Resolve a path and reject targets outside the repository."""
    supplied = Path(path)
    resolved = (
        supplied.resolve()
        if supplied.is_absolute()
        else (PROJECT_ROOT / supplied).resolve()
    )
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path is outside the repository: {supplied}")
    return resolved


def repo_path(value: str | Path) -> Path:
    """Argparse converter accepting only paths relative to the repository root."""
    supplied = Path(value)
    if supplied.is_absolute():
        raise argparse.ArgumentTypeError(
            "paths must be relative to the repository root"
        )
    try:
        return ensure_repo_path(supplied)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def relative_path(path: str | Path) -> str:
    """Return a stable POSIX path relative to the repository root."""
    return ensure_repo_path(path).relative_to(PROJECT_ROOT).as_posix()


def repo_child(parent: str | Path, value: str | Path) -> Path:
    """Join a manifest path below a repository directory without allowing escape."""
    resolved_parent = ensure_repo_path(parent)
    child = Path(value)
    if child.is_absolute():
        raise ValueError(f"manifest paths must be relative: {value}")
    resolved = (resolved_parent / child).resolve()
    if not resolved.is_relative_to(resolved_parent):
        raise ValueError(f"manifest path escapes its repository directory: {value}")
    return resolved
