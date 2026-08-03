#!/usr/bin/env python3
"""Run lightweight structural checks for the refactored figure project."""

from __future__ import annotations

import ast
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES = ("fig1", "fig2", "fig3", "fig4")
FORBIDDEN_PATH_FRAGMENTS = ("/orcd/", "meta_clock_bio")


def validate_python_file(path: Path) -> list[str]:
    """Return validation errors for one Python source file."""
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{path}: syntax error: {exc}")
    for fragment in FORBIDDEN_PATH_FRAGMENTS:
        if fragment in source:
            errors.append(f"{path}: machine-specific path fragment {fragment!r}")
    return errors


def validate_manifest(path: Path, table_root: Path | None = None) -> list[str]:
    """Check that one result manifest is ordered and references existing files."""
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        return [f"{path}: manifest is empty"]
    required = {"figure", "panel_order", "table", "description"}
    if not required.issubset(rows[0]):
        return [f"{path}: expected columns {sorted(required)}"]
    orders = [(row["figure"], row["panel_order"]) for row in rows]
    if orders != sorted(orders):
        errors.append(f"{path}: figure/panel_order is not sorted")
    table_root = table_root or path.parent
    for row in rows:
        table = table_root / row["table"]
        if not table.is_file():
            errors.append(f"{path}: missing table {row['table']}")
    return errors


def validate_project(root: Path = PROJECT_ROOT) -> list[str]:
    """Validate figure structure, Python syntax, relative paths, and manifests."""
    errors: list[str] = []
    for figure in FIGURES:
        figure_dir = root / figure
        for required in ("README.md", "scripts", "datasets", "results"):
            if not (figure_dir / required).exists():
                errors.append(f"{figure_dir}: missing {required}")
        for script in sorted((figure_dir / "scripts").glob("*.py")):
            errors.extend(validate_python_file(script))
        manifest = figure_dir / "results" / "manifest.tsv"
        if manifest.is_file():
            errors.extend(validate_manifest(manifest))
        else:
            errors.append(f"{figure_dir}: missing results/manifest.tsv")
    root_manifest = root / "results_manifest.tsv"
    if root_manifest.is_file():
        errors.extend(validate_manifest(root_manifest, table_root=root))
    else:
        errors.append(f"{root}: missing results_manifest.tsv")
    return errors


def main() -> int:
    """Print all validation findings and return a shell-friendly status."""
    errors = validate_project()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("OK: figure directories, Python syntax, paths, and manifests validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
