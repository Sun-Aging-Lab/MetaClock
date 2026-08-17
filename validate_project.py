#!/usr/bin/env python3
"""Validate the portable MetaClock analysis source tree."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_DIRECTORIES = (
    "chronological_age_prediction_performance",
    "chronological_age_prediction_performance/tests",
    "generalization/array",
    "generalization/tissue",
    "generalization/species",
    "generalization/tests",
    "biological_age_prediction_and_mortality_analysis",
    "biological_age_prediction_and_mortality_analysis/tests",
    "intervention_analysis",
    "intervention_analysis/tests",
)
MACHINE_SPECIFIC_PATH = re.compile(r"(?<![A-Za-z0-9_])/(?:orcd|home|Users)/")
NUMBERED_FIGURE_REFERENCE = re.compile(r"\bfig(?:ure)?\s*[2-5]\b", re.IGNORECASE)
PLACEHOLDER_PATH = re.compile(r"Path\(\s*['\"]/path/to/your/file['\"]\s*\)")
RELATIVE_EXTERNAL_PATH = re.compile(
    r"(?:Path\(\s*['\"](?:data|datasets)/|PROJECT_ROOT\s*/\s*['\"](?:data|datasets)/)"
)


def validate_structure() -> list[str]:
    """Return errors for missing project-level files or semantic directories."""
    errors: list[str] = []
    for relative in ("README.md", "pyproject.toml", "uv.lock", ".python-version"):
        if not (ROOT / relative).is_file():
            errors.append(f"missing file: {relative}")
    for relative in REQUIRED_DIRECTORIES:
        if not (ROOT / relative).is_dir():
            errors.append(f"missing directory: {relative}")
    return errors


def validate_sources() -> tuple[list[str], int, int]:
    """Parse Python and inspect Python/R sources for machine-specific paths."""
    errors: list[str] = []
    python_count = 0
    for path in sorted(ROOT.rglob("*.py")):
        python_count += 1
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        try:
            tree = ast.parse(text, filename=str(relative))
        except SyntaxError as error:
            errors.append(f"syntax error in {relative}: {error}")
            tree = None
        if path != Path(__file__) and MACHINE_SPECIFIC_PATH.search(text):
            errors.append(f"machine-specific absolute path in {relative}")
        if tree is not None and path != Path(__file__):
            lines = text.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                segment = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                marked = "# fill in your path" in segment
                if marked and not PLACEHOLDER_PATH.search(segment):
                    errors.append(
                        f"input placeholder is not exactly /path/to/your/file in "
                        f"{relative}:{node.lineno}"
                    )
                if not marked and RELATIVE_EXTERNAL_PATH.search(segment):
                    errors.append(f"relative external input in {relative}:{node.lineno}")
    r_count = 0
    for path in sorted(ROOT.rglob("*.R")):
        r_count += 1
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        if MACHINE_SPECIFIC_PATH.search(text):
            errors.append(f"machine-specific absolute path in {relative}")
        for line_number, line in enumerate(text.splitlines(), 1):
            if "# fill in your path" in line and "/path/to/your/file" not in line:
                errors.append(
                    f"R input placeholder is not exactly /path/to/your/file in "
                    f"{relative}:{line_number}"
                )
    return errors, python_count, r_count


def validate_documentation() -> list[str]:
    """Require content-oriented README wording without manuscript numbering."""
    errors: list[str] = []
    for path in sorted(ROOT.rglob("README.md")):
        relative = path.relative_to(ROOT)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if NUMBERED_FIGURE_REFERENCE.search(line):
                errors.append(
                    f"numbered figure reference in {relative}:{line_number}: {line.strip()}"
                )
    return errors


def main() -> int:
    """Run all lightweight checks and print an auditable summary."""
    errors = validate_structure()
    source_errors, python_count, r_count = validate_sources()
    errors.extend(source_errors)
    errors.extend(validate_documentation())
    if errors:
        print("Project validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Project validation passed: {python_count} Python and {r_count} R source files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
