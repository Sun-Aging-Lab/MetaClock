#!/usr/bin/env python3
"""Validate the intervention volcano PDF/SVG pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = (
    PROJECT_ROOT / "results/intervention/intervention_fdr_volcano.pdf"
)  # fill in your path
DEFAULT_SVG = (
    PROJECT_ROOT / "results/intervention/intervention_fdr_volcano.svg"
)  # fill in your path
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT / "datasets/intervention/volcano_annotations.csv"
)  # fill in your path
DEFAULT_SUMMARY = PROJECT_ROOT / "results/intervention/volcano_validation.json"
EXPECTED_MM = (110.0, 60.0)


def validate_pdf(path: Path, labels: set[str]) -> dict[str, object]:
    """Check canvas geometry, required labels, and curated annotations."""
    import fitz

    with fitz.open(path) as document:
        if len(document) != 1:
            raise ValueError("Volcano PDF must contain exactly one page.")
        page = document[0]
        observed_mm = (
            page.rect.width * 25.4 / 72.0,
            page.rect.height * 25.4 / 72.0,
        )
        if any(abs(a - b) > 0.02 for a, b in zip(observed_mm, EXPECTED_MM, strict=True)):
            raise ValueError(f"Unexpected PDF canvas: {observed_mm} mm")
        text = page.get_text()
        required = {"Δ MetaClock Prediction", "-log10(FDR)", *labels}
        missing = sorted(value for value in required if value not in text)
        if missing:
            raise ValueError(f"PDF is missing text: {missing}")
        if "-1.0" in text:
            raise ValueError("The below-zero plotting range must not display a -1.0 tick.")
    return {"canvas_mm": list(observed_mm), "annotation_labels": sorted(labels)}


def validate_svg(path: Path, labels: set[str]) -> dict[str, object]:
    """Check that the editable SVG contains the same required text."""
    svg = path.read_text(encoding="utf-8")
    missing = sorted(
        value for value in {"Δ MetaClock Prediction", "-log10(FDR)", *labels} if value not in svg
    )
    if missing:
        raise ValueError(f"SVG is missing text: {missing}")
    return {"annotation_labels": sorted(labels), "text_remains_editable": "<text" in svg}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = set(pd.read_csv(args.annotations)["short_label"].astype(str))
    payload = {
        "status": "passed",
        "pdf": validate_pdf(args.pdf, labels),
        "svg": validate_svg(args.svg, labels),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
