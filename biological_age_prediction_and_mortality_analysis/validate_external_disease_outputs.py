#!/usr/bin/env python3
"""Validate the unrestricted external-disease source and vector outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXTERNAL_MODELS

DEFAULT_FIGURES = Path("results/figure4/figures")
POINTS_PER_MM = 72 / 25.4
EXPECTED_MM = {"03_external_disease_heatmap": (90, 60)}


def require_finite(table: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Require numeric, finite values in selected columns."""
    values = table.loc[:, list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values.")


def validate_vector_pair(
    directory: Path, stem: str, size_mm: tuple[int, int]
) -> dict[str, object]:
    """Check paired one-page vectors, exact canvas, font, and global exclusion."""
    import pymupdf

    pdf, svg = directory / f"{stem}.pdf", directory / f"{stem}.svg"
    if not pdf.is_file() or not svg.is_file():
        raise FileNotFoundError(f"Missing PDF/SVG pair for {stem}.")
    with pymupdf.open(pdf) as document:
        if len(document) != 1:
            raise ValueError(f"{pdf} must contain exactly one page.")
        page = document[0]
        observed = (page.rect.width, page.rect.height)
        expected = tuple(value * POINTS_PER_MM for value in size_mm)
        if any(
            abs(left - right) > 0.02
            for left, right in zip(observed, expected, strict=True)
        ):
            raise ValueError(f"Unexpected canvas for {pdf}: {observed}; expected {expected}.")
        svg_text, pdf_text = svg.read_text(encoding="utf-8"), page.get_text()
    if "dnamfitage" in (svg_text + pdf_text).lower():
        raise ValueError(f"Excluded component appears in {stem}.")
    if "Liberation Sans" not in svg_text:
        raise ValueError(f"{svg} does not preserve the required Liberation Sans font family.")
    if not re.search(r"font-size:\s*(?:6|8|9|9\.5|9\.75)px", svg_text):
        raise ValueError(f"{svg} does not expose an expected manuscript font size.")
    return {
        "size_mm": size_mm,
        "pdf_bytes": pdf.stat().st_size,
        "svg_bytes": svg.stat().st_size,
    }


def validate_source(figure_dir: Path) -> dict[str, int]:
    """Validate the external-disease source schema and model counts."""
    external = pd.read_csv(figure_dir / "03_external_disease_heatmap_source_data.csv")
    if (
        len(external) != 18
        or external[["dataset", "disease"]].drop_duplicates().shape[0] != 6
        or not external.groupby(["dataset", "disease"])["model"].nunique().eq(3).all()
        or external.duplicated(["dataset", "disease", "model"]).any()
        or set(external["model"]) != set(EXTERNAL_MODELS)
    ):
        raise ValueError("External heatmap source has duplicate keys or unexpected models.")
    require_finite(
        external,
        ("mean_difference_disease_minus_control", "bonferroni_p_value"),
        "External source",
    )
    if not external["bonferroni_p_value"].between(0, 1).all():
        raise ValueError("External corrected P values must be in [0, 1].")
    return {"external_rows": len(external)}


def main() -> int:
    """Run source and vector validation and write one report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    report = {
        "source": validate_source(args.figure_dir),
        "panel": validate_vector_pair(
            args.figure_dir,
            "03_external_disease_heatmap",
            EXPECTED_MM["03_external_disease_heatmap"],
        ),
        "globally_excluded_component": "dnamfitage",
    }
    (args.figure_dir / "validation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
