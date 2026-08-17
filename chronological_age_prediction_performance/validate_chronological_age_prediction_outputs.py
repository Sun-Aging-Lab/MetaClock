#!/usr/bin/env python3
"""Validate chronological-age prediction PDF/SVG output pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SIZE_PT = (180 / 25.4 * 72, 65 / 25.4 * 72)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/chronological_age_prediction"
EXPECTED_STEMS = (
    "scatter/pearson_mae_model_architecture",
    "scatter/pearson_mae_submodel",
    "ablation/ablation_step_01_without_zhangblup",
    "ablation/ablation_step_02_without_han",
    "ablation/ablation_step_03_without_skinandblood",
    "architecture_comparison/pearson_mae_submodel_architectures",
)


def validate_pdf(path: Path) -> None:
    """Validate a one-page PDF canvas and its embedded Liberation Sans font."""
    import pymupdf

    if not path.is_file():
        raise FileNotFoundError(path)
    with pymupdf.open(path) as document:
        if len(document) != 1:
            raise ValueError(f"{path} has {len(document)} pages; expected one")
        observed = (document[0].rect.width, document[0].rect.height)
        if any(
            abs(value - expected) > 0.01
            for value, expected in zip(observed, EXPECTED_SIZE_PT, strict=True)
        ):
            raise ValueError(f"{path} has page size {observed}, expected {EXPECTED_SIZE_PT}")
        fonts = {font[3].replace("-", "") for font in document.get_page_fonts(0)}
    if not any("LiberationSans" in font for font in fonts):
        raise ValueError(f"{path} does not embed Liberation Sans: {sorted(fonts)}")


def validate_svg(path: Path) -> None:
    """Ensure the matching SVG exists and retains editable text nodes."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if "<text" not in path.read_text(encoding="utf-8"):
        raise ValueError(f"{path} does not contain editable SVG text")


def parse_args() -> argparse.Namespace:
    """Parse the root directory containing chronological-age prediction outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    """Validate all expected PDF/SVG pairs at the documented print dimensions."""
    args = parse_args()
    for stem in EXPECTED_STEMS:
        pdf_path = args.output_root / f"{stem}.pdf"
        svg_path = args.output_root / f"{stem}.svg"
        validate_pdf(pdf_path)
        validate_svg(svg_path)
        print(f"validated {pdf_path} and {svg_path}")


if __name__ == "__main__":
    main()
