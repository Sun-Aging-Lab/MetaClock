"""Validate all 33 final generalization PDF/SVG pairs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from generalization.array.datasets import DATASETS as ARRAY_DATASETS
from generalization.modeling import atomic_write_json, project_path
from generalization.species.species_config import DISPLAY_NAMES, SPECIES
from generalization.tissue.datasets import DATASETS as TISSUE_DATASETS

if TYPE_CHECKING:
    import pymupdf

POINTS_PER_MM = 72.0 / 25.4
SIZE_TOLERANCE_PT = 0.02
EXPECTED_PAIR_COUNT = 90


@dataclass(frozen=True)
class ExpectedPanel:
    """Validation contract for one PDF/SVG stem."""

    stem: str
    group: str
    title: str | None
    mean: bool = False
    universal: bool = False
    point_color: str | None = None
    curve_color: str | None = None
    shade_color: str | None = None


def expected_panels() -> list[ExpectedPanel]:
    """Return the exact 33-panel final-output contract."""
    panels: list[ExpectedPanel] = []
    for name, spec in {**ARRAY_DATASETS, **TISSUE_DATASETS}.items():
        base = f"{name}_fewshot_clock_mi_svm"
        panels.extend(
            (
                ExpectedPanel(base, "generalization", spec.title),
                ExpectedPanel(
                    f"{base}_with_mean_baseline", "generalization", spec.title, mean=True
                ),
            )
        )
    for species in SPECIES:
        base = f"{species}_fewshot_clock"
        panels.extend(
            (
                ExpectedPanel(base, "generalization", DISPLAY_NAMES[species]),
                ExpectedPanel(
                    f"{base}_with_mean_and_universal_baselines",
                    "generalization",
                    DISPLAY_NAMES[species],
                    mean=True,
                    universal=True,
                ),
            )
        )
    panels.extend(
        (
            ExpectedPanel(
                "component_en_coef_vs_molecular_evolutionary_distance",
                "evolution",
                None,
                point_color="#839a8e",
                curve_color="#557264",
                shade_color="#c8d4cd",
            ),
            ExpectedPanel(
                "component_en_coef_vs_max_lifespan_difference",
                "evolution",
                None,
                point_color="#7895a1",
                curve_color="#466b7a",
                shade_color="#c7d3d8",
            ),
            ExpectedPanel(
                "component_en_coef_vs_gestation_difference",
                "evolution",
                None,
                point_color="#ac8782",
                curve_color="#8a625e",
                shade_color="#d8c4c1",
            ),
        )
    )
    if len(panels) != 33 or len({panel.stem for panel in panels}) != 33:
        raise RuntimeError("Expected exactly 33 unique generalization panels")
    return panels


def _pdf_spans(page: pymupdf.Page) -> list[dict[str, object]]:
    return [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]


def validate_pdf(path: Path, panel: ExpectedPanel) -> dict[str, object]:
    """Validate canvas, embedded font family, font sizes, and text bounds."""
    import pymupdf

    with pymupdf.open(path) as document:
        if len(document) != 1:
            raise ValueError(f"{path}: expected one page")
        page = document[0]
        expected_size = (90 * POINTS_PER_MM, 60 * POINTS_PER_MM)
        observed = (float(page.rect.width), float(page.rect.height))
        if any(abs(a - b) > SIZE_TOLERANCE_PT for a, b in zip(observed, expected_size)):
            raise ValueError(f"{path}: canvas differs from 90 x 60 mm")
        spans = _pdf_spans(page)
        unexpected = sorted(
            {
                str(span["font"])
                for span in spans
                if not str(span["font"]).replace("-", "").startswith("LiberationSans")
            }
        )
        if unexpected:
            raise ValueError(f"{path}: unexpected fonts {unexpected}")
        expected_fonts = (
            {8.0, 9.0, 9.75} if panel.group == "evolution" else {6.0, 8.0, 9.5, 9.75, 10.5}
        )
        observed_fonts = {round(float(span["size"]), 2) for span in spans}
        if observed_fonts != expected_fonts:
            raise ValueError(f"{path}: font sizes {observed_fonts} != {expected_fonts}")
        outside = [
            str(span["text"])
            for span in spans
            if not page.rect.contains(pymupdf.Rect(span["bbox"]))
        ]
        if outside:
            raise ValueError(f"{path}: text outside page {outside}")
        if panel.group == "generalization":
            title_span = next((span for span in spans if str(span["text"]) == panel.title), None)
            if title_span is None or not int(title_span["flags"]) & 16:
                raise ValueError(f"{path}: title is missing or not bold")
            expected_italic = panel.stem.startswith(tuple(SPECIES))
            if bool(int(title_span["flags"]) & 2) != expected_italic:
                raise ValueError(f"{path}: title italic state differs")
            for subplot in ("SVM", "EN"):
                title = next(
                    (
                        span
                        for span in spans
                        if str(span["text"]) == subplot and round(float(span["size"]), 2) == 9.5
                    ),
                    None,
                )
                if title is None or int(title["flags"]) & 16:
                    raise ValueError(f"{path}: {subplot} subplot title differs")
            texts = [str(span["text"]) for span in spans]
            if ("Mean" in texts) != panel.mean:
                raise ValueError(f"{path}: Mean legend state differs")
            if ("Universal Clock" in texts) != panel.universal:
                raise ValueError(f"{path}: Universal Clock legend state differs")
    return {"canvas_mm": [90.0, 60.0], "font_sizes_pt": sorted(observed_fonts)}


def validate_svg(path: Path, panel: ExpectedPanel) -> dict[str, object]:
    """Validate SVG geometry, required text, colors, and evolution marks."""
    svg = path.read_text(encoding="utf-8")
    size = re.search(r'<svg[^>]+width="([0-9.]+)pt"[^>]+height="([0-9.]+)pt"', svg)
    expected_size = (90 * POINTS_PER_MM, 60 * POINTS_PER_MM)
    if size is None or any(
        abs(float(value) - expected) > SIZE_TOLERANCE_PT
        for value, expected in zip(size.groups(), expected_size)
    ):
        raise ValueError(f"{path}: SVG canvas differs from 90 x 60 mm")
    lower = svg.lower()
    if panel.group == "generalization":
        if svg.count("># Training Samples</text>") != 2:
            raise ValueError(f"{path}: expected two training-sample labels")
        if svg.count(">Mean</text>") != (2 if panel.mean else 0):
            raise ValueError(f"{path}: expected Mean in both legends")
        if svg.count(">Universal Clock</text>") != (2 if panel.universal else 0):
            raise ValueError(f"{path}: Universal Clock legend state differs")
        if panel.title is None or f">{panel.title}</text>" not in svg:
            raise ValueError(f"{path}: expected title is missing")
        title_match = re.search(rf'<text style="([^"]+)"[^>]*>{re.escape(panel.title)}</text>', svg)
        if title_match is None or "font-weight: 700" not in title_match.group(1):
            raise ValueError(f"{path}: title is not bold")
        expected_italic = panel.stem.startswith(tuple(SPECIES))
        if ("font-style: italic" in title_match.group(1)) != expected_italic:
            raise ValueError(f"{path}: title italic state differs")
        if "font-family: 'Liberation Sans'" not in title_match.group(1):
            raise ValueError(f"{path}: title font differs")
        colors = ["#7a7a7a", "#9c89b8"]
        if panel.mean:
            colors.append("#9ec5e5")
        if panel.universal:
            colors.append("#efc28b")
        for color in colors:
            if color not in lower:
                raise ValueError(f"{path}: missing {color}")
        if not panel.mean and "#9ec5e5" in lower:
            raise ValueError(f"{path}: unexpected Mean baseline color")
        if not panel.universal and "#efc28b" in lower:
            raise ValueError(f"{path}: unexpected Universal Clock baseline color")
    else:
        for color in (panel.point_color, panel.curve_color, panel.shade_color):
            if color not in lower:
                raise ValueError(f"{path}: missing {color}")
        styles = re.findall(r'style="([^"]+)"', lower)
        point_styles = [style for style in styles if f"fill: {panel.point_color}" in style]
        if len(point_styles) != EXPECTED_PAIR_COUNT or any(
            "stroke:" in style for style in point_styles
        ):
            raise ValueError(f"{path}: expected 90 outline-free points")
        marker = re.search(r'<g id="PathCollection_1">\s*<defs>\s*<path[^>]+d="M 0 ([0-9.]+)', svg)
        if marker is None or not math.isclose(
            float(marker.group(1)), math.sqrt(8) / 2, abs_tol=1e-5
        ):
            raise ValueError(f"{path}: marker radius differs")
        shade = [style for style in styles if f"fill: {panel.shade_color}" in style]
        if len(shade) != 1 or "stroke:" in shade[0]:
            raise ValueError(f"{path}: confidence shade differs")
        curves = [style for style in styles if f"stroke: {panel.curve_color}" in style]
        if len(curves) != 1:
            raise ValueError(f"{path}: trend curve differs")
        width = re.search(r"stroke-width: ([0-9.]+)", curves[0])
        observed_width = 1.0 if width is None else float(width.group(1))
        if not math.isclose(observed_width, 1.0, abs_tol=1e-12):
            raise ValueError(f"{path}: trend linewidth differs")
    return {"required_text_and_colors": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/figure3/generalization"))
    parser.add_argument("--stem", action="append", help="Validate selected stems only")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    requested = set(args.stem or [])
    panels = [panel for panel in expected_panels() if not requested or panel.stem in requested]
    if requested - {panel.stem for panel in panels}:
        raise ValueError(f"Unknown requested stems: {sorted(requested)}")
    output_root = project_path(args.output_dir)
    results: dict[str, object] = {}
    for panel in panels:
        group_dir = (
            "species"
            if panel.stem.startswith(tuple(SPECIES)) or panel.group == "evolution"
            else ("array" if panel.stem.startswith(tuple(ARRAY_DATASETS)) else "tissue")
        )
        stem = output_root / group_dir / panel.stem
        pdf, svg = stem.with_suffix(".pdf"), stem.with_suffix(".svg")
        if not pdf.is_file() or not svg.is_file():
            raise FileNotFoundError(f"Missing PDF/SVG pair: {stem}")
        results[panel.stem] = {
            "pdf": validate_pdf(pdf, panel),
            "svg": validate_svg(svg, panel),
        }
    payload = {"validated_panels": len(results), "panels": results}
    if args.summary is not None:
        atomic_write_json(payload, project_path(args.summary))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
