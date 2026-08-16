#!/usr/bin/env python3
"""Validate model provenance, source schemas, and biological-age/mortality vectors."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import EXTERNAL_MODELS, FIGURE4, FIGURE4_MODELS, SURVIVAL_MODELS
    from .provenance import validate_all672_model_directory
except ImportError:  # Direct-script fallback.
    from config import EXTERNAL_MODELS, FIGURE4, FIGURE4_MODELS, SURVIVAL_MODELS
    from provenance import validate_all672_model_directory


DEFAULT_FIGURES = Path("results/figure4/figures")  # fill in your path
DEFAULT_SOURCE = Path("results/figure4/source_data")  # fill in your path
DEFAULT_MODEL_DIR = Path("models/figure4/all672_without_dnamfitage")  # fill in your path
POINTS_PER_MM = 72 / 25.4
EXPECTED_MM = {
    "bioage_prediction_mae_pearson": (60, 50),
    "duplicate_sample_reliability": (60, 50),
    "external_disease_heatmap": (90, 60),
    "cox_coefficients_standardized": (90, 50),
    "kaplan_meier_survival": (100, 50),
}


def require_finite(table: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Require numeric, finite values in selected columns."""
    values = table.loc[:, list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values.")


def validate_vector_pair(directory: Path, stem: str, size_mm: tuple[int, int]) -> dict[str, object]:
    """Check paired one-page vectors, exact canvas, font, and global exclusion."""
    import fitz

    pdf, svg = directory / f"{stem}.pdf", directory / f"{stem}.svg"
    if not pdf.is_file() or not svg.is_file():
        raise FileNotFoundError(f"Missing PDF/SVG pair for {stem}.")
    document = fitz.open(pdf)
    if len(document) != 1:
        raise ValueError(f"{pdf} must contain exactly one page.")
    page = document[0]
    observed = (page.rect.width, page.rect.height)
    expected = tuple(value * POINTS_PER_MM for value in size_mm)
    if any(abs(left - right) > 0.02 for left, right in zip(observed, expected, strict=True)):
        raise ValueError(f"Unexpected canvas for {pdf}: {observed}; expected {expected}.")
    svg_text, pdf_text = svg.read_text(encoding="utf-8"), page.get_text()
    if "dnamfitage" in (svg_text + pdf_text).lower():
        raise ValueError(f"Excluded component appears in {stem}.")
    if "Liberation Sans" not in svg_text:
        raise ValueError(f"{svg} does not preserve the required Liberation Sans font family.")
    if not re.search(r"font-size:\s*(?:6|8|9|9\.5|9\.75)px", svg_text):
        raise ValueError(f"{svg} does not expose an expected manuscript font size.")
    return {"size_mm": size_mm, "pdf_bytes": pdf.stat().st_size, "svg_bytes": svg.stat().st_size}


def validate_sources(source_dir: Path, figure_dir: Path) -> dict[str, int]:
    """Validate the scientific schema and expected sample/model counts of every panel."""
    participants, pairs = int(FIGURE4["participants"]), int(FIGURE4["duplicate_pairs"])
    prediction = pd.read_csv(source_dir / "bioage_prediction_metrics_source_data.csv")
    reliability = pd.read_csv(source_dir / "duplicate_sample_reliability_source_data.csv")
    if tuple(prediction["model"]) != FIGURE4_MODELS or not prediction["n"].eq(participants).all():
        raise ValueError("Prediction source must be ordered BioMeta/Baseline with n=672.")
    if tuple(reliability["model"]) != FIGURE4_MODELS or not reliability["n_pairs"].eq(pairs).all():
        raise ValueError("Reliability source must be ordered BioMeta/Baseline with 44 pairs.")
    if any("icc" in column.lower() for column in reliability.columns):
        raise ValueError("ICC is not part of the formal replicate-reliability statistic.")
    require_finite(prediction, ("mae", "pearson_r"), "Prediction source")
    require_finite(reliability, ("mean_absolute_delta",), "Reliability source")

    external = pd.read_csv(figure_dir / "external_disease_heatmap_source_data.csv")
    if external.duplicated(["dataset", "disease", "model"]).any() or set(external["model"]) != set(
        EXTERNAL_MODELS
    ):
        raise ValueError("External heatmap source has duplicate keys or unexpected models.")
    require_finite(
        external, ("mean_difference_disease_minus_control", "bonferroni_p_value"), "External source"
    )
    if not external["bonferroni_p_value"].between(0, 1).all():
        raise ValueError("External corrected P values must be in [0, 1].")

    cox = pd.read_csv(source_dir / "standardized_cox_effects_source_data.csv")
    if (
        len(cox) != 9
        or cox.duplicated(["panel", "model"]).any()
        or set(cox["model"]) != set(SURVIVAL_MODELS)
    ):
        raise ValueError("Cox source must contain the nine unique formal panel/model estimates.")
    require_finite(cox, ("coefficient", "ci_lower", "ci_upper", "p_value"), "Cox source")
    if not (cox["ci_lower"].le(cox["coefficient"]) & cox["coefficient"].le(cox["ci_upper"])).all():
        raise ValueError("A Cox confidence interval does not contain its coefficient.")

    curves = pd.read_csv(source_dir / "kaplan_meier_curves_source_data.csv")
    tests = pd.read_csv(source_dir / "kaplan_meier_logrank_source_data.csv")
    curve_columns = {
        "grouping",
        "model",
        "bioage_group",
        "time",
        "survival",
        "ci_lower",
        "ci_upper",
        "n_at_risk",
        "n_events",
        "n_group",
        "n_events_group",
    }
    test_columns = {
        "grouping",
        "model",
        "bottom_20_n",
        "bottom_20_events",
        "top_20_n",
        "top_20_events",
        "observed_top_events",
        "expected_top_events",
        "logrank_variance",
        "logrank_chi2",
        "logrank_p_value",
    }
    if not curve_columns.issubset(curves.columns) or not test_columns.issubset(tests.columns):
        raise ValueError("Kaplan-Meier or log-rank source is missing formal audit columns.")
    tail_n = int(np.floor(participants * 0.20))
    if set(tests["model"]) != set(SURVIVAL_MODELS) or len(tests) != 3:
        raise ValueError("Log-rank source must contain exactly the three survival models.")
    if not tests[["bottom_20_n", "top_20_n"]].eq(tail_n).all().all():
        raise ValueError(f"Tail groups must use floor(672 * 0.20) = {tail_n} participants.")
    if not curves["n_group"].eq(tail_n).all() or not curves["survival"].between(0, 1).all():
        raise ValueError("KM source contains an invalid tail count or survival probability.")
    require_finite(
        curves, ("time", "survival", "ci_lower", "ci_upper", "n_at_risk", "n_events"), "KM source"
    )
    require_finite(
        tests,
        (
            "observed_top_events",
            "expected_top_events",
            "logrank_variance",
            "logrank_chi2",
            "logrank_p_value",
        ),
        "Log-rank source",
    )
    if not tests["logrank_p_value"].between(0, 1).all():
        raise ValueError("Log-rank P values must be in [0, 1].")
    return {
        "prediction_rows": len(prediction),
        "external_rows": len(external),
        "cox_rows": len(cox),
        "km_rows": len(curves),
    }


def main() -> int:
    """Run model, source, and vector validation and write one report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    training = validate_all672_model_directory(args.model_dir)
    report = {
        "model_training": {
            "participants": training["n_participants"],
            "component_features": training["models"]["bio_metaclock"]["n_features"],
            "baseline_features": training["models"]["baseline"]["n_features"],
        },
        "sources": validate_sources(args.source_dir, args.figure_dir),
        "panels": {
            stem: validate_vector_pair(args.figure_dir, stem, size)
            for stem, size in EXPECTED_MM.items()
        },
        "globally_excluded_component": "dnamfitage",
    }
    (args.figure_dir / "validation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
