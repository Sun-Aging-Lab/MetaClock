#!/usr/bin/env python3
"""Plot external intervention or cohort screening results."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from intervention_analysis.paths import file_sha256, plot_input, portable_reference, relative_path, repo_path
from intervention_analysis.volcano import CUTOFF, LABELS, render_volcano

FDR_COLUMN = "bh_q_value_after_dataset_exclusion"
COMBINED_FDR_COLUMN = "combined_bh_q_value"
EFFECT_COLUMN = "mean_diff_experiment_minus_control"
EXTERNAL_NAMES = {
    "curated_6ac8603ac6e44a0e": "Cytarabine (AraC)",
    "curated_1fa2b9d81d6ea894": "Fumaric acid esters",
    "GSE191297 (6 weeks)": "Anti-TNF therapy 1b",
    **{key.split("::", 1)[-1]: entry[0] for key, entry in LABELS["external"].items()},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=plot_input, required=True)
    parser.add_argument("--selection-audit", type=plot_input, help="Selected GEO comparison names.")
    parser.add_argument("--output-stem", type=repo_path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--combined-geo-translage", action="store_true")
    source.add_argument("--cohort", choices=("fhs", "whi"), help="Plot a cohort Welch/BH results table.")
    parser.add_argument("--axis-reference-results", type=plot_input, help="GEO results to include in the axis range.")
    parser.add_argument("--audit-json", type=repo_path)
    return parser.parse_args()


def validate_and_annotate(results_path: Path, audit_path: Path | None, *,
                          combined_geo_translage: bool = False, cohort: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(results_path, float_precision="round_trip")
    if cohort:
        frame[EFFECT_COLUMN] = frame["effect_years"]
        frame[FDR_COLUMN] = frame["bh_q_value"]
        frame["intervention_label"] = frame["plot_label"] if "plot_label" in frame else frame["label"]
    elif combined_geo_translage:
        required = {"family_test_id", "source", "source_dataset_id", "public_id", "comparison_id",
                    "intervention", COMBINED_FDR_COLUMN, EFFECT_COLUMN}
        if missing := required.difference(frame.columns):
            raise KeyError(f"Combined results are missing columns: {sorted(missing)}")
        frame["comparison_id"] = frame["family_test_id"]
        frame["gse_id"] = frame["source_dataset_id"]
        frame["intervention_label"] = frame["intervention"]
        frame[FDR_COLUMN] = frame[COMBINED_FDR_COLUMN]
    else:
        if missing := {"comparison_id", "gse_id", EFFECT_COLUMN, FDR_COLUMN}.difference(frame.columns):
            raise KeyError(f"GEO results are missing columns: {sorted(missing)}")
        if audit_path is None:
            raise ValueError("--selection-audit is required for GEO-only results")
        audit = pd.read_csv(audit_path, usecols=["comparison_id", "intervention_label", "selected"])
        chosen = audit["selected"].astype(str).str.lower().eq("true")
        frame = frame.merge(audit.loc[chosen, ["comparison_id", "intervention_label"]], on="comparison_id",
                            how="left", validate="one_to_one")
    if frame.empty or frame["comparison_id"].isna().any() or not frame["comparison_id"].is_unique:
        raise ValueError("Expected one row per comparison ID.")
    if frame["intervention_label"].isna().any():
        raise ValueError("Missing intervention names.")
    frame[EFFECT_COLUMN] = pd.to_numeric(frame[EFFECT_COLUMN], errors="raise")
    frame[FDR_COLUMN] = pd.to_numeric(frame[FDR_COLUMN], errors="raise")
    if not np.isfinite(frame[EFFECT_COLUMN]).all() or not frame[FDR_COLUMN].between(0, 1, inclusive="right").all():
        raise ValueError("Effects must be finite and FDR values must be in (0, 1].")
    frame["minus_log10_fdr"] = -np.log10(frame[FDR_COLUMN])
    frame["fdr_category"] = np.where(frame[FDR_COLUMN].le(CUTOFF), "FDR ≤ 5%", "FDR > 5%")
    frame["short_intervention_label"] = frame["intervention_label"]
    if not cohort:
        names = frame["comparison_id"].str.split("::").str[-1].map(EXTERNAL_NAMES)
        frame["short_intervention_label"] = names.fillna(frame["intervention_label"])
    return frame


def draw_volcano(frame: pd.DataFrame, output_stem: Path, *, label_layout=None,
                 axis_reference: pd.DataFrame | None = None, panel: str = "external") -> dict:
    data = frame[["comparison_id", EFFECT_COLUMN, FDR_COLUMN, "short_intervention_label"]].copy()
    data.columns = ["id", "effect", "fdr", "name"]
    reference = None
    if axis_reference is not None:
        reference = axis_reference[[EFFECT_COLUMN, "minus_log10_fdr"]].rename(
            columns={EFFECT_COLUMN: "effect", "minus_log10_fdr": "height"})
    return render_volcano(data, output_stem, panel, reference=reference, offsets=label_layout)


def plot_results(results: Path, output_stem: Path, *, cohort: str | None = None,
                 combined_geo_translage: bool = False, selection_audit: Path | None = None,
                 axis_reference_results: Path | None = None, audit_json: Path | None = None) -> None:
    """Draw one intervention screen and record the source of its adjusted P values."""
    frame = validate_and_annotate(results, selection_audit,
                                  combined_geo_translage=combined_geo_translage, cohort=cohort)
    reference = (validate_and_annotate(axis_reference_results, selection_audit)
                 if axis_reference_results is not None else None)
    audit = draw_volcano(frame, output_stem, axis_reference=reference, panel=cohort or "external")
    audit_path = audit_json or output_stem.with_name(f"{output_stem.name}_audit.json")
    audit.update({
        "python": platform.python_version(), "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__, "numpy": np.__version__, "cohort": cohort,
        "results": portable_reference(results), "combined_geo_translage": combined_geo_translage,
        "results_sha256": file_sha256(results),
        "selection_audit": portable_reference(selection_audit) if selection_audit else None,
        "axis_reference_results": portable_reference(axis_reference_results) if axis_reference_results else None,
        "n_interventions": len(frame), "fdr_category_counts": frame["fdr_category"].value_counts().to_dict(),
        "output_pdf": relative_path(output_stem.with_suffix(".pdf")),
        "output_svg": relative_path(output_stem.with_suffix(".svg")), "audit_json": relative_path(audit_path),
    })
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(audit)


def main() -> int:
    plot_results(**vars(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
