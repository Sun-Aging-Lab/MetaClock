#!/usr/bin/env python3
"""Plot the post-exclusion BioMeta volcano with explicit FDR categories."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

from generalization.fonts import register_fonts
from intervention_analysis.matplotlib_runtime import MPLCONFIGDIR
from intervention_analysis.paths import relative_path, repo_path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# Explicit Okabe-Ito-derived, color-vision-accessible palette. No package defaults.
FDR_5_COLOR = "#0072B2"
FDR_20_COLOR = "#D89000"
NONSIGNIFICANT_COLOR = "#B8BDC5"
FDR_5_LINE_COLOR = "#005A8D"
FDR_20_LINE_COLOR = "#A86B00"
ZERO_LINE_COLOR = "#6B7280"
GRID_COLOR = "#E6E8EB"
TEXT_COLOR = "#202124"
FIGURE_MM = (110.0, 60.0)
MM_PER_INCH = 25.4
AXIS_TITLE_FONT_SIZE = 9.75
TICK_FONT_SIZE = 8.0
INTERVENTION_LABEL_FONT_SIZE = 7.5
OUTLINE_WIDTH_PT = 0.2

FDR_COLUMN = "bh_q_value_after_dataset_exclusion"
COMBINED_FDR_COLUMN = "combined_bh_q_value"
EFFECT_COLUMN = "mean_diff_experiment_minus_control"

# Curated short labels for the interventions passing FDR 20% in this analysis.
# These deliberately omit GSE IDs, control-arm text, p-values, and FDR values.
SHORT_INTERVENTION_LABELS = {
    "curated_1fc35799c70a1f70": "Abdominal radiotherapy",
    "curated_fe95adeead1b848b": "Dexamethasone",
    "curated_a5aa77c6d7ffdccc": "Cigarette smoking",
    "curated_6ac8603ac6e44a0e": "Cytarabine (AraC)",
    "curated_1fa2b9d81d6ea894": "FAE",
}

COMBINED_SHORT_INTERVENTION_LABELS = {
    **{f"geo::{key}": value for key, value in SHORT_INTERVENTION_LABELS.items()},
    "translage::GSE217633": "Antiretroviral therapy",
    "translage::GSE191297 (6 weeks)": "Anti-TNF therapy 1b",
}

# Point-specific offsets keep the five requested labels legible on a 110 x 60 mm
# canvas while preserving a direct connector to each observation.
LABEL_LAYOUT = {
    "curated_1fc35799c70a1f70": (-5, -1, "right", "center"),
    "curated_fe95adeead1b848b": (-5, -1, "right", "center"),
    "curated_a5aa77c6d7ffdccc": (-5, 16, "right", "bottom"),
    "curated_6ac8603ac6e44a0e": (5, 8, "left", "bottom"),
    "curated_1fa2b9d81d6ea894": (-5, 8, "right", "bottom"),
}

COMBINED_LABEL_LAYOUT = {
    **{f"geo::{key}": value for key, value in LABEL_LAYOUT.items()},
    "geo::curated_1fc35799c70a1f70": (5, -4, "left", "top"),
    "geo::curated_6ac8603ac6e44a0e": (5, 8, "left", "bottom"),
    "translage::GSE217633": (5, -4, "left", "top"),
    "translage::GSE191297 (6 weeks)": (-5, 18, "right", "bottom"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=repo_path, required=True)
    parser.add_argument("--selection-audit", type=repo_path)
    parser.add_argument("--output-stem", type=repo_path, required=True)
    parser.add_argument(
        "--combined-geo-translage",
        action="store_true",
        help="Read the audited combined GEO + TranslAGE result schema.",
    )
    parser.add_argument(
        "--axis-reference-results",
        type=repo_path,
        help="Optional GEO-only result table used only to preserve the original axes.",
    )
    parser.add_argument(
        "--audit-json",
        type=repo_path,
        help="Optional audit path; defaults beside the plot output.",
    )
    return parser.parse_args()


def validate_and_annotate(
    results_path: Path,
    audit_path: Path | None,
    *,
    combined_geo_translage: bool = False,
) -> pd.DataFrame:
    """Validate the explicitly selected result schema and assign FDR categories."""
    frame = pd.read_csv(results_path)
    if combined_geo_translage:
        required = {
            "family_test_id",
            "source",
            "source_dataset_id",
            "public_id",
            "comparison_id",
            "intervention",
            COMBINED_FDR_COLUMN,
            EFFECT_COLUMN,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Combined results are missing columns: {sorted(missing)}")
        if frame["family_test_id"].duplicated().any():
            raise ValueError("Combined results contain duplicate family_test_id values")
        frame = frame.copy()
        frame["comparison_id"] = frame["family_test_id"].astype(str)
        frame["gse_id"] = frame["source_dataset_id"].astype(str)
        frame["intervention_label"] = frame["intervention"].astype(str)
        frame[FDR_COLUMN] = frame[COMBINED_FDR_COLUMN]
        short_labels = COMBINED_SHORT_INTERVENTION_LABELS
    else:
        if audit_path is None:
            raise ValueError("--selection-audit is required for GEO-only results")
        short_labels = SHORT_INTERVENTION_LABELS
        required = {"comparison_id", "gse_id", EFFECT_COLUMN, FDR_COLUMN}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Results are missing columns: {sorted(missing)}")
        if frame["comparison_id"].duplicated().any():
            raise ValueError("Results contain duplicate comparison_id values")

        audit = pd.read_csv(audit_path)
        audit_required = {"comparison_id", "intervention_label", "selected"}
        audit_missing = audit_required.difference(audit.columns)
        if audit_missing:
            raise KeyError(
                f"Selection audit is missing columns: {sorted(audit_missing)}"
            )
        selected = audit["selected"]
        if selected.dtype != bool:
            selected = selected.astype(str).str.lower().eq("true")
        audit = audit.loc[selected, ["comparison_id", "intervention_label"]]
        if audit["comparison_id"].duplicated().any():
            raise ValueError(
                "Selection audit contains duplicate selected comparison IDs"
            )

        frame = frame.merge(
            audit, on="comparison_id", how="left", validate="one_to_one"
        )
        if frame["intervention_label"].isna().any():
            missing_ids = frame.loc[frame["intervention_label"].isna(), "comparison_id"]
            raise ValueError(f"Missing intervention labels for: {sorted(missing_ids)}")

    required = {"comparison_id", "gse_id", EFFECT_COLUMN, FDR_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Results are missing columns: {sorted(missing)}")
    if frame["comparison_id"].duplicated().any():
        raise ValueError("Results contain duplicate comparison_id values")

    frame[EFFECT_COLUMN] = pd.to_numeric(frame[EFFECT_COLUMN], errors="raise")
    frame[FDR_COLUMN] = pd.to_numeric(frame[FDR_COLUMN], errors="raise")
    if not np.isfinite(frame[EFFECT_COLUMN]).all():
        raise ValueError("All effect sizes must be finite")
    if not frame[FDR_COLUMN].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("All FDR values must be in [0, 1]")
    if frame[FDR_COLUMN].eq(0).any():
        raise ValueError("Zero FDR values cannot be transformed with -log10")

    frame["minus_log10_fdr"] = -np.log10(frame[FDR_COLUMN])
    frame["fdr_category"] = np.select(
        [frame[FDR_COLUMN].le(0.05), frame[FDR_COLUMN].le(0.20)],
        ["FDR ≤ 5%", "5% < FDR ≤ 20%"],
        default="FDR > 20%",
    )
    label_ids = set(frame.loc[frame[FDR_COLUMN].le(0.20), "comparison_id"])
    missing_short_labels = label_ids.difference(short_labels)
    extra_short_labels = set(short_labels).difference(label_ids)
    if missing_short_labels or extra_short_labels:
        raise ValueError(
            "Curated short-label mapping does not exactly match FDR <= 20% results; "
            f"missing={sorted(missing_short_labels)}, extra={sorted(extra_short_labels)}"
        )
    frame["short_intervention_label"] = frame["comparison_id"].map(short_labels)
    return frame


def configure_style() -> None:
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    register_fonts()
    plt.rcParams.update(
        {
            "font.family": ["Liberation Sans"],
            "font.sans-serif": ["Liberation Sans"],
            "axes.labelsize": AXIS_TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": 7,
            "axes.linewidth": OUTLINE_WIDTH_PT,
            "xtick.major.width": OUTLINE_WIDTH_PT,
            "xtick.minor.width": OUTLINE_WIDTH_PT,
            "ytick.major.width": OUTLINE_WIDTH_PT,
            "ytick.minor.width": OUTLINE_WIDTH_PT,
            "patch.linewidth": OUTLINE_WIDTH_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_volcano(
    frame: pd.DataFrame,
    output_stem: Path,
    *,
    label_layout: dict[str, tuple[int, int, str, str]] = LABEL_LAYOUT,
    axis_reference: pd.DataFrame | None = None,
) -> dict[str, object]:
    configure_style()
    width_in, height_in = (value / MM_PER_INCH for value in FIGURE_MM)
    figure, axis = plt.subplots(figsize=(width_in, height_in))
    figure.subplots_adjust(left=0.12, right=0.975, bottom=0.17, top=0.965)

    point_styles = [
        ("FDR > 20%", NONSIGNIFICANT_COLOR, 18, 0.72, 1),
        ("5% < FDR ≤ 20%", FDR_20_COLOR, 29, 0.95, 3),
        ("FDR ≤ 5%", FDR_5_COLOR, 32, 1.00, 4),
    ]
    for category, color, size, alpha, zorder in point_styles:
        selected = frame.loc[frame["fdr_category"].eq(category)]
        axis.scatter(
            selected[EFFECT_COLUMN],
            selected["minus_log10_fdr"],
            s=size,
            color=color,
            alpha=alpha,
            edgecolors="white" if category != "FDR > 20%" else "none",
            linewidths=OUTLINE_WIDTH_PT,
            zorder=zorder,
        )

    fdr_5_y = -math.log10(0.05)
    fdr_20_y = -math.log10(0.20)
    axis.axhline(
        fdr_5_y,
        color=FDR_5_LINE_COLOR,
        linestyle=(0, (5, 2.5)),
        linewidth=0.85,
        zorder=2,
    )
    axis.axhline(
        fdr_20_y,
        color=FDR_20_LINE_COLOR,
        linestyle=(0, (2, 2)),
        linewidth=0.85,
        zorder=2,
    )
    axis.axvline(
        0.0,
        color=ZERO_LINE_COLOR,
        linestyle=(0, (1.5, 2.5)),
        linewidth=0.65,
        zorder=1,
    )

    axis.set_xlabel("Δ MetaClock Prediction")
    axis.set_ylabel("-log10(FDR)")
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONT_SIZE,
        width=OUTLINE_WIDTH_PT,
        length=3,
    )
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.45, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(TEXT_COLOR)
    axis.spines["bottom"].set_color(TEXT_COLOR)

    axis_frame = frame if axis_reference is None else axis_reference
    x_min = float(axis_frame[EFFECT_COLUMN].min())
    x_max = float(axis_frame[EFFECT_COLUMN].max())
    x_padding = max(0.8, 0.075 * (x_max - x_min))
    axis.set_xlim(x_min - x_padding, x_max + x_padding)
    y_max = float(axis_frame["minus_log10_fdr"].max())
    axis.set_ylim(-1.0, y_max * 1.075)
    # Preserve the plotting range below zero without displaying a -1.0 tick.
    axis.set_yticks(np.arange(0.0, y_max * 1.075 + 0.01, 2.5))

    for row in frame.loc[frame[FDR_COLUMN].le(0.20)].itertuples(index=False):
        comparison_id = str(row.comparison_id)
        dx, dy, horizontal_alignment, vertical_alignment = label_layout[comparison_id]
        category = str(row.fdr_category)
        color = FDR_5_LINE_COLOR if category == "FDR ≤ 5%" else FDR_20_LINE_COLOR
        axis.annotate(
            str(row.short_intervention_label),
            xy=(float(getattr(row, EFFECT_COLUMN)), float(row.minus_log10_fdr)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=horizontal_alignment,
            va=vertical_alignment,
            fontsize=INTERVENTION_LABEL_FONT_SIZE,
            color=color,
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.45,
                "alpha": 0.82,
                "shrinkA": 1.5,
                "shrinkB": 1.5,
            },
            zorder=5,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=FDR_5_COLOR,
            markeredgecolor="white",
            markeredgewidth=OUTLINE_WIDTH_PT,
            label="FDR ≤ 5%",
            markersize=5.2,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=FDR_20_COLOR,
            markeredgecolor="white",
            markeredgewidth=OUTLINE_WIDTH_PT,
            label="5% < FDR ≤ 20%",
            markersize=5.2,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=NONSIGNIFICANT_COLOR,
            markeredgecolor="none",
            label="FDR > 20%",
            markersize=5.2,
        ),
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        handletextpad=0.45,
        labelspacing=0.35,
        borderaxespad=0.25,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "",
        "Author": "",
        "Subject": "Post-exclusion BioMeta FDR volcano plot",
        "Creator": (
            f"Python {platform.python_version()}; Matplotlib {matplotlib.__version__}; "
            f"pandas {pd.__version__}; NumPy {np.__version__}"
        ),
    }
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        facecolor="white",
        metadata=metadata,
    )
    figure.savefig(
        output_stem.with_suffix(".svg"),
        facecolor="white",
        metadata={"Title": "", "Description": metadata["Subject"]},
    )
    plt.close(figure)
    return {
        "canvas_mm": list(FIGURE_MM),
        "xlim": [x_min - x_padding, x_max + x_padding],
        "ylim": [-1.0, y_max * 1.075],
        "yticks": np.arange(0.0, y_max * 1.075 + 0.01, 2.5).tolist(),
    }


def main() -> int:
    args = parse_args()
    frame = validate_and_annotate(
        args.results,
        args.selection_audit,
        combined_geo_translage=args.combined_geo_translage,
    )
    axis_reference = None
    if args.axis_reference_results is not None:
        axis_reference = validate_and_annotate(
            args.axis_reference_results,
            args.selection_audit,
            combined_geo_translage=False,
        )
    label_layout = (
        COMBINED_LABEL_LAYOUT if args.combined_geo_translage else LABEL_LAYOUT
    )
    axis_audit = draw_volcano(
        frame,
        args.output_stem,
        label_layout=label_layout,
        axis_reference=axis_reference,
    )
    counts = frame["fdr_category"].value_counts().to_dict()
    audit = {
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        **axis_audit,
        "combined_geo_translage": args.combined_geo_translage,
        "results": relative_path(args.results),
        "selection_audit": (
            relative_path(args.selection_audit) if args.selection_audit else None
        ),
        "axis_reference_results": (
            relative_path(args.axis_reference_results)
            if args.axis_reference_results
            else None
        ),
        "n_interventions": len(frame),
        "fdr_category_counts": {str(key): int(value) for key, value in counts.items()},
        "n_labels": int(frame[FDR_COLUMN].le(0.20).sum()),
        "output_pdf": relative_path(args.output_stem.with_suffix(".pdf")),
        "output_svg": relative_path(args.output_stem.with_suffix(".svg")),
    }
    audit_path = args.audit_json or args.output_stem.with_name(
        f"{args.output_stem.name}_audit.json"
    )
    audit["audit_json"] = relative_path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
