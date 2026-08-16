#!/usr/bin/env python3
"""Draw the intervention false-discovery-rate volcano plot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    PROJECT_ROOT / "results/intervention/selected/selected_intervention_results.csv"
)  # fill in your path
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT / "datasets/intervention/volcano_annotations.csv"
)  # fill in your path
DEFAULT_OUTPUT_STEM = PROJECT_ROOT / "results/intervention/intervention_fdr_volcano"
FDR_COLUMN = "bh_q_value_after_dataset_exclusion"
EFFECT_COLUMN = "mean_diff_experiment_minus_control"
FIGURE_MM = (110.0, 60.0)
FDR_5_COLOR = "#0072B2"
FDR_20_COLOR = "#D89000"
NONSIGNIFICANT_COLOR = "#B8BDC5"


def load_plot_data(results_path: Path, annotations_path: Path) -> pd.DataFrame:
    """Validate result columns and attach curated labels and point offsets."""
    frame = pd.read_csv(results_path)
    annotations = pd.read_csv(annotations_path)
    required = {"comparison_id", EFFECT_COLUMN, FDR_COLUMN}
    annotation_required = {"comparison_id", "short_label", "dx", "dy", "ha", "va"}
    if missing := required.difference(frame.columns):
        raise KeyError(f"Results are missing columns: {sorted(missing)}")
    if missing := annotation_required.difference(annotations.columns):
        raise KeyError(f"Annotation table is missing columns: {sorted(missing)}")
    if frame["comparison_id"].duplicated().any():
        raise ValueError("Results contain duplicate comparison_id values.")
    frame[EFFECT_COLUMN] = pd.to_numeric(frame[EFFECT_COLUMN], errors="raise")
    frame[FDR_COLUMN] = pd.to_numeric(frame[FDR_COLUMN], errors="raise")
    if not np.isfinite(frame[[EFFECT_COLUMN, FDR_COLUMN]].to_numpy(float)).all():
        raise ValueError("Effects and FDR values must be finite.")
    if not frame[FDR_COLUMN].between(0.0, 1.0).all() or frame[FDR_COLUMN].eq(0).any():
        raise ValueError("FDR values must be in (0, 1].")
    frame["minus_log10_fdr"] = -np.log10(frame[FDR_COLUMN])
    frame["fdr_category"] = np.select(
        [frame[FDR_COLUMN].le(0.05), frame[FDR_COLUMN].le(0.20)],
        ["FDR ≤ 5%", "5% < FDR ≤ 20%"],
        default="FDR > 20%",
    )
    labelled_ids = set(frame.loc[frame[FDR_COLUMN].le(0.20), "comparison_id"])
    if labelled_ids != set(annotations["comparison_id"]):
        raise ValueError("Annotation rows must exactly match interventions with FDR ≤ 20%.")
    return frame.merge(annotations, on="comparison_id", how="left", validate="one_to_one")


def configure_style() -> None:
    """Use the manuscript's embedded Liberation Sans typography."""
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 8.0,
            "axes.labelsize": 9.75,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_volcano(frame: pd.DataFrame, output_stem: Path) -> tuple[Path, Path]:
    """Render PDF and SVG without changing the tested intervention statistics."""
    configure_style()
    figure, axis = plt.subplots(figsize=(FIGURE_MM[0] / 25.4, FIGURE_MM[1] / 25.4))
    figure.subplots_adjust(left=0.12, right=0.975, bottom=0.17, top=0.965)
    styles = (
        ("FDR > 20%", NONSIGNIFICANT_COLOR, 18, 0.72),
        ("5% < FDR ≤ 20%", FDR_20_COLOR, 29, 0.95),
        ("FDR ≤ 5%", FDR_5_COLOR, 32, 1.0),
    )
    for zorder, (category, color, size, alpha) in enumerate(styles, start=1):
        subset = frame.loc[frame["fdr_category"].eq(category)]
        axis.scatter(
            subset[EFFECT_COLUMN],
            subset["minus_log10_fdr"],
            s=size,
            color=color,
            alpha=alpha,
            edgecolors="white" if category != "FDR > 20%" else "none",
            linewidths=0.6,
            zorder=zorder,
        )
    axis.axhline(-math.log10(0.05), color="#005A8D", linestyle=(0, (5, 2.5)), linewidth=0.85)
    axis.axhline(-math.log10(0.20), color="#A86B00", linestyle=(0, (2, 2)), linewidth=0.85)
    axis.axvline(0.0, color="#6B7280", linestyle=(0, (1.5, 2.5)), linewidth=0.65)
    axis.set_xlabel("Δ MetaClock Prediction")
    axis.set_ylabel("-log10(FDR)")
    axis.grid(axis="y", color="#E6E8EB", linewidth=0.45)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    effect_min, effect_max = frame[EFFECT_COLUMN].agg(["min", "max"])
    padding = max(0.8, 0.075 * float(effect_max - effect_min))
    axis.set_xlim(float(effect_min) - padding, float(effect_max) + padding)
    y_max = float(frame["minus_log10_fdr"].max()) * 1.075
    axis.set_ylim(-1.0, y_max)
    axis.set_yticks(np.arange(0.0, y_max + 0.01, 2.5))

    for row in frame.loc[frame[FDR_COLUMN].le(0.20)].itertuples(index=False):
        color = "#005A8D" if row.fdr_category == "FDR ≤ 5%" else "#A86B00"
        axis.annotate(
            str(row.short_label),
            xy=(float(getattr(row, EFFECT_COLUMN)), float(row.minus_log10_fdr)),
            xytext=(float(row.dx), float(row.dy)),
            textcoords="offset points",
            ha=str(row.ha),
            va=str(row.va),
            fontsize=7.5,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.45},
        )
    handles = [
        Line2D([], [], marker="o", linestyle="", color=color, label=label, markersize=5.2)
        for label, color in (
            ("FDR ≤ 5%", FDR_5_COLOR),
            ("5% < FDR ≤ 20%", FDR_20_COLOR),
            ("FDR > 20%", NONSIGNIFICANT_COLOR),
        )
    ]
    axis.legend(handles=handles, loc="upper left", frameon=False)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path, svg_path = output_stem.with_suffix(".pdf"), output_stem.with_suffix(".svg")
    figure.savefig(pdf_path, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)
    return pdf_path, svg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = plot_volcano(load_plot_data(args.results, args.annotations), args.output_stem)
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
