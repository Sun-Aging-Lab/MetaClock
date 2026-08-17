#!/usr/bin/env python3
"""Draw the intervention false-discovery-rate volcano plot."""

from __future__ import annotations

import argparse
import math
import os
import platform
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metaclock-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "metaclock-xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from generalization.fonts import register_fonts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results/intervention/selected/selected_intervention_results.csv"
DEFAULT_ANNOTATIONS = Path("/path/to/your/file")  # fill in your path
DEFAULT_OUTPUT_STEM = PROJECT_ROOT / "results/intervention/intervention_fdr_volcano"
FDR_COLUMN = "bh_q_value_after_dataset_exclusion"
EFFECT_COLUMN = "mean_diff_experiment_minus_control"
FIGURE_MM = (110.0, 60.0)
FDR_5_COLOR = "#0072B2"
FDR_20_COLOR = "#D89000"
NONSIGNIFICANT_COLOR = "#B8BDC5"
FDR_5_LINE_COLOR = "#005A8D"
FDR_20_LINE_COLOR = "#A86B00"
ZERO_LINE_COLOR = "#6B7280"
GRID_COLOR = "#E6E8EB"
TEXT_COLOR = "#202124"
OUTLINE_WIDTH_PT = 0.2
AXIS_TITLE_FONT_SIZE = 9.75
TICK_FONT_SIZE = 8.0
INTERVENTION_LABEL_FONT_SIZE = 7.5

# These five labels and offsets are part of the formal Figure 5 specification.
# They remain in a small external annotation table for workflow flexibility,
# but ``load_plot_data`` validates that table against this exact mapping.
SHORT_INTERVENTION_LABELS = {
    "curated_1fc35799c70a1f70": "Abdominal radiotherapy",
    "curated_fe95adeead1b848b": "Dexamethasone",
    "curated_a5aa77c6d7ffdccc": "Cigarette smoking",
    "curated_6ac8603ac6e44a0e": "Cytarabine (AraC)",
    "curated_1fa2b9d81d6ea894": "FAE",
}
LABEL_LAYOUT = {
    "curated_1fc35799c70a1f70": (-5, -1, "right", "center"),
    "curated_fe95adeead1b848b": (-5, -1, "right", "center"),
    "curated_a5aa77c6d7ffdccc": (-5, 16, "right", "bottom"),
    "curated_6ac8603ac6e44a0e": (5, 8, "left", "bottom"),
    "curated_1fa2b9d81d6ea894": (-5, 8, "right", "bottom"),
}


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
    if annotations["comparison_id"].duplicated().any():
        raise ValueError("Annotation table contains duplicate comparison_id values.")
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
    if labelled_ids != set(SHORT_INTERVENTION_LABELS):
        raise ValueError(
            "Current FDR <= 20% IDs differ from the formal five-intervention mapping: "
            f"{sorted(labelled_ids)}"
        )
    annotation_rows = annotations.set_index("comparison_id")
    for comparison_id, expected_label in SHORT_INTERVENTION_LABELS.items():
        expected_dx, expected_dy, expected_ha, expected_va = LABEL_LAYOUT[comparison_id]
        observed = annotation_rows.loc[comparison_id]
        observed_layout = (
            float(observed["dx"]),
            float(observed["dy"]),
            str(observed["ha"]),
            str(observed["va"]),
        )
        if str(observed["short_label"]) != expected_label or observed_layout != (
            float(expected_dx),
            float(expected_dy),
            expected_ha,
            expected_va,
        ):
            raise ValueError(f"Annotation mapping/layout differs for {comparison_id}")
    return frame.merge(annotations, on="comparison_id", how="left", validate="one_to_one")


def configure_style() -> None:
    """Use the formal bundled Liberation Sans manuscript typography."""
    register_fonts()
    plt.rcParams.update(
        {
            "font.family": ["Liberation Sans"],
            "font.sans-serif": ["Liberation Sans"],
            "font.size": TICK_FONT_SIZE,
            "axes.labelsize": AXIS_TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": 7.0,
            "axes.linewidth": OUTLINE_WIDTH_PT,
            "xtick.major.width": OUTLINE_WIDTH_PT,
            "xtick.minor.width": OUTLINE_WIDTH_PT,
            "ytick.major.width": OUTLINE_WIDTH_PT,
            "ytick.minor.width": OUTLINE_WIDTH_PT,
            "patch.linewidth": OUTLINE_WIDTH_PT,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Sans",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.fallback": None,
            "text.usetex": False,
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
        ("FDR > 20%", NONSIGNIFICANT_COLOR, 18, 0.72, 1),
        ("5% < FDR ≤ 20%", FDR_20_COLOR, 29, 0.95, 3),
        ("FDR ≤ 5%", FDR_5_COLOR, 32, 1.0, 4),
    )
    for category, color, size, alpha, zorder in styles:
        subset = frame.loc[frame["fdr_category"].eq(category)]
        axis.scatter(
            subset[EFFECT_COLUMN],
            subset["minus_log10_fdr"],
            s=size,
            color=color,
            alpha=alpha,
            edgecolors="white" if category != "FDR > 20%" else "none",
            linewidths=OUTLINE_WIDTH_PT,
            zorder=zorder,
        )
    axis.axhline(
        -math.log10(0.05),
        color=FDR_5_LINE_COLOR,
        linestyle=(0, (5, 2.5)),
        linewidth=0.85,
        zorder=2,
    )
    axis.axhline(
        -math.log10(0.20),
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
    effect_min, effect_max = frame[EFFECT_COLUMN].agg(["min", "max"])
    padding = max(0.8, 0.075 * float(effect_max - effect_min))
    axis.set_xlim(float(effect_min) - padding, float(effect_max) + padding)
    y_max = float(frame["minus_log10_fdr"].max()) * 1.075
    axis.set_ylim(-1.0, y_max)
    axis.set_yticks(np.arange(0.0, y_max + 0.01, 2.5))

    for row in frame.loc[frame[FDR_COLUMN].le(0.20)].itertuples(index=False):
        color = FDR_5_LINE_COLOR if row.fdr_category == "FDR ≤ 5%" else FDR_20_LINE_COLOR
        axis.annotate(
            str(row.short_label),
            xy=(float(getattr(row, EFFECT_COLUMN)), float(row.minus_log10_fdr)),
            xytext=(float(row.dx), float(row.dy)),
            textcoords="offset points",
            ha=str(row.ha),
            va=str(row.va),
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
    pdf_path, svg_path = output_stem.with_suffix(".pdf"), output_stem.with_suffix(".svg")
    metadata = {
        "Title": "",
        "Author": "",
        "Subject": "Post-exclusion BioMeta FDR volcano plot",
        "Creator": (
            f"Python {platform.python_version()}; Matplotlib {matplotlib.__version__}; "
            f"pandas {pd.__version__}; NumPy {np.__version__}"
        ),
    }
    figure.savefig(pdf_path, facecolor="white", metadata=metadata)
    figure.savefig(
        svg_path,
        facecolor="white",
        metadata={"Title": "", "Description": metadata["Subject"]},
    )
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
