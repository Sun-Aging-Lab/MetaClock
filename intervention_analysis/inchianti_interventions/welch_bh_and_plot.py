#!/usr/bin/env python3
"""Test predefined InCHIANTI contrasts and plot age + sex adjusted BioMetaClock.

Interventions were filtered to exclude disease context.

Inputs are the participant table produced by ``predict_and_adjust_biometaclock.py``,
a CSV/CSV.GZ membership table (comparison_id, ID, sampleID, group), and a comparison
CSV (comparison_id, plot_label; optional variable, label_type, group0_definition,
group1_definition). Memberships contain the already assigned, disjoint groups
0 and 1. Every supplied comparison is tested. Extra input columns are ignored.

Effects are group 1 minus group 0 mean residuals, in predicted-age years. SciPy
provides two-sided Welch tests, 95% confidence intervals, and one Benjamini-Hochberg
family across all comparisons. PDF, SVG and PNG volcanoes show every q <= 0.20
label using the established manuscript layout and direction-dependent colors.

All four CLI paths are required: --predictions, --memberships, --comparisons,
--output-dir. Paths may be absolute or relative to the current working directory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if (REPOSITORY_ROOT / "biological_age_prediction_and_mortality_analysis").is_dir():
    sys.path.insert(0, str(REPOSITORY_ROOT))

PREDICTION_COLUMNS = (
    "ID", "sampleID", "age_years", "SEX", "pred_age", "bioage_age_sex_residual"
)
MEMBERSHIP_COLUMNS = ("comparison_id", "ID", "sampleID", "group")
COMPARISON_COLUMNS = (
    "comparison_id", "plot_label", "variable", "label_type",
    "group0_definition", "group1_definition",
)
RESIDUAL_COLUMN = "bioage_age_sex_residual"
INTERVENTION_CONTEXT = "Interventions were filtered to exclude disease context."
PALETTE = {
    "negative_05": "#2166AC", "negative_20": "#92C5DE",
    "positive_05": "#B2182B", "positive_20": "#F4A582", "other": "#B8BDC5",
}


def parse_args() -> argparse.Namespace:
    """Parse explicit input paths without inferring any intervention assignments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="672-person predictions and cohort-wide age + sex residuals.")
    parser.add_argument("--memberships", type=Path, required=True,
                        help="Predefined comparison_id, ID, sampleID, group CSV(.gz).")
    parser.add_argument("--comparisons", type=Path, required=True,
                        help="Comparison labels and optional group definitions CSV(.gz).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for Welch/BH tables, volcanoes and provenance.")
    return parser.parse_args()


def read_columns(path: Path, allowed: tuple[str, ...], required: tuple[str, ...]) -> pd.DataFrame:
    """Load only the declared input contract, preserving identifiers as strings."""
    frame = pd.read_csv(
        path, usecols=lambda name: name in allowed, float_precision="round_trip",
        dtype={name: "string" for name in ("ID", "sampleID", "comparison_id", "plot_label")},
    )
    if missing := set(required).difference(frame.columns):
        raise ValueError(f"{path.resolve()}: missing required columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path.resolve()}: input contains no rows")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path.resolve()}: required columns contain missing values")
    for name in ("ID", "sampleID", "comparison_id", "plot_label"):
        if name in frame and frame[name].str.strip().eq("").any():
            raise ValueError(f"{path.resolve()}: {name} contains an empty identifier or label")
    return frame


def load_inputs(
    predictions_path: Path, memberships_path: Path, comparisons_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate the 672-person cohort and join only fresh predictions to memberships.

    Both participant and sample identifiers must match. Duplicate participants
    within a comparison are rejected, including any assignment to both groups.
    Comparison metadata must describe exactly the supplied membership contrasts.
    """
    predictions = read_columns(predictions_path, PREDICTION_COLUMNS, PREDICTION_COLUMNS)
    memberships = read_columns(memberships_path, MEMBERSHIP_COLUMNS, MEMBERSHIP_COLUMNS)
    comparisons = read_columns(comparisons_path, COMPARISON_COLUMNS, ("comparison_id", "plot_label"))
    if len(predictions) != 672 or any(predictions[name].duplicated().any() for name in ("ID", "sampleID")):
        raise ValueError("Predictions must contain exactly 672 unique participants and samples")
    numeric = ["age_years", "SEX", "pred_age", RESIDUAL_COLUMN]
    predictions[numeric] = predictions[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(predictions[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Participant ages, sex codes, predictions and residuals must be finite")
    if not predictions.SEX.isin([1, 2]).all():
        raise ValueError("SEX must use 1 = male and 2 = female")
    if comparisons.comparison_id.duplicated().any():
        raise ValueError("Comparison metadata contains duplicate comparison_id values")
    if memberships.duplicated(["comparison_id", "ID"]).any():
        raise ValueError("Duplicate participants or overlapping groups within a comparison")
    if memberships.duplicated(["comparison_id", "sampleID"]).any():
        raise ValueError("Duplicate samples within a comparison")
    memberships["group"] = pd.to_numeric(memberships.group, errors="raise")
    if not memberships.group.isin([0, 1]).all():
        raise ValueError("Every membership group must be 0 or 1")
    memberships["group"] = memberships.group.astype(int)
    if set(comparisons.comparison_id) != set(memberships.comparison_id):
        raise ValueError("Comparison metadata IDs must exactly match membership comparison IDs")
    # usecols above prevents stale prediction/residual columns from entering this merge.
    members = memberships.merge(predictions, on=["ID", "sampleID"], how="left",
                                validate="many_to_one", indicator=True)
    if not members["_merge"].eq("both").all():
        raise ValueError("Some membership ID/sampleID pairs are absent from the prediction cohort")
    members = members.drop(columns="_merge")
    counts = members.groupby(["comparison_id", "group"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[0, 1], fill_value=0)
    if counts.lt(2).any().any():
        invalid = counts.index[counts.lt(2).any(axis=1)].tolist()
        raise ValueError(f"At least two participants per group are required: {invalid}")
    return predictions, members, comparisons


def test_comparisons(members: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    """Use maintained SciPy Welch/CI/BH methods for every supplied comparison."""
    metadata = comparisons.set_index("comparison_id")
    rows: list[dict[str, Any]] = []
    for comparison_id, subset in members.groupby("comparison_id", sort=True):
        groups = {group: subset.loc[subset.group.eq(group)] for group in (0, 1)}
        values = {group: frame[RESIDUAL_COLUMN].to_numpy(dtype=float)
                  for group, frame in groups.items()}
        # SciPy reports the first array minus the second, matching group 1 - group 0.
        test = stats.ttest_ind(values[1], values[0], equal_var=False, alternative="two-sided")
        interval = test.confidence_interval(confidence_level=0.95)
        statistics = np.asarray([test.statistic, test.df, test.pvalue, interval.low, interval.high])
        if not np.isfinite(statistics).all() or not 0 <= test.pvalue <= 1 or test.df <= 0:
            raise ValueError(f"{comparison_id}: Welch test/95% CI is undefined or nonfinite")
        row: dict[str, Any] = {"comparison_id": comparison_id, **metadata.loc[comparison_id].to_dict()}
        row.update({
            "effect_direction": "group1 minus group0",
            "effect_years": float(values[1].mean() - values[0].mean()),
            "welch_t": float(test.statistic), "welch_df": float(test.df),
            "p_value": float(test.pvalue), "effect_ci95_low": float(interval.low),
            "effect_ci95_high": float(interval.high), "fit_n": 672,
            "age_adjustment": "pred_age ~ age_years + C(SEX)",
            "residual_column": RESIDUAL_COLUMN,
        })
        for group, frame in groups.items():
            row.update({
                f"n_group{group}": len(frame),
                f"mean_residual_group{group}": float(values[group].mean()),
                f"sd_residual_group{group}": float(values[group].std(ddof=1)),
                f"mean_prediction_group{group}": float(frame.pred_age.mean()),
                f"mean_age_group{group}": float(frame.age_years.mean()),
                f"n_female_group{group}": int(frame.SEX.eq(2).sum()),
            })
        rows.append(row)
    results = pd.DataFrame(rows)
    # The supplied contrast set forms one family, regardless of intervention type.
    results["bh_q_value"] = stats.false_discovery_control(results.p_value.to_numpy(), method="bh")
    results["bh_family_n"] = len(results)
    results["significant_bh_05"] = results.bh_q_value.le(0.05)
    results["significant_bh_20"] = results.bh_q_value.le(0.20)
    results["minus_log10_bh"] = -np.log10(results.bh_q_value.clip(lower=np.finfo(float).tiny))
    return results.sort_values(["bh_q_value", "p_value", "comparison_id"], kind="stable").reset_index(drop=True)


def directional_colors(effects: pd.Series, q_values: pd.Series) -> np.ndarray:
    """Map residual direction and BH category to the established blue/red palette."""
    return np.select(
        [(effects < 0) & (q_values <= .05), (effects > 0) & (q_values <= .05),
         (effects < 0) & (q_values <= .20), (effects > 0) & (q_values <= .20)],
        [PALETTE[name] for name in ("negative_05", "positive_05", "negative_20", "positive_20")],
        default=PALETTE["other"],
    )


def plot_volcano(results: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Render the manuscript layout and audit every significant text annotation.

    Annotation columns are retained from the established InCHIANTI figure. The
    canvas height grows with the number of labels; no significant label is capped.
    Only this presentation layout is custom, while Matplotlib supplies rendering.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from generalization.fonts import register_fonts
    from intervention_analysis import plot_geo_biometaclock_volcano_fdr as style

    register_fonts()
    # Reuse shared manuscript constants without creating its unrelated cache path.
    plt.rcParams.update({
        "font.family": ["Liberation Sans"], "font.sans-serif": ["Liberation Sans"],
        "axes.labelsize": style.AXIS_TITLE_FONT_SIZE, "xtick.labelsize": style.TICK_FONT_SIZE,
        "ytick.labelsize": style.TICK_FONT_SIZE, "legend.fontsize": 7,
        "axes.linewidth": style.OUTLINE_WIDTH_PT, "xtick.major.width": style.OUTLINE_WIDTH_PT,
        "xtick.minor.width": style.OUTLINE_WIDTH_PT, "ytick.major.width": style.OUTLINE_WIDTH_PT,
        "ytick.minor.width": style.OUTLINE_WIDTH_PT, "patch.linewidth": style.OUTLINE_WIDTH_PT,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    data = results.copy()
    data["fdr_category"] = np.select([data.bh_q_value.le(.05), data.bh_q_value.le(.20)],
                                     ["FDR ≤ 5%", "5% < FDR ≤ 20%"], default="FDR > 20%")
    labels = data.loc[data.bh_q_value.le(.20)].sort_values(
        ["minus_log10_bh", "comparison_id"], ascending=[False, True])
    sides = {side: labels.loc[labels.effect_years.lt(0).eq(side < 0)] for side in (-1, 1)}
    largest = max(map(len, sides.values()))
    canvas = [300., max(160., (largest - 1) * 11. * 25.4 / 72 / (.77 * .86))]
    fig, ax = plt.subplots(figsize=np.asarray(canvas) / 25.4)
    fig.subplots_adjust(left=.29, right=.70, bottom=.17, top=.94)
    for category, size, alpha, zorder in [
        ("FDR > 20%", 20, .70, 1), ("5% < FDR ≤ 20%", 32, .95, 3), ("FDR ≤ 5%", 36, 1., 4),
    ]:
        subset = data.loc[data.fdr_category.eq(category)]
        ax.scatter(subset.effect_years, subset.minus_log10_bh, s=size,
                   c=directional_colors(subset.effect_years, subset.bh_q_value), alpha=alpha,
                   edgecolors="white" if category != "FDR > 20%" else "none",
                   linewidths=style.OUTLINE_WIDTH_PT, zorder=zorder)
    for cutoff in (.05, .20):
        ax.axhline(-np.log10(cutoff), color=style.ZERO_LINE_COLOR,
                   linestyle=(0, (5, 2.5)), linewidth=.85, zorder=2)
        ax.text(.02, -np.log10(cutoff) + .015, f"BH FDR = {cutoff:g}",
                transform=ax.get_yaxis_transform(), va="bottom", fontsize=7,
                color=style.ZERO_LINE_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
    ax.axvline(0, color=style.ZERO_LINE_COLOR, linestyle=(0, (1.5, 2.5)), linewidth=.65)
    ax.set_xlabel("Δ age + sex adjusted BioMeta (years)")
    ax.set_ylabel("−log₁₀(BH-adjusted P)")
    ax.yaxis.set_label_coords(-.64, .5)
    ax.grid(axis="y", color=style.GRID_COLOR, linewidth=.45)
    ax.spines[["top", "right"]].set_visible(False)
    xmax = max(1., float(data.effect_years.abs().max()) * 1.3)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-.08, max(1.55, float(data.minus_log10_bh.max()) * 1.25))
    annotations: dict[int, list[Any]] = {}
    records: list[dict[str, Any]] = []
    for side, subset in sides.items():
        if len(subset) <= 8:
            ymin, ymax = ax.get_ylim()
            ys = np.clip((subset.minus_log10_bh.to_numpy() - ymin) / (ymax - ymin), .07, .93)
            minimum_gap = 11. / (canvas[1] / 25.4 * 72 * .77)
            for index in range(1, len(ys)):
                ys[index] = min(ys[index], ys[index - 1] - minimum_gap)
            if len(ys) and ys[-1] < .07:
                ys += .07 - ys[-1]
        else:
            ys = np.linspace(.93, .07, len(subset))
        annotations[side] = []
        for row, y in zip(subset.itertuples(index=False), ys, strict=True):
            color = PALETTE["negative_05" if row.effect_years < 0 else "positive_05"]
            annotation = ax.annotate(
                row.plot_label, xy=(row.effect_years, row.minus_log10_bh),
                xytext=(1.04 if side > 0 else -.08, y), textcoords=ax.transAxes,
                ha="left" if side > 0 else "right", va="center", fontsize=7.5, color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": .5,
                            "relpos": (0., .5) if side > 0 else (1., .5)}, annotation_clip=False,
            )
            annotations[side].append(annotation)
            records.append({"comparison_id": row.comparison_id, "plot_label": row.plot_label,
                            "bh_q_value": row.bh_q_value, "side": side, "text_y_axes_fraction": float(y)})
    legend = [Line2D([], [], marker="o", linestyle="", markerfacecolor=PALETTE[key],
                     markeredgecolor="white", markeredgewidth=style.OUTLINE_WIDTH_PT,
                     label=label, markersize=5.2) for key, label in [
                         ("negative_05", "Lower residual: FDR ≤ 5%"),
                         ("negative_20", "Lower residual: 5% < FDR ≤ 20%"),
                         ("positive_05", "Higher residual: FDR ≤ 5%"),
                         ("positive_20", "Higher residual: 5% < FDR ≤ 20%"),
                         ("other", "FDR > 20% / zero effect"),
                     ]]
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=7)
    fig.text(.29, .095, "Negative: lower residual in group 1     Positive: higher residual in group 1", fontsize=7.5)
    fig.text(.29, .067, f"{len(data)} Welch tests; one BH family; all {len(labels)} labels with q ≤ 20% shown", fontsize=7.5)
    fig.text(.29, .039, "Residuals adjusted for age + sex; one model fitted on all 672 participants", fontsize=7)
    fig.text(.29, .015, INTERVENTION_CONTEXT, fontsize=7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for items in annotations.values():
        bounds = [Text.get_window_extent(item, renderer=renderer) for item in items]
        if not all(b.x0 >= 0 and b.x1 <= fig.bbox.width and b.y0 >= 0 and b.y1 <= fig.bbox.height for b in bounds):
            plt.close(fig)
            raise ValueError("A volcano label extends outside the canvas; shorten its supplied plot_label")
        if any(a.overlaps(b) for index, a in enumerate(bounds) for b in bounds[index + 1:]):
            plt.close(fig)
            raise ValueError("Volcano annotation text overlaps")
    stem = output_dir / "inchianti_lifestyle_biometaclock_volcano"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=300, facecolor="white")
    plt.close(fig)
    pd.DataFrame(records, columns=["comparison_id", "plot_label", "bh_q_value", "side", "text_y_axes_fraction"]).to_csv(
        output_dir / "volcano_label_audit.csv", index=False)
    return {"n_tests": len(data), "n_labels": len(labels), "canvas_mm": canvas,
            "annotation_text_disjoint_inside_canvas": True, "directional_palette": PALETTE,
            "label_q_threshold": .20, "font_family": "Liberation Sans",
            "shared_style_parameters": {
                "axes_label_size": style.AXIS_TITLE_FONT_SIZE,
                "tick_label_size": style.TICK_FONT_SIZE,
                "outline_width_pt": style.OUTLINE_WIDTH_PT,
            }}


def main() -> None:
    """Validate inputs, run every Welch/BH comparison, render figures and provenance."""
    args = parse_args()
    started = datetime.now(timezone.utc).isoformat()
    inputs = {
        name: str(getattr(args, name).resolve())
        for name in ("predictions", "memberships", "comparisons")
    }
    predictions, members, comparisons = load_inputs(args.predictions, args.memberships, args.comparisons)
    results = test_comparisons(members, comparisons)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output_dir / "matplotlib_cache")
    os.environ["XDG_CACHE_HOME"] = str(output_dir / "xdg_cache")
    results.to_csv(output_dir / "lifestyle_welch_bh_results.csv", index=False)
    results.loc[results.significant_bh_05].to_csv(output_dir / "significant_bh_05.csv", index=False)
    results.loc[results.significant_bh_20].to_csv(output_dir / "significant_bh_20.csv", index=False)
    plot = plot_volcano(results, output_dir)
    manifest = {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "intervention_context": INTERVENTION_CONTEXT, "inputs": inputs,
        "script": str(Path(__file__).resolve()),
        "software": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "matplotlib")}},
        "runtime": {"platform": platform.platform(), "python_executable": sys.executable,
                    "command": sys.argv, "working_directory": str(Path.cwd())},
        "cohort": {"n_participants": len(predictions), "n_samples": predictions.sampleID.nunique(),
                   "prediction_reference": inputs["predictions"], "fit_scope": "single cohort-wide fit",
                   "adjustment_formula": "pred_age ~ constant + age_years + female",
                   "sex_codes": {"1": "male", "2": "female"}, "residual_column": RESIDUAL_COLUMN},
        "statistics": {"test": "scipy.stats.ttest_ind", "equal_var": False, "alternative": "two-sided",
                       "effect": "group1 mean residual minus group0 mean residual", "confidence_level": .95,
                       "bh_method": "scipy.stats.false_discovery_control(method='bh')",
                       "bh_family_n": len(results), "minimum_group_n": 2,
                       "n_significant_bh_05": int(results.significant_bh_05.sum()),
                       "n_significant_bh_20": int(results.significant_bh_20.sum())},
        "plot": plot,
        "custom_logic": "Input adaptation, validation, reporting and manuscript annotation layout only.",
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Completed {len(results)} Welch tests in one BH family; outputs: {output_dir}")


if __name__ == "__main__":
    main()
