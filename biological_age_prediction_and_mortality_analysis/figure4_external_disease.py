#!/usr/bin/env python3
"""Rebuild the Figure 4 external-disease panel from public group summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multitest import multipletests

from .plotting import configure_style, save_vector_pair

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "external_disease_summary"
INPUTS = (
    INPUT_DIR / "external_disease_metaclock19_sexfix_three_model_summary.csv",
    INPUT_DIR / "external_disease_remaining9_three_model_summary.csv",
)
OUTPUT_DIR = ROOT / "outputs" / "figure4_external_disease"
STEM = "03_external_disease_heatmap"
MODELS = ("metaclock19", "baseline", "dnamphenoage")
MODEL_LABELS = ("Metaclock\n-biological", "Baseline", "DNAmPhenoAge")
PUBLIC_COLUMNS = (
    "accession", "disease", "control_n", "disease_n", "model",
    "mean_difference_disease_minus_control", "welch_ttest_p_value",
)
DISEASE_LABELS = {
    "Bowel": "IBD", "rheumatoid_arthritis": "RA", "obesity": "OB",
    "carcinoma": "HCC", "neurodegenerative_tauopathy": "NDT",
    "atherosclerosis": "ATH", "lupus": "SLE",
}


def star(p_value: float) -> str:
    return "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""


def build_source(paths: tuple[Path, ...] = INPUTS) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if missing := set(PUBLIC_COLUMNS).difference(frame.columns):
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame.loc[frame["model"].isin(MODELS), list(PUBLIC_COLUMNS)])
    summary = pd.concat(frames, ignore_index=True)
    if summary.duplicated(["accession", "model"]).any():
        raise ValueError("Each accession/model pair must occur exactly once.")
    if len(summary) != 42 or summary["accession"].nunique() != 14:
        raise ValueError("Expected exactly 14 cohorts and 42 rows for the three displayed models.")
    model_sets = summary.groupby("accession")["model"].agg(lambda values: set(values))
    if not model_sets.map(lambda values: values == set(MODELS)).all():
        raise ValueError("Every cohort must contain exactly the three displayed models.")
    if not np.isfinite(summary["mean_difference_disease_minus_control"]).all():
        raise ValueError("Age-residualized score differences must be finite.")
    if not summary["welch_ttest_p_value"].between(0, 1, inclusive="both").all():
        raise ValueError("Raw p-values must be between zero and one.")
    summary["bonferroni_p_value"] = summary.groupby("accession")[
        "welch_ttest_p_value"
    ].transform(lambda values: multipletests(values, method="bonferroni")[1])
    selected = summary.loc[
        summary.groupby("accession")["bonferroni_p_value"].transform("min").lt(0.05)
    ].copy()
    accessions = selected["accession"].drop_duplicates().tolist()
    selected["disease_order"] = selected["accession"].map(accessions.index)
    selected["disease_label"] = selected["disease"].map(DISEASE_LABELS)
    selected["model_label"] = selected["model"].map(dict(zip(
        MODELS, ("Metaclock-biological", "Baseline", "DNAmPhenoAge"), strict=True
    )))
    selected["model_order"] = selected["model"].map(MODELS.index)
    selected["significance"] = selected["bonferroni_p_value"].map(star)
    if len(accessions) != 7 or selected["disease_label"].isna().any():
        raise ValueError("Expected the seven Figure 4 diseases: IBD, RA, OB, HCC, NDT, ATH, SLE.")
    columns = [
        "disease", "model", "control_n", "disease_n",
        "mean_difference_disease_minus_control", "welch_ttest_p_value", "accession",
        "bonferroni_p_value", "disease_order", "disease_label", "model_label",
        "model_order", "significance",
    ]
    return selected.sort_values(["disease_order", "model_order"])[columns].reset_index(drop=True)


def draw_heatmap(source: pd.DataFrame) -> plt.Figure:
    configure_style()
    ordered = source[["disease", "disease_label", "disease_order"]].drop_duplicates()
    ordered = ordered.sort_values("disease_order")
    disease_order = ordered["disease"].tolist()
    effects = source.pivot(index="disease", columns="model", values="mean_difference_disease_minus_control")
    p_values = source.pivot(index="disease", columns="model", values="bonferroni_p_value")
    matrix = effects.loc[disease_order, list(MODELS)].to_numpy(float)
    p_matrix = p_values.loc[disease_order, list(MODELS)].to_numpy(float)
    figure, axis = plt.subplots(figsize=(90 / 25.4, 60 / 25.4))
    scaled_width_mm = (0.77 - 0.24) * 60 * 1.2
    left_mm = (90 - scaled_width_mm) / 2
    figure.subplots_adjust(left=left_mm / 90, right=(left_mm + scaled_width_mm) / 90,
                           bottom=0.30, top=0.94)
    cmap = LinearSegmentedColormap.from_list(
        "nature_diverging", ["#3C5488", "#F7F7F7", "#E64B35"], N=257
    )
    image = axis.imshow(matrix, cmap=cmap, vmin=-6, vmax=6, aspect="auto")
    axis.set_xticks(range(3), MODEL_LABELS, rotation=25, ha="right", fontsize=8)
    axis.set_yticks(range(7), ordered["disease_label"], fontsize=8)
    axis.set_ylabel("Disease")
    for row, column in np.ndindex(matrix.shape):
        stars = star(p_matrix[row, column])
        annotation = f"{matrix[row, column]:.1f}" + (f"\n{stars}" if stars else "")
        axis.text(column, row, annotation, ha="center", va="center",
                  color="#FFFFFF" if abs(matrix[row, column]) >= 3.9 else "#111111",
                  fontsize=6, linespacing=0.9)
    axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    axis.grid(which="minor", color="#F7F7F7", linewidth=0.35)
    axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.055, pad=0.04, shrink=0.9)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.set_ticks([-6, -4, -2, 0, 2, 4, 6])
    colorbar.set_label("Δ age-residualized score")
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_linewidth(0.2)
    return figure


def render_figure4_external_disease(
    output_dir: Path = OUTPUT_DIR, inputs: tuple[Path, ...] = INPUTS,
) -> tuple[Path, Path, Path]:
    source = build_source(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"{STEM}_source_data.csv"
    source.to_csv(source_path, index=False)
    pdf_path, svg_path = save_vector_pair(draw_heatmap(source), output_dir, STEM)
    return pdf_path, svg_path, source_path


def main() -> int:
    render_figure4_external_disease()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
