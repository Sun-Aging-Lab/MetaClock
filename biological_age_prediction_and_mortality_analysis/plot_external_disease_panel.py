"""Render the external disease-control difference heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .config import EXTERNAL_MODELS
from .plotting import LABELS, save_vector_pair

DISEASE_LABELS = {
    "sclerosis": "SSc",
    "copd": "COPD",
    "coeliac_disease": "CeD",
    "alzheimer_disease": "AD",
    "Bowel": "IBD",
    "rheumatoid_arthritis": "RA",
    "obesity": "OB",
    "carcinoma": "HCC",
    "neurodegenerative_tauopathy": "NDT",
    "huntington_disease": "HD",
    "atherosclerosis": "ATH",
    "nafld_nash": "NAFLD",
    "parkinson_disease": "PD",
    "graves_disease": "GD",
    "lupus": "SLE",
}


def significance(p_value: float) -> str:
    """Map a corrected P value to a star label."""
    return "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""


def build_heatmap_source(tests: pd.DataFrame) -> pd.DataFrame:
    """Retain datasets significant in any model and order by BioMeta effect."""
    required = {
        "dataset",
        "disease",
        "model",
        "mean_difference_disease_minus_control",
        "bonferroni_p_value",
    }
    if missing := required.difference(tests.columns):
        raise KeyError(f"External tests are missing columns: {sorted(missing)}")
    if tests.duplicated(["dataset", "disease", "model"]).any():
        raise ValueError("External test source contains duplicate dataset/disease/model rows.")
    retained = tests.loc[
        tests["bonferroni_p_value"].lt(0.05), ["dataset", "disease"]
    ].drop_duplicates()
    source = tests.merge(retained, on=["dataset", "disease"], how="inner")
    if source.empty:
        raise ValueError("No external dataset is significant in any displayed model.")
    biometa = source[source["model"].eq("bio_metaclock")].sort_values(
        "mean_difference_disease_minus_control", ascending=False
    )
    order = biometa[["dataset", "disease"]].assign(disease_order=np.arange(len(biometa)))
    source = source.merge(order, on=["dataset", "disease"], validate="many_to_one")
    source["disease_label"] = source["disease"].map(DISEASE_LABELS)
    if source["disease_label"].isna().any():
        raise ValueError("Missing an external-disease display abbreviation.")
    source["significance"] = source["bonferroni_p_value"].map(significance)
    source["model_order"] = source["model"].map(
        {model: index for index, model in enumerate(EXTERNAL_MODELS)}
    )
    return source.sort_values(["disease_order", "model_order"]).reset_index(drop=True)


def plot_external_disease_heatmap(tests_path: Path, output_dir: Path) -> None:
    """Plot age-adjusted disease-control differences and corrected significance."""
    source = build_heatmap_source(pd.read_csv(tests_path))
    ordered = (
        source[["dataset", "disease", "disease_label", "disease_order"]]
        .drop_duplicates()
        .sort_values("disease_order")
    )
    keys = list(ordered[["dataset", "disease"]].itertuples(index=False, name=None))
    indexed = source.set_index(["dataset", "disease", "model"])
    values = np.array(
        [
            [
                indexed.loc[(*key, model), "mean_difference_disease_minus_control"]
                for model in EXTERNAL_MODELS
            ]
            for key in keys
        ],
        float,
    )
    corrected = np.array(
        [
            [indexed.loc[(*key, model), "bonferroni_p_value"] for model in EXTERNAL_MODELS]
            for key in keys
        ],
        float,
    )
    figure, axis = plt.subplots(figsize=(90 / 25.4, 60 / 25.4))
    layout_width_mm = (0.77 - 0.24) * 60 * 1.2
    layout_left_mm = (90 - layout_width_mm) / 2
    figure.subplots_adjust(
        left=layout_left_mm / 90,
        right=(layout_left_mm + layout_width_mm) / 90,
        bottom=0.27 * 50 * 1.2 / 60,
        top=0.94 * 50 * 1.2 / 60,
    )
    cmap = LinearSegmentedColormap.from_list(
        "disease", ["#3C5488", "#F7F7F7", "#E64B35"], N=257
    )
    image = axis.imshow(values, cmap=cmap, vmin=-6, vmax=6, aspect="auto")
    axis.set_xticks(range(3), [LABELS[model] for model in EXTERNAL_MODELS], rotation=25, ha="right")
    axis.set_yticks(range(len(keys)), ordered["disease_label"])
    axis.set_ylabel("Disease")
    for row, column in np.ndindex(values.shape):
        stars = significance(corrected[row, column])
        label = f"{values[row, column]:.1f}" + (f"\n{stars}" if stars else "")
        axis.text(
            column,
            row,
            label,
            ha="center",
            va="center",
            fontsize=6,
            color="white" if abs(values[row, column]) >= 3.9 else "#111111",
            linespacing=0.9,
        )
    axis.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    axis.grid(which="minor", color="#F7F7F7", linewidth=0.35)
    axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.055, pad=0.04, shrink=0.9)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.set_ticks([-6, -4, -2, 0, 2, 4, 6])
    colorbar.set_label("Δ predicted phenoage")
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_linewidth(0.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(output_dir / "03_external_disease_heatmap_source_data.csv", index=False)
    save_vector_pair(figure, output_dir, "03_external_disease_heatmap")
