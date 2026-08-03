#!/usr/bin/env python3
"""Age-correct external predictions and build the disease heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression


FIGURE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = FIGURE_DIR / "datasets/disease/external_disease_predictions.csv"
DEFAULT_OUTPUT_DIR = FIGURE_DIR / "results"

EXPECTED_MODELS = (
    "baseline",
    "bio_metaclock",
    "dnamic",
    "dnamphenoage",
    "dunedinpace",
    "hrsinchphenoage",
    "systemsage",
    "systemsageblood",
    "systemsagebrain",
    "systemsageheart",
    "systemsagehormone",
    "systemsageimmune",
    "systemsageinflammation",
    "systemsagekidney",
    "systemsageliver",
    "systemsagelung",
    "systemsagemetabolic",
    "systemsagemusculoskeletal",
    "zhangmortality",
)
FOCAL_MODELS = ("bio_metaclock", "baseline", "dnamphenoage")
MODEL_LABELS = {
    "bio_metaclock": "BioMeta",
    "baseline": "Baseline",
    "dnamphenoage": "DNAm\nPheno",
}
# Manuscript abbreviations and their full disease names.
DISEASE_LABELS = {
    "Bowel": (
        "IBD",
        "inflammatory bowel disease (Crohn disease / ulcerative colitis)",
    ),
    "rheumatoid_arthritis": ("RA", "rheumatoid arthritis"),
    "obesity": ("OB", "obesity"),
    "carcinoma": ("HCC", "hepatocellular carcinoma"),
    "neurodegenerative_tauopathy": ("NDT", "neurodegenerative tauopathy"),
    "atherosclerosis": ("ATH", "atherosclerosis"),
}
CMAP = LinearSegmentedColormap.from_list(
    "scientific_red", ("#FFF5F0", "#FEE0D2", "#FC9272", "#DE2D26", "#A50F15")
)
MM_TO_INCH = 1.0 / 25.4
FIGSIZE = (60 * MM_TO_INCH, 50 * MM_TO_INCH)


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    """Raise a readable error when an input table is incomplete."""
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def residualize_predictions_on_age(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fit one linear age model per clock and store prediction residuals."""
    output = predictions.copy()
    output["age_residual"] = np.nan
    for _, rows in output.groupby("model", sort=False):
        age = pd.to_numeric(rows["age"], errors="raise").to_numpy(dtype=float)
        predicted = pd.to_numeric(rows["predicted_age"], errors="raise").to_numpy(
            dtype=float
        )
        if not np.isfinite([age, predicted]).all():
            raise ValueError("Age correction requires finite age and prediction values")
        fit = LinearRegression().fit(age.reshape(-1, 1), predicted)
        output.loc[rows.index, "age_residual"] = predicted - fit.predict(
            age.reshape(-1, 1)
        )
    return output


def summarize_disease_control(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate disease-control residual differences and two-sided Welch tests."""
    rows: list[dict[str, float | int | str]] = []
    for (dataset, disease, model), values in predictions.groupby(
        ["dataset", "disease", "model"], sort=False
    ):
        control = values.loc[values["group"].eq("control"), "age_residual"].dropna()
        cases = values.loc[values["group"].eq("disease"), "age_residual"].dropna()
        if control.empty or cases.empty:
            raise ValueError(f"Missing disease/control rows for {dataset}/{model}")
        rows.append(
            {
                "dataset": dataset,
                "disease": disease,
                "model": model,
                "control_n": len(control),
                "disease_n": len(cases),
                "mean_difference_disease_minus_control": float(cases.mean() - control.mean()),
                "welch_ttest_p_value": float(
                    ttest_ind(cases, control, equal_var=False).pvalue
                ),
            }
        )
    return pd.DataFrame(rows)


def select_heatmap_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply within-disease Bonferroni correction and retain focal heatmap rows."""
    output = summary.copy()
    raw_p = pd.to_numeric(output["welch_ttest_p_value"], errors="coerce")
    group_columns = ["dataset", "disease"]
    expected = set(EXPECTED_MODELS)
    invalid_groups = [
        keys
        for keys, values in output.groupby(group_columns, sort=False)
        if set(values["model"]) != expected
    ]
    if invalid_groups:
        raise ValueError(
            "Every dataset/disease comparison must contain the exact 19-model "
            f"manuscript Bonferroni family; invalid groups: {invalid_groups}"
        )
    output["bonferroni_p_value"] = (raw_p * len(EXPECTED_MODELS)).clip(upper=1.0)
    output = output[output["model"].isin(FOCAL_MODELS)].copy()
    retained = output.loc[
        output["bonferroni_p_value"].lt(0.05), ["dataset", "disease"]
    ].drop_duplicates()
    output = output.merge(retained, on=["dataset", "disease"], how="inner")
    if output.empty:
        raise ValueError("No focal disease passes within-disease Bonferroni correction")
    unknown = set(output["disease"]).difference(DISEASE_LABELS)
    if unknown:
        raise ValueError(f"Add heatmap labels for diseases: {sorted(unknown)}")
    output["disease_abbreviation"] = output["disease"].map(
        lambda value: DISEASE_LABELS[str(value)][0]
    )
    output["disease_full_name"] = output["disease"].map(
        lambda value: DISEASE_LABELS[str(value)][1]
    )
    return output


def significance_stars(p_value: float) -> str:
    """Return a conventional corrected-p significance label."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plot_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    """Draw the compact Figure 4 disease heatmap."""
    order = (
        summary.loc[summary["model"].eq("bio_metaclock")]
        .sort_values(
            "mean_difference_disease_minus_control",
            key=lambda values: values.abs(),
            ascending=False,
        )["disease_abbreviation"]
        .tolist()
    )
    values = summary.pivot(
        index="disease_abbreviation",
        columns="model",
        values="mean_difference_disease_minus_control",
    ).reindex(index=order, columns=FOCAL_MODELS)
    corrected = summary.pivot(
        index="disease_abbreviation", columns="model", values="bonferroni_p_value"
    ).reindex(index=order, columns=FOCAL_MODELS)
    matrix = values.to_numpy(dtype=float)
    if float(np.nanmin(matrix)) < 0:
        raise ValueError("Sequential red heatmap requires non-negative selected values")
    vmax = max(float(np.nanmax(matrix)), 1.0)
    figure, axis = plt.subplots(figsize=FIGSIZE)
    figure.subplots_adjust(left=0.24, right=0.75, bottom=0.27, top=0.94)
    image = axis.imshow(matrix, cmap=CMAP, norm=Normalize(0, vmax), aspect="auto")
    axis.set_xticks(np.arange(len(FOCAL_MODELS)), [MODEL_LABELS[m] for m in FOCAL_MODELS])
    axis.tick_params(axis="x", rotation=25)
    axis.set_yticks(np.arange(len(order)), order)
    axis.set_ylabel("Disease")
    for row, column in np.ndindex(matrix.shape):
        value = matrix[row, column]
        stars = significance_stars(float(corrected.iloc[row, column]))
        color = "white" if value > 0.62 * vmax else "#111111"
        axis.text(column, row - (0.20 if stars else 0), f"{value:.1f}", ha="center", va="center", color=color)
        if stars:
            axis.text(column, row + 0.25, stars, ha="center", va="center", color=color)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.035)
    colorbar.set_label(r"$\Delta$ predicted phenoage")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "03_external_disease_heatmap.pdf")
    figure.savefig(output_dir / "03_external_disease_heatmap.svg")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse the relative dataset path and optional replacement path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Build panel 3 and its displayed result table."""
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    # Accept the public, explicit schema; rename upstream fields before this step.
    require_columns(
        predictions,
        {"dataset", "disease", "sample_id", "age", "group", "model", "predicted_age"},
        args.predictions,
    )
    if not predictions["group"].isin(["control", "disease"]).all():
        raise ValueError("group must contain only 'control' and 'disease'")
    summary = summarize_disease_control(residualize_predictions_on_age(predictions))
    heatmap = select_heatmap_rows(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap.to_csv(args.output_dir / "03_disease_heatmap_statistics.csv", index=False)
    plot_heatmap(heatmap, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
