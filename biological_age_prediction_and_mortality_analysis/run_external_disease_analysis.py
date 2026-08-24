#!/usr/bin/env python3
"""Infer three age models in one external disease dataset and draw a heatmap.

Edit only ``DATASET_PATH``. The published BioMetaClock weights are bundled in
this repository. Run ``run_biological_age_prediction.py`` first only to create
the direct-CpG baseline artifact, which is intentionally not bundled because
it contains one weight per training CpG.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

from repository_paths import relative_path

from .config import EXTERNAL_COMPONENTS
from .data import numeric_matrix, validate_dnam_feature_names
from .plotting import LABELS, configure_style, save_vector_pair
from .provenance import software_versions
from .published_biometaclock import load_published_biometaclock

# --- Paths: this is the only user-edited raw input ---

DATASET_PATH = Path("data/biological_age/external_disease.parquet")
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs" / "biological_age_prediction" / "models"
OUTPUT_DIR = ROOT / "outputs" / "external_disease"
PYAGING_CACHE_DIR = ROOT / "pyaging_data"


# --- Fixed model and schema definitions ---

MODEL_ORDER = ("bio_metaclock", "baseline", "dnamphenoage")
CONTROL_LABELS = {"control", "ctrl", "healthy", "normal", "0", "0.0", "false"}
REQUIRED_COLUMNS = {"sampleID", "age", "disease_state"}


# --- Input and trained-artifact loading ---


def read_dataset(path: Path) -> pd.DataFrame:
    """Read one row-oriented external DNAm CSV/CSV.GZ or Parquet dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Set DATASET_PATH to the external disease dataset: {path}")
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise KeyError(f"External dataset is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["sampleID"] = frame["sampleID"].astype(str)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame = frame.dropna(subset=["sampleID", "age", "disease_state"]).reset_index(drop=True)
    if frame["sampleID"].duplicated().any():
        raise ValueError("sampleID must identify one external DNAm row.")
    frame["dataset"] = frame["dataset"].astype(str) if "dataset" in frame else path.stem
    frame["disease"] = frame["disease"].astype(str) if "disease" in frame else path.stem
    frame["group"] = normalize_group(frame["disease_state"])
    return frame


def normalize_group(values: pd.Series) -> pd.Series:
    """Normalize common healthy/control labels; all other nonmissing labels are disease."""
    labels = values.astype("string").str.strip().str.lower()
    return pd.Series(
        np.where(
            labels.isin(CONTROL_LABELS) | labels.str.contains("control", na=False),
            "control",
            "disease",
        ),
        index=values.index,
        dtype="string",
    )


def load_trained_artifact(name: str) -> tuple[Any, list[str]]:
    """Load the complete fitted pipeline and its exact ordered feature manifest."""
    model_path = MODEL_DIR / f"{name}.joblib"
    feature_path = MODEL_DIR / f"{name}_features.csv"
    if not model_path.exists() or not feature_path.exists():
        raise FileNotFoundError(
            f"Missing {name} artifact. Run run_biological_age_prediction.py first; "
            f"expected {model_path} and {feature_path}."
        )
    features = pd.read_csv(feature_path)["feature"].astype(str).tolist()
    model = joblib.load(model_path)
    if int(getattr(model, "n_features_in_", -1)) != len(features):
        raise ValueError(f"{name} model and ordered feature manifest have different widths.")
    return model, features


# --- PyAging API inference for BioMeta components and DNAmPhenoAge ---


def run_pyaging(frame: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    """Call PyAging predict_age on the external CpG matrix."""
    try:
        import anndata as ad
        import pyaging as pya
    except ImportError as exc:
        raise RuntimeError("Install anndata and pyaging for external disease inference.") from exc
    cpgs = [str(column) for column in frame.columns if str(column).lower().startswith("cg")]
    if not cpgs:
        raise ValueError("External dataset has no CpG beta columns named cg....")
    matrix = numeric_matrix(frame, cpgs)
    adata = ad.AnnData(matrix)
    adata.var_names = cpgs
    adata.obs = frame[["sampleID", "age"]].copy().set_index("sampleID", drop=False)
    adata.obs["Age"] = adata.obs["age"]
    for source in ("female", "Female"):
        if source in frame.columns:
            adata.obs["female"] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
            adata.obs["Female"] = adata.obs["female"]
            break
    PYAGING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pya.pred.predict_age(
        adata,
        clocks,
        dir=relative_path(PYAGING_CACHE_DIR),
        batch_size=512,
        clean=True,
        verbose=False,
    )
    outputs = adata.obs[["sampleID", *clocks]].reset_index(drop=True)
    del adata, matrix
    gc.collect()
    return outputs


def baseline_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Build the direct-CpG design in training order, zero-filling unavailable probes."""
    matrix = np.zeros((len(frame), len(features)), dtype=np.float32)
    present = [feature for feature in features if feature in frame.columns]
    if present:
        positions = {feature: index for index, feature in enumerate(features)}
        values = numeric_matrix(frame, present)
        matrix[:, [positions[feature] for feature in present]] = values
    return matrix


# --- Within-dataset age correction and Welch disease-control tests ---


def age_correct_and_test(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Regress prediction on age within dataset/model, then run two-sided Welch tests."""
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:
        raise RuntimeError("Install statsmodels for Bonferroni correction.") from exc
    corrected = predictions.copy()
    corrected["difference"] = np.nan
    fit_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for (dataset, disease, model), subset in corrected.groupby(
        ["dataset", "disease", "model"], sort=False
    ):
        age = subset["age"].to_numpy(float)
        pred = subset["pred_age"].to_numpy(float)
        fit = LinearRegression().fit(age[:, None], pred)
        residual = pred - fit.predict(age[:, None])
        corrected.loc[subset.index, "difference"] = residual
        fit_rows.append(
            {
                "dataset": dataset,
                "disease": disease,
                "model": model,
                "age_coef": float(fit.coef_[0]),
                "intercept": float(fit.intercept_),
                "r2_age_explained": float(fit.score(age[:, None], pred)),
                "n_fit": len(subset),
            }
        )
        groups = subset["group"].to_numpy(str)
        controls, cases = residual[groups == "control"], residual[groups == "disease"]
        if min(len(controls), len(cases)) < 2:
            raise ValueError(f"{dataset}/{model} needs at least two control and disease samples.")
        test_rows.append(
            {
                "dataset": dataset,
                "disease": disease,
                "model": model,
                "control_n": len(controls),
                "disease_n": len(cases),
                "mean_difference_disease_minus_control": float(cases.mean() - controls.mean()),
                "welch_ttest_p_value": float(
                    ttest_ind(cases, controls, equal_var=False, nan_policy="raise").pvalue
                ),
            }
        )
    tests = pd.DataFrame(test_rows)
    tests["bonferroni_p_value"] = np.nan
    for _, subset in tests.groupby(["dataset", "disease"], sort=False):
        _, adjusted, _, _ = multipletests(
            subset["welch_ttest_p_value"].to_numpy(float),
            method="bonferroni",
        )
        tests.loc[subset.index, "bonferroni_p_value"] = adjusted
    return corrected, pd.DataFrame(fit_rows), tests


# --- Heatmap ---


def significance(value: float) -> str:
    """Convert a Bonferroni-adjusted P value to a star annotation."""
    return "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else ""


def plot_heatmap(tests: pd.DataFrame) -> None:
    """Plot age-corrected disease-minus-control effects for all three models."""
    configure_style()
    order = (
        tests[tests["model"].eq("bio_metaclock")]
        .sort_values("mean_difference_disease_minus_control", ascending=False)[
            ["dataset", "disease"]
        ]
        .drop_duplicates()
    )
    keys = list(order.itertuples(index=False, name=None))
    indexed = tests.set_index(["dataset", "disease", "model"])
    values = np.array(
        [
            [
                indexed.loc[(*key, model), "mean_difference_disease_minus_control"]
                for model in MODEL_ORDER
            ]
            for key in keys
        ],
        dtype=float,
    )
    p_values = np.array(
        [
            [indexed.loc[(*key, model), "bonferroni_p_value"] for model in MODEL_ORDER]
            for key in keys
        ],
        dtype=float,
    )
    height = max(45, 8 * len(keys) + 20) / 25.4
    figure, axis = plt.subplots(figsize=(90 / 25.4, height))
    cmap = LinearSegmentedColormap.from_list("disease", ["#3C5488", "#F7F7F7", "#E64B35"])
    limit = max(6.0, float(np.nanmax(np.abs(values))))
    image = axis.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(
        range(len(MODEL_ORDER)),
        [LABELS[model] for model in MODEL_ORDER],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(range(len(keys)), [f"{disease}\n{dataset}" for dataset, disease in keys])
    for row, column in np.ndindex(values.shape):
        stars = significance(p_values[row, column])
        axis.text(
            column,
            row,
            f"{values[row, column]:.1f}" + (f"\n{stars}" if stars else ""),
            ha="center",
            va="center",
            fontsize=6,
            color="white" if abs(values[row, column]) >= 0.65 * limit else "#111111",
        )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.055, pad=0.04)
    colorbar.set_label("Δ predicted phenoage")
    figure.tight_layout()
    save_vector_pair(figure, OUTPUT_DIR, "03_external_disease_heatmap")


def main() -> int:
    """Load formal artifacts, infer all models, age-correct, test, and plot."""
    frame = read_dataset(DATASET_PATH)
    biometa, biometa_features = load_published_biometaclock()
    baseline, baseline_features = load_trained_artifact("baseline")
    if biometa_features != list(EXTERNAL_COMPONENTS):
        raise ValueError("BioMeta artifact does not use the canonical ordered 17 components.")
    validate_dnam_feature_names(baseline_features)
    clocks = list(dict.fromkeys([*biometa_features, "dnamphenoage"]))
    clock_outputs = run_pyaging(frame, clocks).set_index("sampleID")
    biometa_x = numeric_matrix(
        clock_outputs.loc[frame["sampleID"]], biometa_features, dtype=np.float64
    )
    baseline_x = baseline_matrix(frame, baseline_features)
    predictions: list[pd.DataFrame] = []
    template = frame[["dataset", "disease", "sampleID", "age", "group"]].copy()
    for model, values in (
        ("bio_metaclock", biometa.predict(biometa_x)),
        ("baseline", baseline.predict(baseline_x)),
        ("dnamphenoage", clock_outputs.loc[frame["sampleID"], "dnamphenoage"].to_numpy(float)),
    ):
        output = template.copy()
        output["model"] = model
        output["pred_age"] = np.asarray(values, dtype=float)
        predictions.append(output)
    corrected, fits, tests = age_correct_and_test(pd.concat(predictions, ignore_index=True))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(OUTPUT_DIR / "external_disease_predictions.csv", index=False)
    fits.to_csv(OUTPUT_DIR / "external_disease_age_fits.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "external_disease_tests.csv", index=False)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "models": list(MODEL_ORDER),
                "biometa_weights": relative_path(biometa.weights_path),
                "biometa_metadata": relative_path(biometa.metadata_path),
                "baseline_artifact": relative_path(MODEL_DIR / "baseline.joblib"),
                "baseline_features": len(baseline_features),
                "baseline_features_present": sum(
                    feature in frame.columns for feature in baseline_features
                ),
                "baseline_features_zero_filled": sum(
                    feature not in frame.columns for feature in baseline_features
                ),
                "pyaging_api_clock": "dnamphenoage",
                "age_correction": "within dataset/model using pooled disease and control samples",
                "test": "two-sided Welch t-test; Bonferroni across three models",
                "software": software_versions(
                    ["numpy", "pandas", "scipy", "statsmodels", "pyaging", "anndata"]
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_heatmap(tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
