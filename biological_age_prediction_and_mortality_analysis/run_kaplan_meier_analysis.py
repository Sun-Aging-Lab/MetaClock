#!/usr/bin/env python3
"""Train five-fold OOF age models and draw age-corrected KM curves.

Edit only ``DATASET_PATH``.  The combined InCHIANTI table must include the
biological-age columns/CpGs and ``INTV_MOR``: wave 0 is alive/censored and
waves 4, 5, and 6 are deaths.  One deterministic baseline DNAm sample is used
per participant before the formal age-bin fold assignment.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EXTERNAL_COMPONENTS
from .plotting import LABELS, clean_axis, configure_style, save_vector_pair

# --- Paths: this is the only user-edited input ---

DATASET_PATH = Path("/path/to/your/file")  # fill in your path
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "kaplan_meier"
PYAGING_CACHE_DIR = Path(__file__).resolve().parent / "pyaging_data"


# --- Fixed formal parameters ---

TARGET = "clinical_phenoage"
COMPLETE_CASE = "clinical_phenoage_complete_case"
N_FOLDS = 5
SEED = 0
AGE_BIN_YEARS = 10
TRAIN_WAVE = 0.0
DEATH_WAVES = (4.0, 5.0, 6.0)
ALIVE_CENSOR_TIME = 6.0
TAIL_FRACTION = 0.20
ALPHAS = np.logspace(0, -5, 30)
TRAINED_MODELS = ("bio_metaclock", "baseline")
MODEL_ORDER = (*TRAINED_MODELS, "clinical_phenoage")
GROUP_ORDER = ("bottom_20", "top_20")
GROUP_COLORS = {"bottom_20": "#4C78A8", "top_20": "#E45756"}
REQUIRED_COLUMNS = {"ID", "sampleID", "age_years", TARGET, COMPLETE_CASE, "INTV_MOR"}


# --- Complete cases, deterministic baseline sample, and vital-status endpoint ---


def read_dataset(path: Path) -> pd.DataFrame:
    """Read one analysis-ready CSV/CSV.GZ or Parquet cohort."""
    if not path.exists():
        raise FileNotFoundError(f"Set DATASET_PATH to the combined InCHIANTI cohort: {path}")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def complete_case_mask(values: pd.Series) -> pd.Series:
    """Parse the clinical PhenoAge complete-case flag."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "t", "yes", "y"})


def prepare_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep valid rows, select lowest sampleID suffix per ID, and define FU4-6 survival."""
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise KeyError(f"KM dataset is missing columns: {sorted(missing)}")
    cohort = frame.loc[complete_case_mask(frame[COMPLETE_CASE])].copy()
    cohort["ID"] = cohort["ID"].astype(str)
    cohort["sampleID"] = cohort["sampleID"].astype(str)
    for column in ("age_years", TARGET, "INTV_MOR"):
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce")
    cohort = cohort.dropna(subset=["ID", "sampleID", "age_years", TARGET, "INTV_MOR"])
    suffix = pd.to_numeric(cohort["sampleID"].str.extract(r"(\d+)$", expand=False), errors="coerce")
    cohort["_suffix"] = suffix.fillna(np.inf)
    cohort = (
        cohort.sort_values(["ID", "_suffix", "sampleID"], kind="mergesort")
        .drop_duplicates("ID", keep="first")
        .drop(columns="_suffix")
        .reset_index(drop=True)
    )
    cohort = cohort[
        cohort["INTV_MOR"].eq(TRAIN_WAVE) | cohort["INTV_MOR"].isin(DEATH_WAVES)
    ].reset_index(drop=True)
    cohort["km_event"] = cohort["INTV_MOR"].isin(DEATH_WAVES).astype(int)
    cohort["km_time"] = np.where(cohort["km_event"].eq(1), cohort["INTV_MOR"], ALIVE_CENSOR_TIME)
    cohort["event_group"] = np.where(cohort["km_event"].eq(1), "vital", "control")
    if len(cohort) < N_FOLDS or cohort["km_event"].nunique() != 2:
        raise ValueError(
            "KM analysis needs at least five participants, deaths, and censored controls."
        )
    return cohort


def cpg_columns(frame: pd.DataFrame) -> list[str]:
    """Return CpG beta columns in file order."""
    columns = [str(column) for column in frame.columns if str(column).lower().startswith("cg")]
    if not columns:
        raise ValueError("No CpG beta columns named cg... were found.")
    return columns


# --- KM-specific 10-year age-bin outer folds ---


def assign_km_folds(cohort: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, pd.DataFrame]:
    """Randomized round-robin assignment within 10-year participant age bins."""
    table = cohort[["ID", "age_years", "km_event"]].copy()
    table["age_bin"] = (table["age_years"] // AGE_BIN_YEARS * AGE_BIN_YEARS).astype(int)
    folds = np.full(len(table), -1, dtype=int)
    rng = np.random.default_rng(seed)
    for _, subset in table.groupby("age_bin", sort=True):
        positions = subset.index.to_numpy(dtype=int)
        rng.shuffle(positions)
        labels = np.arange(len(positions), dtype=int) % N_FOLDS
        folds[positions] = labels
    summary = (
        table.assign(fold=folds + 1)
        .groupby(["fold", "age_bin"], as_index=False)
        .agg(n_ids=("ID", "nunique"), events=("km_event", "sum"))
    )
    for fold in range(N_FOLDS):
        subset = table[folds == fold]
        if subset.empty or subset["km_event"].nunique() != 2:
            raise ValueError(f"KM fold {fold + 1} must contain deaths and censored controls.")
    return folds, summary


# --- PyAging components ---


def component_predictions(cohort: pd.DataFrame, cpgs: list[str]) -> pd.DataFrame:
    """Reuse component columns or infer all 17 canonical BioMeta inputs with PyAging."""
    if set(EXTERNAL_COMPONENTS).issubset(cohort.columns):
        return cohort[["sampleID", *EXTERNAL_COMPONENTS]].copy()
    try:
        import anndata as ad
        import pyaging as pya
    except ImportError as exc:
        raise RuntimeError("Install anndata and pyaging to compute BioMeta inputs.") from exc
    matrix = cohort[cpgs].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    adata = ad.AnnData(matrix)
    adata.var_names = cpgs
    adata.obs = cohort[["sampleID", "age_years"]].copy().set_index("sampleID", drop=False)
    adata.obs["age"] = adata.obs["age_years"]
    adata.obs["Age"] = adata.obs["age_years"]
    PYAGING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pya.pred.predict_age(
        adata,
        list(EXTERNAL_COMPONENTS),
        dir=str(PYAGING_CACHE_DIR),
        batch_size=512,
        clean=True,
        verbose=False,
    )
    output = adata.obs[["sampleID", *EXTERNAL_COMPONENTS]].reset_index(drop=True)
    del adata, matrix
    gc.collect()
    return output


# --- KM-specific model fit: scaler + ENCV with max_iter=10000 ---


def fit_km_prediction_model(x: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    """Fit the formal KM predictor without a selected-alpha second refit."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "elasticnetcv",
                ElasticNetCV(
                    l1_ratio=0.5,
                    alphas=ALPHAS,
                    cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
                    max_iter=10_000,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def train_oof_predictions(
    cohort: pd.DataFrame,
    designs: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train both biological-age predictors in all outer folds and add Clinical PhenoAge."""
    folds, diagnostics = assign_km_folds(cohort)
    y = cohort[TARGET].to_numpy(float)
    outputs: list[pd.DataFrame] = []
    model_dir = OUTPUT_DIR / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        train, test = folds != fold, folds == fold
        for model_name in TRAINED_MODELS:
            model = fit_km_prediction_model(designs[model_name][train], y[train], SEED + fold)
            joblib.dump(model, model_dir / f"{model_name}_fold{fold + 1}.joblib")
            output = cohort.loc[
                test, ["ID", "sampleID", "age_years", TARGET, "INTV_MOR", "km_time", "km_event"]
            ].copy()
            output["fold"] = fold + 1
            output["model"] = model_name
            output["prediction"] = model.predict(designs[model_name][test])
            outputs.append(output)
        clinical = cohort.loc[
            test, ["ID", "sampleID", "age_years", TARGET, "INTV_MOR", "km_time", "km_event"]
        ].copy()
        clinical["fold"] = fold + 1
        clinical["model"] = "clinical_phenoage"
        clinical["prediction"] = y[test]
        outputs.append(clinical)
    predictions = pd.concat(outputs, ignore_index=True)
    for model in MODEL_ORDER:
        subset = predictions[predictions["model"].eq(model)]
        if len(subset) != len(cohort) or subset["ID"].nunique() != len(cohort):
            raise RuntimeError(f"{model} does not have one OOF/observed score per participant.")
    return predictions, diagnostics


# --- Pooled age correction, top/bottom 20%, KM, and log-rank ---


def age_correct_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove each model's pooled linear relationship between age and prediction."""
    corrected = predictions.copy()
    corrected["age_corrected_prediction"] = np.nan
    rows: list[dict[str, object]] = []
    for model, subset in corrected.groupby("model", sort=False):
        age = subset["age_years"].to_numpy(float)
        pred = subset["prediction"].to_numpy(float)
        fit = LinearRegression().fit(age[:, None], pred)
        residual = pred - fit.predict(age[:, None])
        corrected.loc[subset.index, "age_corrected_prediction"] = residual
        rows.append(
            {
                "model": model,
                "age_coef": float(fit.coef_[0]),
                "intercept": float(fit.intercept_),
                "r2_age_explained": float(fit.score(age[:, None], pred)),
                "n_fit": len(subset),
            }
        )
    return corrected, pd.DataFrame(rows)


def tail_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select deterministic lowest/highest 20% residuals within each model."""
    outputs: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        data = predictions[predictions["model"].eq(model)].sort_values(
            ["age_corrected_prediction", "sampleID"], kind="mergesort"
        )
        n_tail = int(np.floor(len(data) * TAIL_FRACTION))
        if n_tail < 1:
            raise ValueError(f"No KM tail samples for {model}.")
        low, high = data.head(n_tail).copy(), data.tail(n_tail).copy()
        low["bioage_group"], high["bioage_group"] = GROUP_ORDER
        outputs.extend([low, high])
    return pd.concat(outputs, ignore_index=True)


def fit_km_curves(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use lifelines KaplanMeierFitter and logrank_test for every age model."""
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError as exc:
        raise RuntimeError("Install lifelines to fit Kaplan–Meier curves.") from exc
    curves: list[pd.DataFrame] = []
    tests: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_data = assignments[assignments["model"].eq(model)]
        groups: dict[str, pd.DataFrame] = {}
        for group in GROUP_ORDER:
            data = model_data[model_data["bioage_group"].eq(group)]
            groups[group] = data
            fitter = KaplanMeierFitter(alpha=0.05, label=group)
            fitter.fit(data["km_time"], event_observed=data["km_event"])
            curve = fitter.survival_function_.rename(columns={group: "survival"}).reset_index()
            curve = curve.rename(columns={curve.columns[0]: "time"})
            confidence = fitter.confidence_interval_.reset_index(drop=True)
            curve["ci_lower"] = confidence.iloc[:, 0].to_numpy(float)
            curve["ci_upper"] = confidence.iloc[:, 1].to_numpy(float)
            curve.insert(0, "bioage_group", group)
            curve.insert(0, "model", model)
            curve["n_group"] = len(data)
            curve["n_events_group"] = int(data["km_event"].sum())
            curves.append(curve)
        low, high = groups[GROUP_ORDER[0]], groups[GROUP_ORDER[1]]
        result = logrank_test(
            low["km_time"],
            high["km_time"],
            event_observed_A=low["km_event"],
            event_observed_B=high["km_event"],
        )
        tests.append(
            {
                "model": model,
                "bottom_20_n": len(low),
                "bottom_20_events": int(low["km_event"].sum()),
                "top_20_n": len(high),
                "top_20_events": int(high["km_event"].sum()),
                "logrank_chi2": float(result.test_statistic),
                "logrank_p_value": float(result.p_value),
            }
        )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(tests)


# --- KM panel ---


def plot_km(curves: pd.DataFrame, tests: pd.DataFrame) -> None:
    """Draw top/bottom 20% age-corrected KM curves with 95% CIs."""
    configure_style()
    figure, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(150 / 25.4, 55 / 25.4), sharey=True)
    for index, (axis, model) in enumerate(zip(axes, MODEL_ORDER, strict=True)):
        for group in GROUP_ORDER:
            curve = curves[curves["model"].eq(model) & curves["bioage_group"].eq(group)]
            axis.step(
                curve["time"],
                curve["survival"],
                where="post",
                color=GROUP_COLORS[group],
                label=group.replace("_", " "),
            )
            axis.fill_between(
                curve["time"],
                curve["ci_lower"],
                curve["ci_upper"],
                step="post",
                color=GROUP_COLORS[group],
                alpha=0.16,
                linewidth=0,
            )
        p_value = float(tests.loc[tests["model"].eq(model), "logrank_p_value"].iloc[0])
        axis.set(
            title=f"{LABELS[model]}\nlog-rank p={p_value:.3g}",
            xlabel="Follow-up wave",
            xlim=(0, 6.05),
            ylim=(-0.03, 1.03),
        )
        if index == 0:
            axis.set_ylabel("Kaplan–Meier survival")
        axis.legend(frameon=False)
        clean_axis(axis)
    figure.tight_layout()
    save_vector_pair(figure, OUTPUT_DIR, "05_km_age_corrected_top_bottom20")


def main() -> int:
    """Prepare FU4-6 endpoint, train OOF predictors, age-correct, and plot KM."""
    cohort = prepare_cohort(read_dataset(DATASET_PATH))
    cpgs = cpg_columns(cohort)
    components = component_predictions(cohort, cpgs).set_index("sampleID")
    designs = {
        "bio_metaclock": components.loc[cohort["sampleID"], list(EXTERNAL_COMPONENTS)].to_numpy(
            float
        ),
        "baseline": (
            cohort[cpgs].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
        ),
    }
    predictions, diagnostics = train_oof_predictions(cohort, designs)
    corrected, age_fits = age_correct_predictions(predictions)
    assignments = tail_groups(corrected)
    curves, tests = fit_km_curves(assignments)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(OUTPUT_DIR / "km_fold_diagnostics.csv", index=False)
    corrected.to_csv(OUTPUT_DIR / "km_oof_predictions.csv", index=False)
    age_fits.to_csv(OUTPUT_DIR / "km_age_correction_fits.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / "km_top_bottom20_assignments.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "km_curve_points.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "km_logrank_tests.csv", index=False)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "endpoint": "INTV_MOR: 0 alive/censored; death waves 4, 5, 6",
                "outer_folds": "participant-level randomized round-robin within 10-year age bins",
                "prediction_model": (
                    "StandardScaler + ElasticNetCV; max_iter=10000; no second refit"
                ),
                "age_correction": "pooled OOF linear residual within model",
                "tails": "deterministic bottom/top 20% within model",
                "models": list(MODEL_ORDER),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_km(curves, tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
