#!/usr/bin/env python3
"""Train five-fold OOF age models and draw age-corrected KM curves.

Run ``prepare_inchianti_dataset`` first; ``DATASET_PATH`` points to its output.
The combined InCHIANTI table must include the biological-age columns/CpGs
and ``INTV_MOR``: wave 0 is alive/censored and waves 4, 5, and 6 are deaths. One deterministic baseline DNAm sample is used
per participant and mapped to the fixed formal Figure 4 OOF assignment.
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

from repository_paths import relative_path

from .config import (
    INCHIANTI_OOF_FOLD_ASSIGNMENT_PATH,
    INCHIANTI_PARTICIPANTS,
    OOF_COMPONENTS,
)
from .data import (
    complete_case_mask as parse_complete_case_mask,
)
from .data import (
    dnam_feature_columns,
    fixed_figure4_oof_folds,
    numeric_matrix,
    read_analysis_dataset,
)
from .plotting import LABELS, clean_axis, configure_style, save_vector_pair
from .provenance import software_versions

# --- Repository-local generated input, outputs, and cache ---

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "preprocessing"
    / "inchianti_analysis_ready.parquet"
)
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
GROUP_COLORS = {"bottom_20": "#E64B35", "top_20": "#4DBBD5"}
REQUIRED_COLUMNS = {
    "ID", "sampleID", "age_years", TARGET, COMPLETE_CASE, "INTV_MOR", "sex_male"
}


# --- Complete cases, deterministic baseline sample, and vital-status endpoint ---


def read_dataset(path: Path) -> pd.DataFrame:
    """Read one analysis-ready CSV/CSV.GZ or Parquet cohort."""
    return read_analysis_dataset(path)


def complete_case_mask(values: pd.Series) -> pd.Series:
    """Parse the clinical PhenoAge complete-case flag."""
    return parse_complete_case_mask(values)


def prepare_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep valid rows, select lowest sampleID suffix per ID, and define FU4-6 survival."""
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise KeyError(f"KM dataset is missing columns: {sorted(missing)}")
    cohort = frame.loc[complete_case_mask(frame[COMPLETE_CASE])].copy()
    cohort["ID"] = cohort["ID"].astype(str)
    cohort["sampleID"] = cohort["sampleID"].astype(str)
    for column in ("age_years", TARGET, "INTV_MOR", "sex_male"):
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
    """Return all 485,577 formal baseline features in file order."""
    return dnam_feature_columns(frame)


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
    summary = summarize_km_folds(cohort, folds)
    return folds, summary


def summarize_km_folds(cohort: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Validate and summarize an explicit participant-level KM fold assignment."""
    folds = np.asarray(folds, dtype=int)
    if len(folds) != len(cohort) or set(folds) != set(range(N_FOLDS)):
        raise ValueError("KM OOF folds must cover every cohort row with labels 0 through 4.")
    if cohort.assign(fold=folds).groupby("ID")["fold"].nunique().max() != 1:
        raise ValueError("KM OOF fold assignment splits samples from the same donor.")
    table = cohort[["ID", "age_years", "km_event"]].copy()
    table["age_bin"] = (table["age_years"] // AGE_BIN_YEARS * AGE_BIN_YEARS).astype(int)
    summary = (
        table.assign(fold=folds + 1)
        .groupby(["fold", "age_bin"], as_index=False)
        .agg(n_ids=("ID", "nunique"), events=("km_event", "sum"))
    )
    for fold in range(N_FOLDS):
        subset = table[folds == fold]
        if subset.empty or subset["km_event"].nunique() != 2:
            raise ValueError(f"KM fold {fold + 1} must contain deaths and censored controls.")
    return summary


# --- PyAging components ---


def component_predictions(cohort: pd.DataFrame, cpgs: list[str]) -> pd.DataFrame:
    """Reuse component columns or infer all 19 formal OOF BioMeta inputs."""
    if set(OOF_COMPONENTS).issubset(cohort.columns):
        return cohort[["sampleID", *OOF_COMPONENTS]].copy()
    try:
        import anndata as ad
        import pyaging as pya
    except ImportError as exc:
        raise RuntimeError("Install anndata and pyaging to compute BioMeta inputs.") from exc
    matrix = numeric_matrix(cohort, cpgs)
    adata = ad.AnnData(matrix)
    adata.var_names = cpgs
    adata.obs = cohort[["sampleID", "age_years"]].copy().set_index("sampleID", drop=False)
    adata.obs["age"] = adata.obs["age_years"]
    adata.obs["Age"] = adata.obs["age_years"]
    female = 1.0 - cohort["sex_male"].to_numpy(float)
    adata.obs["female"] = female
    adata.obs["Female"] = female
    PYAGING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pya.pred.predict_age(
        adata,
        list(OOF_COMPONENTS),
        dir=relative_path(PYAGING_CACHE_DIR),
        batch_size=512,
        clean=True,
        verbose=False,
    )
    output = adata.obs[["sampleID", *OOF_COMPONENTS]].reset_index(drop=True)
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
    *,
    folds: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train both biological-age predictors in all outer folds and add Clinical PhenoAge."""
    if folds is None:
        folds, diagnostics = assign_km_folds(cohort)
    else:
        folds = np.asarray(folds, dtype=int)
        diagnostics = summarize_km_folds(cohort, folds)
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
            curve.insert(0, "grouping", "age_corrected")
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
        observed_top, expected_top, variance = logrank_supporting_statistics(low, high)
        tests.append(
            {
                "grouping": "age_corrected",
                "model": model,
                "bottom_20_n": len(low),
                "bottom_20_events": int(low["km_event"].sum()),
                "top_20_n": len(high),
                "top_20_events": int(high["km_event"].sum()),
                "observed_top_events": observed_top,
                "expected_top_events": expected_top,
                "logrank_variance": variance,
                "logrank_chi2": float(result.test_statistic),
                "logrank_p_value": float(result.p_value),
            }
        )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(tests)


def logrank_supporting_statistics(
    low: pd.DataFrame, high: pd.DataFrame
) -> tuple[float, float, float]:
    """Return O/E/variance fields that lifelines omits from its result object."""
    time = pd.concat([low["km_time"], high["km_time"]], ignore_index=True).to_numpy(float)
    event = pd.concat([low["km_event"], high["km_event"]], ignore_index=True).to_numpy(bool)
    top = np.concatenate(
        [np.zeros(len(low), dtype=bool), np.ones(len(high), dtype=bool)]
    )
    observed_top = 0.0
    expected_top = 0.0
    variance = 0.0
    for event_time in np.sort(np.unique(time[event])):
        at_risk = time >= event_time
        events_at_time = event & (time == event_time)
        n_total = int(at_risk.sum())
        d_total = int(events_at_time.sum())
        if n_total <= 1 or d_total == 0:
            continue
        n_top = int((at_risk & top).sum())
        observed_top += float((events_at_time & top).sum())
        expected_top += float(d_total * n_top / n_total)
        variance += float(
            n_top
            * (n_total - n_top)
            * d_total
            * (n_total - d_total)
            / (n_total * n_total * (n_total - 1))
        )
    return observed_top, expected_top, variance


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
    if len(cohort) != INCHIANTI_PARTICIPANTS:
        raise ValueError(
            f"Figure 4 Kaplan-Meier requires {INCHIANTI_PARTICIPANTS} participants."
        )
    cpgs = cpg_columns(cohort)
    components = component_predictions(cohort, cpgs).set_index("sampleID")
    designs = {
        "bio_metaclock": numeric_matrix(
            components.loc[cohort["sampleID"]], OOF_COMPONENTS, dtype=np.float64
        ),
        "baseline": numeric_matrix(cohort, cpgs),
    }
    folds = fixed_figure4_oof_folds(cohort)
    predictions, diagnostics = train_oof_predictions(cohort, designs, folds=folds)
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
                "outer_folds": "fixed formal Figure 4 donor-level OOF assignment",
                "oof_fold_assignment": {
                    "path": relative_path(INCHIANTI_OOF_FOLD_ASSIGNMENT_PATH),
                    "assignment_labels_internal": list(range(N_FOLDS)),
                    "output_fold_labels": list(range(1, N_FOLDS + 1)),
                },
                "prediction_model": (
                    "StandardScaler + ElasticNetCV; max_iter=10000; no second refit"
                ),
                "age_correction": "pooled OOF linear residual within model",
                "tails": "deterministic bottom/top 20% within model",
                "models": list(MODEL_ORDER),
                "bio_metaclock_components": list(OOF_COMPONENTS),
                "software": software_versions(
                    ["numpy", "pandas", "scikit-learn", "lifelines", "pyaging", "anndata"]
                ),
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
