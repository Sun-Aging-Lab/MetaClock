#!/usr/bin/env python3
"""Draw age/sex-corrected KM curves from the shared primary OOF predictions.

Run the biological-age prediction workflow first. Deaths use follow-up waves
4, 5 and 6; surviving participants are censored at wave 6.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import (
    INCHIANTI_PARTICIPANTS,
    OOF_COMPONENTS,
)
from .data import (
    complete_case_mask as parse_complete_case_mask,
)
from .data import (
    read_analysis_dataset,
)
from .plotting import LABELS, clean_axis, configure_style, save_vector_pair
from .provenance import software_versions
from .production_oof import PRODUCTION_CONFIG_ID, production_oof_folds
from .run_cox_analysis import load_survival_predictions

# --- Repository-local generated input, outputs, and cache ---

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "preprocessing"
    / "inchianti_analysis_ready.parquet"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "kaplan_meier"


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
TRAINED_MODELS = ("bio_metaclock", "baseline")
MODEL_ORDER = (*TRAINED_MODELS, "clinical_phenoage")
GROUP_ORDER = ("bottom_20", "top_20")
GROUP_COLORS = {"bottom_20": "#C96B6B", "top_20": "#5B7FAE"}
GROUP_LABELS = {
    "bottom_20": "Lowest 20% (age/sex adjusted)",
    "top_20": "Highest 20% (age/sex adjusted)",
}
REQUIRED_COLUMNS = {
    "ID", "sampleID", "age_years", TARGET, COMPLETE_CASE, "INTV_MOR", "sex_male"
}


# --- Complete cases, deterministic baseline sample, and vital-status endpoint ---


def read_dataset(path: Path) -> pd.DataFrame:
    """Read one analysis-ready CSV/CSV.GZ or Parquet cohort."""
    return read_analysis_dataset(
        path, columns=sorted(REQUIRED_COLUMNS), float_precision="round_trip"
    )


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
    cohort["_numeric_id"] = pd.to_numeric(cohort["ID"], errors="raise")
    suffix = pd.to_numeric(cohort["sampleID"].str.extract(r"(\d+)$", expand=False), errors="coerce")
    cohort["_suffix"] = suffix.fillna(np.inf)
    cohort = (
        cohort.sort_values(["_numeric_id", "_suffix", "sampleID"], kind="mergesort")
        .drop_duplicates("ID", keep="first")
        .drop(columns=["_numeric_id", "_suffix"])
        .reset_index(drop=True)
    )
    cohort = cohort[
        cohort["INTV_MOR"].eq(TRAIN_WAVE) | cohort["INTV_MOR"].isin(DEATH_WAVES)
    ].reset_index(drop=True)
    if not cohort["sex_male"].isin([0, 1]).all():
        raise ValueError("KM age/sex correction requires sex_male coded 0 or 1 for every participant.")
    cohort["km_event"] = cohort["INTV_MOR"].isin(DEATH_WAVES).astype(int)
    cohort["km_time"] = np.where(cohort["km_event"].eq(1), cohort["INTV_MOR"], ALIVE_CENSOR_TIME)
    cohort["event_group"] = np.where(cohort["km_event"].eq(1), "vital", "control")
    if len(cohort) < N_FOLDS or cohort["km_event"].nunique() != 2:
        raise ValueError(
            "KM analysis needs at least five participants, deaths, and censored controls."
        )
    return cohort


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


# --- Pooled age/sex correction, top/bottom 20%, KM, and log-rank ---


def age_sex_correct_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit pooled prediction ~ 1 + age_years + female separately for each model."""
    corrected = predictions.copy()
    corrected["age_sex_corrected_prediction"] = np.nan
    if not corrected["sex_male"].isin([0, 1]).all():
        raise ValueError("KM age/sex correction requires sex_male coded 0 or 1.")
    rows: list[dict[str, object]] = []
    for model, subset in corrected.groupby("model", sort=False):
        covariates = subset[["age_years"]].assign(female=1.0 - subset["sex_male"])
        pred = subset["prediction"].to_numpy(float)
        fit = LinearRegression().fit(covariates, pred)
        residual = pred - fit.predict(covariates)
        corrected.loc[subset.index, "age_sex_corrected_prediction"] = residual
        rows.append(
            {
                "model": model,
                "age_coef": float(fit.coef_[0]),
                "female_coef": float(fit.coef_[1]),
                "intercept": float(fit.intercept_),
                "r2_age_sex_explained": float(fit.score(covariates, pred)),
                "n_fit": len(subset),
            }
        )
    return corrected, pd.DataFrame(rows)


def tail_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select deterministic lowest/highest 20% residuals within each model."""
    outputs: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        data = predictions[predictions["model"].eq(model)].sort_values(
            ["age_sex_corrected_prediction", "sampleID"], kind="mergesort"
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
            curve.insert(0, "grouping", "age_sex_corrected")
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
                "grouping": "age_sex_corrected",
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
    """Draw the manuscript age/sex-corrected KM curves with 95% CIs."""
    configure_style()
    figure, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(100 / 25.4, 50 / 25.4), sharey=True)
    figure.subplots_adjust(left=0.12, right=0.961, bottom=0.23, top=0.70, wspace=0.18)
    labels = {**LABELS, "clinical_phenoage": "Biological Age"}
    handles = []
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        model_curves = curves[curves["model"].eq(model)]
        for group in GROUP_ORDER:
            curve = model_curves[model_curves["bioage_group"].eq(group)].sort_values("time")
            line = axis.step(
                curve["time"],
                curve["survival"],
                where="post",
                color=GROUP_COLORS[group],
                linewidth=0.5,
                label=GROUP_LABELS[group],
            )[0]
            axis.fill_between(
                curve["time"],
                curve["ci_lower"],
                curve["ci_upper"],
                step="post",
                color=GROUP_COLORS[group],
                alpha=0.14,
                linewidth=0,
            )
            if model == MODEL_ORDER[0]:
                handles.append(line)
        p_value = float(tests.loc[tests["model"].eq(model), "logrank_p_value"].iloc[0])
        wave_lines = []
        for wave in (4, 5, 6):
            survival = {}
            for group in GROUP_ORDER:
                group_curve = model_curves[model_curves["bioage_group"].eq(group)].sort_values("time")
                survival[group] = float(group_curve.loc[group_curve["time"].le(wave), "survival"].iloc[-1])
            difference = int(np.rint(100 * (survival["bottom_20"] - survival["top_20"])))
            wave_lines.append(f"ΔW{wave}: {difference:+d} pp")
        axis.text(
            0.25, 0.12, "\n".join([f"logrank p={p_value:.2e}", *wave_lines]),
            ha="left", va="bottom", fontsize=6,
        )
        axis.set_title(labels[model], pad=3)
        axis.set(xlim=(0, 6.3), ylim=(0, 1.03), xticks=[0, 2, 4, 6], yticks=np.arange(0, 1.01, 0.2))
        axis.tick_params(axis="both", labelsize=8)
        clean_axis(axis, grid=False)
    axes[0].set_ylabel("Survival")
    figure.supxlabel("Follow-up Waves", y=0.03, fontsize=9.75)
    figure.legend(
        handles, [GROUP_LABELS[group] for group in GROUP_ORDER],
        loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2,
        frameon=False, fontsize=6, handlelength=2.5, columnspacing=2.5,
    )
    save_vector_pair(figure, OUTPUT_DIR, "05_km_curves")


def main() -> int:
    """Load shared primary OOF scores, remove age/sex effects, and plot."""
    cohort = prepare_cohort(read_dataset(DATASET_PATH))
    if len(cohort) != INCHIANTI_PARTICIPANTS:
        raise ValueError(
            f"Figure 4 Kaplan-Meier requires {INCHIANTI_PARTICIPANTS} participants."
        )
    predictions = load_survival_predictions(cohort)
    diagnostics = summarize_km_folds(cohort, production_oof_folds(cohort))
    corrected, age_fits = age_sex_correct_predictions(predictions)
    assignments = tail_groups(corrected)
    curves, tests = fit_km_curves(assignments)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(OUTPUT_DIR / "km_fold_diagnostics.csv", index=False)
    corrected.to_csv(OUTPUT_DIR / "km_oof_predictions.csv", index=False)
    age_fits.to_csv(OUTPUT_DIR / "km_age_sex_correction_fits.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / "km_top_bottom20_assignments.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "km_curve_points.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "km_logrank_tests.csv", index=False)
    source_metadata = json.dumps(
        {
            "endpoint": "INTV_MOR: 0 alive/censored; death waves 4, 5, 6",
            "prediction_config_id": PRODUCTION_CONFIG_ID,
            "prediction_source": "shared biological_age_prediction/oof_predictions.csv",
            "outer_folds": "primary participants; age-decade by sex; StratifiedKFold(5, shuffle=True, random_state=0)",
            "prediction_model": "19-component MetaClock and full-CpG baseline; shared fold-local OOF predictions",
            "training_order": "numeric participant ID, numeric sample suffix",
            "age_sex_correction": "pooled prediction ~ 1 + age_years + female within model",
            "tails": "deterministic bottom/top 20% within model",
            "models": list(MODEL_ORDER),
            "bio_metaclock_components": list(OOF_COMPONENTS),
            "software": software_versions(
                ["numpy", "pandas", "scikit-learn", "lifelines", "pyaging", "anndata"]
            ),
        },
        indent=2,
    ) + "\n"
    for filename in ("run_summary.json", "figure_source.json"):
        (OUTPUT_DIR / filename).write_text(source_metadata, encoding="utf-8")
    plot_km(curves, tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
