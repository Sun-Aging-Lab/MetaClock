#!/usr/bin/env python3
"""Build age-corrected Kaplan-Meier curves and log-rank tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LinearRegression


FIGURE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PARTICIPANTS = FIGURE_DIR / "datasets/inchianti/km_analysis_participants.csv"
DEFAULT_OUTPUT_DIR = FIGURE_DIR / "results"
TAIL_FRACTION = 0.20
MODELS = ("bio_metaclock", "baseline", "clinical_phenoage")
MODEL_LABELS = {
    "bio_metaclock": "BioMeta",
    "baseline": "Baseline",
    "clinical_phenoage": "PhenoAge",
}
GROUP_STYLES = {
    "bottom_20": ("20% lowest bioage", "#5E3C99"),
    "top_20": ("20% highest bioage", "#1B9E77"),
}
MM_TO_INCH = 1.0 / 25.4
FIGSIZE = (150 * MM_TO_INCH, 50 * MM_TO_INCH)


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    """Raise a readable error when an input table is incomplete."""
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def assign_age_corrected_tails(
    participants: pd.DataFrame,
    score: str,
    tail_fraction: float = TAIL_FRACTION,
) -> pd.DataFrame:
    """Residualize one score on age and select stable top/bottom tails."""
    if not 0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be between 0 and 0.5")
    output = participants.copy()
    age = pd.to_numeric(output["age_years"], errors="raise").to_numpy(float)
    values = pd.to_numeric(output[score], errors="raise").to_numpy(float)
    if not np.isfinite([age, values]).all():
        raise ValueError(f"{score} age correction contains non-finite values")
    age_fit = LinearRegression().fit(age.reshape(-1, 1), values)
    output["corrected_score"] = values - age_fit.predict(age.reshape(-1, 1))
    output["bioage_group"] = ""
    n_tail = int(len(output) * tail_fraction)
    if n_tail < 1:
        raise ValueError("Not enough participants for the requested KM tails")
    ordered = output.sort_values(["corrected_score", "participant_id"], kind="mergesort")
    output.loc[ordered.index[:n_tail], "bioage_group"] = "bottom_20"
    output.loc[ordered.index[-n_tail:], "bioage_group"] = "top_20"
    return output


def kaplan_meier_curve(rows: pd.DataFrame) -> pd.DataFrame:
    """Return event-time survival estimates and 95% log-log intervals."""
    # This Figure 4 endpoint is observed at follow-up waves (4, 5, and 6),
    # not at continuously measured survival times in years.
    durations = rows["followup_wave"].to_numpy(float)
    observed = rows["event"].to_numpy(int).astype(bool)
    model = KaplanMeierFitter(alpha=0.05)
    model.fit(durations, event_observed=observed, label="survival")
    survival = model.survival_function_.rename(columns={"survival": "survival"})
    confidence = model.confidence_interval_survival_function_.copy()
    confidence.columns = ["ci_lower", "ci_upper"]
    event_table = model.event_table[["at_risk", "observed"]].rename(
        columns={"observed": "n_events"}
    )
    curve = survival.join(confidence).join(event_table)
    event_times = np.unique(durations[observed])
    curve = curve.loc[curve.index.isin(np.concatenate(([0.0], event_times)))]
    curve.index.name = "time"
    return curve.reset_index().rename(columns={"at_risk": "n_at_risk"})


def logrank_summary(assignments: pd.DataFrame) -> dict[str, float | int | str]:
    """Run the top-versus-bottom log-rank test and return audit fields."""
    tails = assignments[assignments["bioage_group"].isin(GROUP_STYLES)].copy()
    bottom = tails["bioage_group"].eq("bottom_20").to_numpy()
    top = tails["bioage_group"].eq("top_20").to_numpy()
    durations = tails["followup_wave"].to_numpy(float)
    observed = tails["event"].to_numpy(int).astype(bool)
    result = logrank_test(
        durations[bottom],
        durations[top],
        event_observed_A=observed[bottom],
        event_observed_B=observed[top],
    )
    observed_top = 0.0
    expected_top = 0.0
    variance = 0.0
    for time in np.sort(np.unique(durations[observed])):
        at_risk = durations >= time
        deaths = observed & (durations == time)
        n_total = int(at_risk.sum())
        n_deaths = int(deaths.sum())
        if n_total <= 1 or n_deaths == 0:
            continue
        n_top = int((at_risk & top).sum())
        observed_top += float((deaths & top).sum())
        expected_top += n_deaths * n_top / n_total
        variance += (
            n_top
            * (n_total - n_top)
            * n_deaths
            * (n_total - n_deaths)
            / (n_total * n_total * (n_total - 1))
        )
    return {
        "grouping": "age_corrected",
        "bottom_20_n": int(bottom.sum()),
        "bottom_20_events": int(observed[bottom].sum()),
        "top_20_n": int(top.sum()),
        "top_20_events": int(observed[top].sum()),
        "observed_top_events": observed_top,
        "expected_top_events": expected_top,
        "logrank_variance": variance,
        "logrank_chi2": float(result.test_statistic),
        "logrank_p_value": float(result.p_value),
    }


def build_km_results(participants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create curve-point and log-rank tables for all three scores."""
    curve_frames: list[pd.DataFrame] = []
    logrank_rows: list[dict[str, float | int | str]] = []
    for score in MODELS:
        assignments = assign_age_corrected_tails(participants, score)
        for group in GROUP_STYLES:
            rows = assignments[assignments["bioage_group"].eq(group)]
            curve = kaplan_meier_curve(rows)
            curve.insert(0, "bioage_group", group)
            curve.insert(0, "model", score)
            curve.insert(0, "grouping", "age_corrected")
            curve["n_group"] = len(rows)
            curve["n_events_group"] = int(rows["event"].sum())
            curve_frames.append(curve)
        logrank_rows.append({"model": score, **logrank_summary(assignments)})
    return pd.concat(curve_frames, ignore_index=True), pd.DataFrame(logrank_rows)


def format_p_value(value: float) -> str:
    """Format a log-rank p-value for panel annotation."""
    return f"p={value:.1e}" if value < 0.001 else f"p={value:.3f}"


def plot_km(curves: pd.DataFrame, logrank: pd.DataFrame, output_dir: Path) -> None:
    """Plot panel 5 KM curves for the three age-corrected scores."""
    figure, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS, strict=True):
        values = curves[curves["model"].eq(model)]
        for group, (label, color) in GROUP_STYLES.items():
            curve = values[values["bioage_group"].eq(group)].sort_values("time")
            axis.step(curve["time"], curve["survival"], where="post", color=color, label=label)
            axis.fill_between(
                curve["time"], curve["ci_lower"], curve["ci_upper"],
                step="post", color=color, alpha=0.14, linewidth=0,
            )
        p_value = float(logrank.loc[logrank["model"].eq(model), "logrank_p_value"].iloc[0])
        axis.text(0.04, 0.08, f"log-rank {format_p_value(p_value)}", transform=axis.transAxes)
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Follow-up wave")
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel("Survival probability")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "05_km_curves.pdf")
    figure.savefig(output_dir / "05_km_curves.svg")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse the relative InCHIANTI path and optional replacement path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, default=DEFAULT_PARTICIPANTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Build panel 5 and write curve-point and log-rank result tables."""
    args = parse_args()
    participants = pd.read_csv(args.participants)
    require_columns(
        participants,
        {"participant_id", "followup_wave", "event", "age_years", *MODELS},
        args.participants,
    )
    if participants["participant_id"].duplicated().any():
        raise ValueError("KM input must contain one row per participant")
    if not participants["event"].isin([0, 1]).all():
        raise ValueError("event must contain only 0 and 1")
    curves, logrank = build_km_results(participants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logrank.to_csv(args.output_dir / "05a_km_logrank.csv", index=False)
    curves.to_csv(args.output_dir / "05b_km_curve_points.csv", index=False)
    plot_km(curves, logrank, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
