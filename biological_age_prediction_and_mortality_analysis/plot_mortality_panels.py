"""Render standardized Cox-effect and Kaplan-Meier mortality panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .config import SURVIVAL_MODELS
    from .plotting import COLORS, LABELS, clean_axis, ordered_rows, save_vector_pair
except ImportError:  # Direct-script fallback.
    from config import SURVIVAL_MODELS
    from plotting import COLORS, LABELS, clean_axis, ordered_rows, save_vector_pair


COX_PANELS = (
    ("prediction_plus_chronological_age_sex", "Age + sex"),
    ("prediction_plus_sex", "Sex"),
    ("prediction_plus_chronological_age", "Age"),
)


def plot_standardized_cox_effects(source_dir: Path, output_dir: Path) -> None:
    """Plot standardized prediction coefficients for the three Cox specifications."""
    source = pd.read_csv(source_dir / "standardized_cox_effects_source_data.csv")
    expected = {(panel, model) for panel, _ in COX_PANELS for model in SURVIVAL_MODELS}
    observed = set(source[["panel", "model"]].itertuples(index=False, name=None))
    if source.duplicated(["panel", "model"]).any() or observed != expected:
        raise ValueError("Cox source must contain one row for every configured panel/model pair.")
    figure, axes = plt.subplots(1, 3, figsize=(90 / 25.4, 50 / 25.4), sharey=True)
    figure.subplots_adjust(left=0.30, right=0.98, bottom=0.28, top=0.84, wspace=0.28)
    y = np.arange(len(SURVIVAL_MODELS))
    for index, (axis, (panel, title)) in enumerate(zip(axes, COX_PANELS, strict=True)):
        rows = ordered_rows(source[source["panel"].eq(panel)], SURVIVAL_MODELS)
        for row, model in enumerate(SURVIVAL_MODELS):
            estimate = float(rows.loc[row, "coefficient"])
            lower, upper = float(rows.loc[row, "ci_lower"]), float(rows.loc[row, "ci_upper"])
            if not lower <= estimate <= upper:
                raise ValueError(f"Invalid Cox confidence interval for {panel}/{model}.")
            axis.errorbar(
                estimate,
                y[row],
                xerr=[[estimate - lower], [upper - estimate]],
                fmt="o",
                color=COLORS[model],
                markersize=2.8,
                linewidth=0.8,
                capsize=1.5,
            )
            axis.text(0.25, y[row] + (0.24 if row < 2 else -0.24), f"β={estimate:.2f}", fontsize=6)
        axis.axvline(0, color="#777777", linestyle="--", linewidth=0.5)
        axis.set(title=title, xlim=(-0.29, 2.23), xticks=[0, 2], yticks=y)
        if index == 0:
            axis.set_yticklabels([LABELS[model] for model in SURVIVAL_MODELS])
        clean_axis(axis, grid=False)
    axes[0].invert_yaxis()
    figure.supxlabel("Log Hazard Ratio", y=0.07, fontsize=9.75)
    save_vector_pair(figure, output_dir, "cox_coefficients_standardized")


def format_pvalue(value: float) -> str:
    """Format log-rank P values compactly."""
    return f"logrank p={value:.1e}" if value < 0.001 else f"logrank p={value:.3f}"


def survival_at(curve: pd.DataFrame, time: float) -> float:
    """Evaluate a right-continuous Kaplan-Meier step function at a follow-up wave."""
    eligible = curve[curve["time"].le(time)].sort_values("time")
    return 1.0 if eligible.empty else float(eligible["survival"].iloc[-1])


def plot_kaplan_meier_survival(source_dir: Path, output_dir: Path) -> None:
    """Plot floor-defined bottom/top 20% age-corrected survival curves."""
    curves = pd.read_csv(source_dir / "kaplan_meier_curves_source_data.csv")
    tests = pd.read_csv(source_dir / "kaplan_meier_logrank_source_data.csv")
    expected_pairs = {
        (model, group) for model in SURVIVAL_MODELS for group in ("bottom_20", "top_20")
    }
    observed_pairs = set(
        curves[["model", "bioage_group"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_pairs != expected_pairs or set(tests["model"]) != set(SURVIVAL_MODELS):
        raise ValueError(
            "Kaplan-Meier source does not contain the configured models and tail groups."
        )
    figure, axes = plt.subplots(1, 3, figsize=(100 / 25.4, 50 / 25.4), sharey=True)
    figure.subplots_adjust(left=0.12, right=0.961, bottom=0.23, top=0.70, wspace=0.18)
    group_colors = {"bottom_20": "#5E3C99", "top_20": "#1B9E77"}
    handles: list[plt.Line2D] = []
    for axis, model in zip(axes, SURVIVAL_MODELS, strict=True):
        for group in ("bottom_20", "top_20"):
            subset = curves[
                curves["model"].eq(model) & curves["bioage_group"].eq(group)
            ].sort_values("time")
            line = axis.step(
                subset["time"],
                subset["survival"],
                where="post",
                color=group_colors[group],
                linewidth=0.5,
            )[0]
            axis.fill_between(
                subset["time"],
                subset["ci_lower"],
                subset["ci_upper"],
                step="post",
                color=group_colors[group],
                alpha=0.14,
                linewidth=0,
            )
            if model == SURVIVAL_MODELS[0]:
                handles.append(line)
        p_values = tests.loc[tests["model"].eq(model), "logrank_p_value"]
        if len(p_values) != 1 or not 0 <= float(p_values.iloc[0]) <= 1:
            raise ValueError(f"Expected one valid log-rank P value for {model}.")
        model_curves = curves[curves["model"].eq(model)]
        low = model_curves[model_curves["bioage_group"].eq("bottom_20")]
        high = model_curves[model_curves["bioage_group"].eq("top_20")]
        waves = [
            "ΔW"
            f"{wave}: "
            f"{int(np.rint(100 * (survival_at(low, wave) - survival_at(high, wave)))):+d} pp"
            for wave in (4, 5, 6)
        ]
        axis.text(
            0.25, 0.12, "\n".join([format_pvalue(float(p_values.iloc[0])), *waves]), fontsize=6
        )
        axis.set(title=LABELS[model], xlim=(0, 6.3), ylim=(0, 1.03), xticks=[0, 2, 4, 6])
        clean_axis(axis, grid=False)
    axes[0].set_ylabel("Survival")
    figure.supxlabel("Follow-up Waves", y=0.03, fontsize=9.75)
    figure.legend(
        handles,
        ["20% lowest bioage", "20% highest bioage"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    save_vector_pair(figure, output_dir, "kaplan_meier_survival")
