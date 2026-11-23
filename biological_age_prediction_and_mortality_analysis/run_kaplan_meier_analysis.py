"""Plot FHS and WHI survival curves for the lowest and highest residual quintiles."""

from pathlib import Path

from .config import MORTALITY_MODELS
from .plotting import clean_axis, configure_style, ordered_rows, save_vector_pair

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "kaplan_meier"
GROUP_ORDER = ("bottom_20", "top_20")
GROUP_COLORS = {"bottom_20": "#C96B6B", "top_20": "#5B7FAE"}


def plot_km(cohort: str, curves: pd.DataFrame, tests: pd.DataFrame) -> None:
    models = MORTALITY_MODELS[cohort]
    stats = ordered_rows(tests, models)
    expected = {(model, group) for model in models for group in GROUP_ORDER}
    if (set(curves["cohort"]) != {cohort}
            or set(zip(curves["model"], curves["group"])) != expected
            or curves.duplicated(["model", "group", "time_years"]).any()):
        raise ValueError(f"Expected unique {cohort.upper()} curve points for each model and quintile.")

    configure_style()
    figure, axes = plt.subplots(1, len(models), figsize=(100 / 25.4, 50 / 25.4), sharey=True)
    figure.subplots_adjust(left=0.12, right=0.961, bottom=0.23, top=0.70, wspace=0.18)
    xmax = np.ceil(curves["time_years"].max() * 1.05)
    xticks = [0, 5, 10, 15] if cohort == "fhs" else [0, 5, 10, 15, 20]
    handles = []
    for axis, row in zip(axes, stats.itertuples(index=False), strict=True):
        model_points = curves.loc[curves["model"].eq(row.model)]
        for group in GROUP_ORDER:
            curve = model_points.loc[model_points["group"].eq(group)].sort_values("time_years")
            color = GROUP_COLORS[group]
            line, = axis.step(curve.time_years, curve.survival, where="post", color=color, linewidth=0.5)
            axis.fill_between(curve.time_years, curve.ci_low, curve.ci_high, step="post",
                              color=color, alpha=0.14, linewidth=0)
            if axis is axes[0]:
                handles.append(line)
        axis.text(0.25 / 6.3, 0.12 / 1.03, f"logrank p={row.logrank_p:.2e}\nΔDeaths: {row.delta_deaths_highest_minus_lowest:+d}",
                  transform=axis.transAxes, ha="left", va="bottom", fontsize=6)
        axis.set_title(row.title, pad=3)
        axis.set(xlim=(0, xmax), ylim=(0, 1.03), xticks=xticks, yticks=np.arange(0, 1.01, 0.2))
        axis.tick_params(axis="both", labelsize=8)
        clean_axis(axis, grid=False)
    axes[0].set_ylabel("Survival")
    figure.supxlabel("Follow-up Years", y=0.03, fontsize=9.75)
    adjusted = "age/sex/cohort adjusted" if cohort == "fhs" else "age/race adjusted"
    labels = [f"{group} 20% ({adjusted})" for group in ("Lowest", "Highest")]
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2,
                  frameon=False, fontsize=6, handlelength=2.5, columnspacing=2.5)
    save_vector_pair(figure, OUTPUT_DIR, f"{cohort}_km_curves")


def main() -> None:
    for cohort in MORTALITY_MODELS:
        curves = pd.read_csv(OUTPUT_DIR / f"{cohort}_km_curve_points.csv")
        tests = pd.read_csv(OUTPUT_DIR / f"{cohort}_km_logrank_and_deaths.csv")
        plot_km(cohort, curves, tests)


if __name__ == "__main__":
    main()
