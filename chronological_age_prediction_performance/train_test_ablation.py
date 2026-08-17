#!/usr/bin/env python3
"""Run every cumulative component-clock ablation on training and test datasets.

The two input files use the same wide schema as ``train_test_architectures``:
``dataset``, ``sample_id``, ``age``, and all 16 Figure-2 component-clock
columns.  Replace only the two path defaults below.  This script refits the
MetaClock on each reduced component set, evaluates each test dataset, and
writes all three ablation scatter plots to a fixed results directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .train_test_architectures import (
    COMPONENT_CLOCKS,
    COMPONENT_COLORS,
    COMPONENT_LABELS,
    DEFAULT_TEST_DATASET,
    DEFAULT_TRAINING_DATASET,
    DISPLAYED_COMPONENTS,
    META_COLOR,
    axes_for,
    finish,
    read_dataset,
)
from .train_test_architectures import (
    configure_style as configure_architecture_style,
)

# ---------------------------------------------------------------------------
# User-editable paths and Figure-2 ablation definition
# ---------------------------------------------------------------------------
DEFAULT_ABLATION_TRAINING_DATASET = DEFAULT_TRAINING_DATASET
DEFAULT_ABLATION_TEST_DATASET = DEFAULT_TEST_DATASET
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results/chronological_age_prediction/ablation"
REMOVAL_ORDER = ("zhangblup", "han", "skinandblood")


# ---------------------------------------------------------------------------
# Reduced MetaClock fitting and held-out metrics
# ---------------------------------------------------------------------------
def fit_metaclock(training: pd.DataFrame, clocks: tuple[str, ...]) -> Pipeline:
    """Fit the formal StandardScaler + linear-SVR MetaClock on one clock set."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="linear")),
        ]
    )
    model.fit(training.loc[:, clocks], training["age"])
    return model


def heldout_metrics(
    test: pd.DataFrame, clocks: tuple[str, ...], model: Pipeline, step: int
) -> pd.DataFrame:
    """Score retained individual clocks and the independently retrained MetaClock."""
    removed = ";".join(REMOVAL_ORDER[:step])
    rows: list[dict[str, object]] = []
    for dataset, panel in test.groupby("dataset", sort=True):
        age = panel["age"].to_numpy(dtype=float)
        prediction = np.asarray(model.predict(panel.loc[:, clocks]), dtype=float)
        values = {
            "MetaClock": prediction,
            **{clock: panel[clock].to_numpy(dtype=float) for clock in clocks},
        }
        for predictor, estimate in values.items():
            if len(age) < 3 or np.std(age) == 0 or np.std(estimate) == 0:
                raise ValueError(
                    f"{dataset}/{predictor} needs at least three non-constant test observations"
                )
            rows.append(
                {
                    "dataset": dataset,
                    "predictor": predictor,
                    "n": len(age),
                    "mae": float(mean_absolute_error(age, estimate)),
                    "pearson_r": float(pearsonr(age, estimate).statistic),
                    "ablation_step": step,
                    "removed_clocks": removed,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Formal cumulative-ablation scatter plots
# ---------------------------------------------------------------------------
def configure_style() -> None:
    configure_architecture_style()


def plot_step(table: pd.DataFrame, step: int) -> None:
    """Plot each retained component and its retrained MetaClock for one step."""
    datasets = sorted(table["dataset"].unique())
    figure, axes = axes_for(datasets)
    retained = tuple(clock for clock in DISPLAYED_COMPONENTS if clock not in REMOVAL_ORDER[:step])
    for index, (dataset, axis) in enumerate(zip(datasets, axes, strict=True)):
        panel = table.loc[table["dataset"].eq(dataset)].set_index("predictor")
        for clock in retained:
            axis.scatter(
                panel.at[clock, "mae"],
                panel.at[clock, "pearson_r"],
                s=17,
                color=COMPONENT_COLORS[clock],
                edgecolor="#222",
                linewidth=0.2,
                alpha=0.9,
            )
        axis.scatter(
            panel.at["MetaClock", "mae"],
            panel.at["MetaClock", "pearson_r"],
            marker="*",
            s=85.5,
            color=META_COLOR,
            edgecolor="#111",
            linewidth=0.2,
            zorder=3,
        )
        axis.set_title(dataset)
    removed_label = COMPONENT_LABELS[REMOVAL_ORDER[step - 1]]
    figure.suptitle(f"Cumulative ablation {step}/3: {removed_label} removed", y=0.99)
    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=META_COLOR,
            markeredgecolor="#111",
            markersize=9,
            label="Retrained MetaClock",
        )
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COMPONENT_COLORS[clock],
            markeredgecolor="#222",
            markersize=3.5,
            label=COMPONENT_LABELS[clock],
        )
        for clock in retained
    )
    figure.legend(
        handles=handles,
        title="Clock",
        loc="center left",
        bbox_to_anchor=(0.83, 0.5),
        frameon=False,
        fontsize=7.5,
    )
    finish(figure, axes, datasets)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"ablation_step_{step:02d}_without_{REMOVAL_ORDER[step - 1]}"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    plt.close(figure)


# ---------------------------------------------------------------------------
# Single two-path CLI: all three fits, tests, tables, and plots
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dataset", type=Path, default=DEFAULT_ABLATION_TRAINING_DATASET)
    parser.add_argument("--test-dataset", type=Path, default=DEFAULT_ABLATION_TEST_DATASET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    training, test = read_dataset(args.training_dataset), read_dataset(args.test_dataset)
    for step in range(1, len(REMOVAL_ORDER) + 1):
        clocks = tuple(clock for clock in COMPONENT_CLOCKS if clock not in REMOVAL_ORDER[:step])
        metrics = heldout_metrics(test, clocks, fit_metaclock(training, clocks), step)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(OUTPUT_DIR / f"step_{step:02d}_source_data.csv", index=False)
        plot_step(metrics, step)


if __name__ == "__main__":
    main()
