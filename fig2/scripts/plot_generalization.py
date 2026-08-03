#!/usr/bin/env python3
"""Plot the Figure 2 array, tissue, and species generalization panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "fig2"
RESULT_FILES = (
    "01_array_generalization.tsv",
    "02_tissue_generalization.tsv",
    "03_species_generalization.tsv",
)
MODEL_STYLE = {
    "SVM": ("#7A7A7A", "o", "-"),
    "Meta SVM": ("#9C89B8", "s", "-"),
    "EN": ("#7A7A7A", "o", "-"),
    "Meta EN": ("#9C89B8", "s", "-"),
    "Mean": ("#B3B3B3", "D", "--"),
}


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load the three panel-ordered compact result tables."""
    frames = [pd.read_csv(results_dir / name, sep="\t") for name in RESULT_FILES]
    result = pd.concat(frames, ignore_index=True)
    required = {
        "panel_order",
        "group",
        "dataset",
        "label",
        "model",
        "training_samples",
        "mean_mae",
        "q025_mae",
        "q975_mae",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"Result tables are missing columns: {sorted(missing)}")
    return result.sort_values(["panel_order", "training_samples"])


def _plot_curve(axis, frame: pd.DataFrame, model: str) -> None:
    """Draw one mean curve and empirical 95% repeat interval."""
    color, marker, linestyle = MODEL_STYLE[model]
    x = frame["training_samples"].to_numpy(dtype=float)
    axis.plot(
        x,
        frame["mean_mae"],
        label=model,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.0,
        markersize=2.8,
    )
    axis.fill_between(
        x,
        frame["q025_mae"],
        frame["q975_mae"],
        color=color,
        alpha=0.12,
    )


def plot_panel(frame: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Plot one dataset as paired SVM and EN subpanels at 90 x 60 mm."""
    panel_order = int(frame["panel_order"].iloc[0])
    dataset = str(frame["dataset"].iloc[0])
    label = str(frame["label"].iloc[0])
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(90 / 25.4, 60 / 25.4))
    for axis, models, title in (
        (axes[0], ("Mean", "SVM", "Meta SVM"), "SVM"),
        (axes[1], ("Mean", "EN", "Meta EN"), "EN"),
    ):
        for model in models:
            part = frame[frame["model"] == model].sort_values("training_samples")
            if not part.empty:
                _plot_curve(axis, part, model)
        axis.set_xscale("log")
        axis.set_xlabel("Training samples")
        axis.set_title(title, fontweight="bold")
        axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=5.5)
    axes[0].set_ylabel("MAE")
    figure.suptitle(label, x=0.575, y=0.95, fontsize=7.0)
    figure.subplots_adjust(left=0.18, right=0.94, bottom=0.18, top=0.80, wspace=0.40)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{panel_order:02d}_{dataset}_generalization"
    pdf, svg = output_dir / f"{stem}.pdf", output_dir / f"{stem}.svg"
    figure.savefig(pdf, dpi=300)
    figure.savefig(svg, dpi=300)
    plt.close(figure)
    return pdf, svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=FIGURE_DIR / "results")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "panels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = load_results(args.results_dir)
    for _, frame in results.groupby("panel_order", sort=True):
        for path in plot_panel(frame, args.output_dir):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
