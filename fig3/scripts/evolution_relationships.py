#!/usr/bin/env python3
"""Compute and plot Figure 3 evolutionary and life-history relationships."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path

_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "metaclock_figure3_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "fig3"
EXPECTED_PAIRS = 90
SPECIES = (
    "BosTaurus",
    "DelphinapterusLeucas",
    "EquusCaballus",
    "HeterocephalusGlaber",
    "HomoSapiens",
    "MacacaMulatta",
    "MusMusculus",
    "OvisAries",
    "PapioHamadryas",
    "RattusNorvegicus",
)
PANELS = (
    (
        1,
        "01_evolutionary_distance.tsv",
        "Molecular evolutionary distance",
        "01_evolutionary_distance",
        (100.0, 75.0),
    ),
    (
        2,
        "02_maximum_lifespan.tsv",
        "Absolute maximum lifespan difference (years)",
        "02_maximum_lifespan",
        (90.0, 60.0),
    ),
    (
        3,
        "03_gestation_time.tsv",
        "Absolute gestation time difference (days)",
        "03_gestation_time",
        (90.0, 60.0),
    ),
)


def load_pairwise_distances(path: Path) -> dict[frozenset[str], float]:
    """Load and validate all 45 undirected distances for ten species."""
    frame = pd.read_csv(path, sep="\t")
    if len(frame) != 45 or set(frame) != {"species_a", "species_b", "distance"}:
        raise ValueError("Distance table must contain 45 species_a/species_b/distance rows")
    distances = {
        frozenset((row.species_a, row.species_b)): float(row.distance)
        for row in frame.itertuples(index=False)
    }
    if len(distances) != 45:
        raise ValueError("Distance table contains duplicate unordered pairs")
    return distances


def fit_directed_coefficients(
    loo_dir: Path,
    distances: dict[frozenset[str], float],
) -> pd.DataFrame:
    """Fit one positive-EN coefficient for every directed species pair."""
    rows: list[dict[str, str | float]] = []
    for target in SPECIES:
        path = loo_dir / f"loo_per_species_predictions_{target}.parquet"
        frame = pd.read_parquet(path)
        components = [species for species in SPECIES if species != target]
        features = [f"{species}_elasticnetcv" for species in components]
        missing = sorted({"age", *features} - set(frame))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        model_frame = frame[["age", *features]].apply(pd.to_numeric, errors="coerce").dropna()
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(alpha=0.001, l1_ratio=0.5, positive=True),
                ),
            ]
        )
        model.fit(model_frame[features], model_frame["age"])
        for component, coefficient in zip(
            components, model.named_steps["model"].coef_, strict=True
        ):
            rows.append(
                {
                    "test_species": target,
                    "component_species": component,
                    "distance": distances[frozenset((target, component))],
                    "coefficient": float(coefficient),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_PAIRS or result[
        ["test_species", "component_species"]
    ].duplicated().any():
        raise ValueError("Expected 90 unique directed species pairs")
    return result


def add_life_history(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Add absolute maximum-lifespan and gestation differences."""
    traits = pd.read_csv(path, sep="\t").set_index("model_species")
    if set(traits.index) != set(SPECIES) or traits.index.has_duplicates:
        raise ValueError("Life-history rows must match the ten model species")
    result = frame.copy()
    for trait, output in (
        ("maximum_lifespan_years", "maximum_lifespan_difference_years"),
        ("gestation_days", "gestation_difference_days"),
    ):
        values = pd.to_numeric(traits[trait], errors="raise")
        test = result["test_species"].map(values)
        component = result["component_species"].map(values)
        result[output] = (test - component).abs()
    return result


def write_panel_tables(frame: pd.DataFrame, results_dir: Path) -> list[Path]:
    """Write one 90-row source-data table per panel in figure order."""
    results_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        (1, "distance", "01_evolutionary_distance.tsv"),
        (2, "maximum_lifespan_difference_years", "02_maximum_lifespan.tsv"),
        (3, "gestation_difference_days", "03_gestation_time.tsv"),
    )
    outputs: list[Path] = []
    for panel_order, metric, filename in specifications:
        output = frame[["test_species", "component_species", metric, "coefficient"]].rename(
            columns={metric: "x_value"}
        )
        output.insert(0, "x_metric", metric)
        output.insert(0, "panel_order", panel_order)
        path = results_dir / filename
        output.to_csv(path, sep="\t", index=False, float_format="%.10g")
        outputs.append(path)
    return outputs


def fit_spline_interval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstraps: int = 200,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the shared cubic spline and pointwise row-bootstrap interval."""
    grid = np.linspace(x.min(), x.max(), 200)

    def make_spline() -> Pipeline:
        return Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(n_knots=3, degree=3, include_bias=False),
                ),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.5)),
            ]
        )

    model = make_spline()
    model.fit(x[:, None], y)
    fitted = model.predict(grid[:, None])
    rng = np.random.default_rng(random_seed)
    boot = np.empty((n_bootstraps, len(grid)))
    for index in range(n_bootstraps):
        sample = rng.integers(0, len(x), len(x))
        model = make_spline()
        model.fit(x[sample, None], y[sample])
        boot[index] = model.predict(grid[:, None])
    lower, upper = np.percentile(boot, (2.5, 97.5), axis=0)
    return grid, fitted, lower, upper


def plot_panel(
    source: Path,
    xlabel: str,
    stem: str,
    canvas_mm: tuple[float, float],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Draw one relationship panel from its ordered 90-row result table."""
    frame = pd.read_csv(source, sep="\t")
    if len(frame) != EXPECTED_PAIRS:
        raise ValueError(f"{source} must contain 90 directed pairs")
    x = frame["x_value"].to_numpy(dtype=float)
    y = frame["coefficient"].to_numpy(dtype=float)
    grid, fitted, lower, upper = fit_spline_interval(x, y)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(
        figsize=(canvas_mm[0] / 25.4, canvas_mm[1] / 25.4)
    )
    axis.scatter(x, y, color="black", alpha=0.45, edgecolors="none", s=38, zorder=2)
    axis.plot(grid, fitted, color="black", linewidth=2.8)
    axis.fill_between(grid, lower, upper, color="black", alpha=0.30)
    axis.text(
        0.02,
        0.98,
        f"Pearson r = {np.corrcoef(x, y)[0, 1]:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none"},
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Model Coefficient")
    axis.set_ylim(bottom=-0.02 * np.ptp(y))
    axis.grid(False)
    figure.subplots_adjust(left=0.18, right=0.96, bottom=0.20, top=0.95)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf, svg = output_dir / f"{stem}.pdf", output_dir / f"{stem}.svg"
    figure.savefig(pdf, dpi=300)
    figure.savefig(svg, dpi=300)
    plt.close(figure)
    return pdf, svg


def compute(data_dir: Path, loo_dir: Path, results_dir: Path) -> list[Path]:
    """Recompute the 90 coefficients and write the three compact result tables."""
    distances = load_pairwise_distances(data_dir / "pairwise_evolutionary_distances.tsv")
    coefficients = fit_directed_coefficients(loo_dir, distances)
    complete = add_life_history(coefficients, data_dir / "species_life_history.tsv")
    return write_panel_tables(complete, results_dir)


def plot(results_dir: Path, output_dir: Path) -> list[Path]:
    """Plot all three panels from committed compact result tables."""
    outputs: list[Path] = []
    for _, filename, xlabel, stem, canvas in PANELS:
        outputs.extend(plot_panel(results_dir / filename, xlabel, stem, canvas, output_dir))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compute", "plot", "all"))
    parser.add_argument("--data-dir", type=Path, default=FIGURE_DIR / "datasets")
    parser.add_argument("--loo-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=FIGURE_DIR / "results")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR / "panels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loo_dir = args.loo_dir or args.data_dir / "loo_predictions"
    outputs: list[Path] = []
    if args.command in {"compute", "all"}:
        outputs.extend(compute(args.data_dir, loo_dir, args.results_dir))
    if args.command in {"plot", "all"}:
        outputs.extend(plot(args.results_dir, args.output_dir))
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
