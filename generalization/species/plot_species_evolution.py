#!/usr/bin/env python3
"""Render the three cross-species evolutionary relationship panels."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from generalization.fonts import register_fonts
from .species_config import SPECIES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "generalization" / "species"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "generalization" / "species" / "evolution"
INPUT_ROOT = Path(__file__).resolve().parent

PANEL_WIDTH_MM = 90.0
PANEL_HEIGHT_MM = 60.0
EVOLUTION_RESAMPLES = 200
EVOLUTION_SEED = 42
EVOLUTION_STYLE = {
    "distance": ("#839A8E", "#557264", "#C8D4CD"),
    "maximum_lifespan_difference_years": ("#7895A1", "#466B7A", "#C7D3D8"),
    "gestation_difference_days": ("#AC8782", "#8A625E", "#D8C4C1"),
}
PANELS = (
    (
        INPUT_ROOT / "pairwise_evolutionary_distances.csv",
        "distance",
        "Molecular evolutionary distance",
        "component_en_coef_vs_molecular_evolutionary_distance",
    ),
    (
        INPUT_ROOT / "pairwise_maximum_lifespan_differences.csv",
        "maximum_lifespan_difference_years",
        "|Maximum lifespan difference| (years)",
        "component_en_coef_vs_max_lifespan_difference",
    ),
    (
        INPUT_ROOT / "pairwise_gestation_time_differences.csv",
        "gestation_difference_days",
        "|Gestation time difference| (days)",
        "component_en_coef_vs_gestation_difference",
    ),
)


def configure_style() -> None:
    """Apply the manuscript style used by the original evolution panels."""
    register_fonts()
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 8.0,
            "axes.labelsize": 9.75,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.2,
            "xtick.major.width": 0.2,
            "ytick.major.width": 0.2,
            "axes.edgecolor": "#CCCCCC",
            "text.color": "#262626",
            "axes.labelcolor": "#262626",
            "xtick.color": "#262626",
            "ytick.color": "#262626",
        }
    )


def collect_evolution_coefficients() -> pd.DataFrame | None:
    """Collect ten target-species coefficient tables, or report pending work."""
    paths = [RESULT_ROOT / species / "evolution_coefficients.tsv" for species in SPECIES]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        print(
            "[evolution pending] No *_elasticnetcv inputs were supplied; "
            "run species training before drawing evolution panels."
        )
        return None
    frames = [pd.read_csv(path, sep="\t") for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    required = {"test_species", "component_species", "coefficient"}
    if not required.issubset(frame.columns):
        raise ValueError("Evolution coefficient tables have an invalid schema")
    target_counts = frame.groupby("test_species")["component_species"].nunique()
    if len(target_counts) < 10 or not target_counts.eq(9).all() or len(frame) < 90:
        print(
            f"[evolution pending] Found {len(target_counts)}/10 species and {len(frame)}/90 "
            "directed coefficients; run all species before drawing evolution panels."
        )
        return None
    if (
        len(target_counts) != 10
        or len(frame) != 90
        or frame.duplicated(["test_species", "component_species"]).any()
    ):
        raise ValueError("Evolution requires exactly 90 unique directed pairs from ten species")
    if set(target_counts.index) != set(SPECIES):
        raise ValueError("Evolution target species differ from the formal ten-species set")
    for target, group in frame.groupby("test_species"):
        expected_components = set(SPECIES) - {str(target)}
        if set(group["component_species"]) != expected_components:
            raise ValueError(f"{target}: evolution component-species set is not leave-one-out")
    coefficients = pd.to_numeric(frame["coefficient"], errors="raise").to_numpy(float)
    if not np.isfinite(coefficients).all() or (coefficients < 0).any():
        raise ValueError("Evolution coefficients must be finite and non-negative")
    return frame.loc[:, ["test_species", "component_species", "coefficient"]].sort_values(
        ["test_species", "component_species"]
    ).reset_index(drop=True)


def expected_pair_set() -> set[frozenset[str]]:
    """Return the complete set of 45 unordered pairs among the formal species."""
    return {
        frozenset((left, right))
        for index, left in enumerate(SPECIES)
        for right in SPECIES[index + 1 :]
    }


def load_pairwise_metric(path: Path, value_column: str) -> dict[frozenset[str], float]:
    """Read and validate one complete 45-pair metric table."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing evolution metric table: {path}")
    frame = pd.read_csv(path)
    required = {"species_a", "species_b", value_column}
    if len(frame) != 45 or not required.issubset(frame.columns):
        raise ValueError(f"{path}: expected 45 rows with {sorted(required)}")
    values: dict[frozenset[str], float] = {}
    for species_a, species_b, raw_value in frame.loc[
        :, ["species_a", "species_b", value_column]
    ].itertuples(index=False, name=None):
        pair = frozenset((str(species_a), str(species_b)))
        if len(pair) != 2:
            raise ValueError(f"{path}: self-pairs are not allowed")
        if pair in values:
            raise ValueError(f"{path}: duplicate unordered pair {sorted(pair)}")
        values[pair] = float(raw_value)
    expected = expected_pair_set()
    if set(values) != expected:
        missing = sorted("/".join(sorted(pair)) for pair in expected - set(values))
        extra = sorted("/".join(sorted(pair)) for pair in set(values) - expected)
        raise ValueError(f"{path}: pair universe differs; missing={missing}, extra={extra}")
    numeric = np.fromiter(values.values(), dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError(f"{path}: metric values must be finite and non-negative")
    return values


def attach_evolution_metrics(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Attach all three pairwise metrics from their dedicated CSV tables."""
    result = coefficients.copy()
    for path, metric, _xlabel, _filename in PANELS:
        values = load_pairwise_metric(path, metric)
        result[metric] = [
            values[frozenset((test, component))]
            for test, component in zip(
                result["test_species"], result["component_species"], strict=True
            )
        ]
    return result


def spline_model() -> Pipeline:
    """Build the formal cubic-spline/fixed-alpha trend model."""
    return Pipeline(
        [
            ("spline", SplineTransformer(n_knots=3, degree=3, include_bias=False)),
            (
                "model",
                ElasticNet(
                    alpha=0.001,
                    l1_ratio=0.5,
                    max_iter=1_000,
                    selection="cyclic",
                    tol=1e-4,
                ),
            ),
        ]
    )


def spline_interval(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Fit the formal trend and a 200-row-resample percentile interval."""
    grid = np.linspace(float(x.min()), float(x.max()), 200)
    rng = np.random.default_rng(EVOLUTION_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = spline_model().fit(x[:, None], y).predict(grid[:, None])
        resample = np.empty((EVOLUTION_RESAMPLES, len(grid)))
        for index in range(EVOLUTION_RESAMPLES):
            sample = rng.integers(0, len(x), len(x))
            resample[index] = spline_model().fit(x[sample, None], y[sample]).predict(
                grid[:, None]
            )
    lower, upper = np.percentile(resample, (2.5, 97.5), axis=0)
    return pd.DataFrame({"x": grid, "fitted": fitted, "lower": lower, "upper": upper})


def render_evolution_panel(
    source: pd.DataFrame, metric: str, xlabel: str, filename: str
) -> list[Path]:
    """Render one coefficient/relationship panel."""
    x = source[metric].to_numpy(float)
    y = source["coefficient"].to_numpy(float)
    trend = spline_interval(x, y)
    point_color, line_color, shade_color = EVOLUTION_STYLE[metric]
    configure_style()
    figure, axis = plt.subplots(figsize=(PANEL_WIDTH_MM / 25.4, PANEL_HEIGHT_MM / 25.4))
    axis.scatter(x, y, color=point_color, alpha=0.68, edgecolors="none", s=8.0, zorder=2)
    axis.plot(trend["x"], trend["fitted"], color=line_color, linewidth=1.0, zorder=3)
    axis.fill_between(
        trend["x"],
        trend["lower"],
        trend["upper"],
        facecolor=shade_color,
        alpha=0.72,
        linewidth=0,
        zorder=1,
    )
    axis.text(
        0.02,
        0.98,
        f"Pearson r = {np.corrcoef(x, y)[0, 1]:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        },
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Model Coefficient")
    axis.set_ylim(bottom=-0.02 * float(np.ptp(y)))
    axis.grid(False)
    figure.subplots_adjust(left=0.18, right=0.96, bottom=0.20, top=0.95)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_ROOT / filename
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".svg")]
    for output in outputs:
        figure.savefig(output, dpi=300)
    plt.close(figure)
    trend.to_csv(FIGURE_ROOT / f"{filename}_trend.tsv", sep="\t", index=False)
    return outputs


def main() -> int:
    coefficients = collect_evolution_coefficients()
    if coefficients is None:
        return 0
    source = attach_evolution_metrics(coefficients)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    source.to_csv(FIGURE_ROOT / "directed_component_coefficients.tsv", sep="\t", index=False)
    outputs = [
        output
        for _path, metric, xlabel, filename in PANELS
        for output in render_evolution_panel(source, metric, xlabel, filename)
    ]
    print(*outputs, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
