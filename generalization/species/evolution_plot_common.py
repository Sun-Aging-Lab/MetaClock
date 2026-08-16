"""Shared formal ENCV coefficients, spline trends, and evolution plotting."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/metaclock_evolution_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/metaclock_evolution_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from generalization.fonts import register_fonts
from generalization.modeling import atomic_write_json, project_path
from generalization.species.species_config import (
    LIFE_HISTORY,
    PAIRWISE_DISTANCE,
    SPECIES,
    encv_loo_path,
)

EVOLUTION_COLUMNS = (
    "test_species",
    "component_species",
    "distance",
    "coefficient",
    "maximum_lifespan_difference_years",
    "gestation_difference_days",
)
DEFAULT_COEFFICIENTS = Path(
    "results/figure3/generalization/species/evolution/directed_component_coefficients.tsv"
)


@dataclass(frozen=True)
class EvolutionConfig:
    """Configuration used by the current formal evolution result code."""

    alphas: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(x) for x in np.logspace(1, -5, 30))
    )
    cv_folds: int = 5
    l1_ratio: float = 0.5
    max_iter: int = 20_000
    tolerance: float = 1e-4
    n_jobs: int = -1
    bootstrap_repeats: int = 200
    random_seed: int = 42
    trend_points: int = 200


PANEL_STYLE = {
    "distance": ("#839A8E", "#557264", "#C8D4CD"),
    "maximum_lifespan_difference_years": ("#7895A1", "#466B7A", "#C7D3D8"),
    "gestation_difference_days": ("#AC8782", "#8A625E", "#D8C4C1"),
}


def nonnegative_component_encv(config: EvolutionConfig) -> Pipeline:
    """Build the formal standardized non-negative component-weight ENCV."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    alphas=np.asarray(config.alphas),
                    cv=config.cv_folds,
                    l1_ratio=config.l1_ratio,
                    max_iter=config.max_iter,
                    n_jobs=config.n_jobs,
                    positive=True,
                    selection="cyclic",
                    tol=config.tolerance,
                ),
            ),
        ]
    )


def load_distances(path: Path = PAIRWISE_DISTANCE) -> dict[frozenset[str], float]:
    """Require all 45 unique, finite, non-negative unordered species pairs."""
    path = project_path(path)
    frame = pd.read_csv(path, sep="\t")
    required = {"species_a", "species_b", "distance"}
    if len(frame) != 45 or set(frame.columns) != required:
        raise ValueError(f"{path}: expected 45 rows with {sorted(required)}")
    pairs = [frozenset((str(a), str(b))) for a, b in zip(frame.species_a, frame.species_b)]
    expected = {frozenset((a, b)) for index, a in enumerate(SPECIES) for b in SPECIES[index + 1 :]}
    values = pd.to_numeric(frame["distance"], errors="raise").to_numpy(float)
    if set(pairs) != expected or len(pairs) != len(set(pairs)):
        raise ValueError(f"{path}: unordered pair set differs")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path}: distance values are invalid")
    return dict(zip(pairs, values, strict=True))


def fit_directed_component_weights(config: EvolutionConfig) -> pd.DataFrame:
    """Fit the 90 formal directed non-negative ENCV component weights once."""
    distances = load_distances()
    rows: list[dict[str, object]] = []
    for target in SPECIES:
        path = project_path(encv_loo_path(target))
        frame = pd.read_parquet(path)
        components = [species for species in SPECIES if species != target]
        features = [f"{species}_elasticnetcv" for species in components]
        if set(frame.columns) != {"age", *features}:
            raise ValueError(f"{path}: expected age and nine ENCV component columns")
        complete = frame[["age", *features]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(complete) < config.cv_folds:
            raise ValueError(f"{path}: fewer than {config.cv_folds} complete samples")
        model = nonnegative_component_encv(config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(complete[features], complete["age"])
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(f"{path}: component ENCV did not converge")
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
    if len(result) != 90 or result.duplicated(["test_species", "component_species"]).any():
        raise RuntimeError("Expected 90 unique directed component coefficients")
    return result


def attach_life_history(frame: pd.DataFrame, path: Path = LIFE_HISTORY) -> pd.DataFrame:
    """Attach finite absolute maximum-lifespan and gestation differences."""
    path = project_path(path)
    traits = pd.read_csv(path, sep="\t").set_index("model_species")
    required = {"maximum_lifespan_years", "gestation_days"}
    if set(traits.index) != set(SPECIES) or traits.index.has_duplicates:
        raise ValueError(f"{path}: species rows differ")
    if not required.issubset(traits.columns):
        raise ValueError(f"{path}: missing life-history columns")
    result = frame.copy()
    for source, destination in (
        ("maximum_lifespan_years", "maximum_lifespan_difference_years"),
        ("gestation_days", "gestation_difference_days"),
    ):
        values = pd.to_numeric(traits[source], errors="raise")
        if not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"{path}: {source} is non-finite")
        result[destination] = (
            result["test_species"].map(values) - result["component_species"].map(values)
        ).abs()
    return result.loc[:, list(EVOLUTION_COLUMNS)]


def validate_coefficient_table(frame: pd.DataFrame) -> None:
    """Validate the reusable 90-row directed coefficient table."""
    if set(frame.columns) != set(EVOLUTION_COLUMNS) or len(frame) != 90:
        raise ValueError("Evolution coefficient table has the wrong shape or columns")
    if frame.duplicated(["test_species", "component_species"]).any():
        raise ValueError("Evolution coefficient table has duplicate directed pairs")
    numeric = frame[list(EVOLUTION_COLUMNS[2:])].to_numpy(float)
    if not np.isfinite(numeric).all() or (frame["coefficient"] < 0).any():
        raise ValueError("Evolution coefficient table contains invalid values")


def prepare_coefficient_table(
    output: Path = DEFAULT_COEFFICIENTS,
    config: EvolutionConfig | None = None,
) -> Path:
    """Compute once or safely reuse the formal directed component table."""
    config = config or EvolutionConfig()
    output = project_path(output)
    provenance_path = output.with_suffix(".provenance.json")
    config_json = json.loads(json.dumps(asdict(config)))
    if output.is_file() and provenance_path.is_file():
        frame = pd.read_csv(output, sep="\t")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        validate_coefficient_table(frame)
        if provenance.get("config") != config_json:
            raise ValueError(f"{output}: existing configuration differs")
        return output
    frame = attach_life_history(fit_directed_component_weights(config))
    validate_coefficient_table(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    frame.to_csv(temporary, sep="\t", index=False, float_format="%.10g")
    temporary.replace(output)
    atomic_write_json(
        {
            "config": config_json,
            "component_model": "StandardScaler + positive ElasticNetCV",
            "spline_model": "SplineTransformer + ElasticNetCV (no second scaler)",
            "rows": len(frame),
        },
        provenance_path,
    )
    return output


def spline_encv(config: EvolutionConfig) -> Pipeline:
    """Build the formal cubic spline plus ENCV, intentionally without a scaler."""
    return Pipeline(
        [
            ("spline", SplineTransformer(n_knots=3, degree=3, include_bias=False)),
            (
                "model",
                ElasticNetCV(
                    alphas=np.asarray(config.alphas),
                    cv=config.cv_folds,
                    l1_ratio=config.l1_ratio,
                    max_iter=config.max_iter,
                    n_jobs=config.n_jobs,
                    selection="cyclic",
                    tol=config.tolerance,
                ),
            ),
        ]
    )


def fit_spline_interval(x: np.ndarray, y: np.ndarray, config: EvolutionConfig) -> pd.DataFrame:
    """Fit the formal spline and its 200-resample percentile interval."""
    if x.size != y.size or x.size < config.cv_folds:
        raise ValueError("Spline inputs have invalid size")
    grid = np.linspace(float(x.min()), float(x.max()), config.trend_points)
    rng = np.random.default_rng(config.random_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = spline_encv(config).fit(x[:, None], y).predict(grid[:, None])
        bootstrap = np.empty((config.bootstrap_repeats, config.trend_points))
        for index in range(config.bootstrap_repeats):
            sample = rng.integers(0, len(x), len(x))
            bootstrap[index] = (
                spline_encv(config).fit(x[sample, None], y[sample]).predict(grid[:, None])
            )
    lower, upper = np.percentile(bootstrap, (2.5, 97.5), axis=0)
    return pd.DataFrame({"x": grid, "fitted": fitted, "lower": lower, "upper": upper})


def configure_evolution_style() -> None:
    """Apply the current formal evolution typography and outlines."""
    register_fonts()
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.family": ["Liberation Sans"],
            "font.sans-serif": ["Liberation Sans"],
            "font.size": 9.0,
            "axes.labelsize": 9.75,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.2,
            "xtick.major.width": 0.2,
            "xtick.minor.width": 0.2,
            "ytick.major.width": 0.2,
            "ytick.minor.width": 0.2,
            "axes.edgecolor": "#CCCCCC",
        }
    )


def plot_evolution_relationship(
    metric: str,
    xlabel: str,
    output_stem: Path,
    *,
    coefficient_path: Path = DEFAULT_COEFFICIENTS,
    config: EvolutionConfig | None = None,
) -> tuple[Path, Path]:
    """Read shared coefficients and render one formal evolution relationship."""
    config = config or EvolutionConfig()
    coefficient_path = project_path(coefficient_path)
    if not coefficient_path.is_file():
        raise FileNotFoundError(f"Run prepare_evolution_coefficients first: {coefficient_path}")
    pairs = pd.read_csv(coefficient_path, sep="\t")
    validate_coefficient_table(pairs)
    x, y = pairs[metric].to_numpy(float), pairs["coefficient"].to_numpy(float)
    trend = fit_spline_interval(x, y, config)
    point_color, line_color, shade_color = PANEL_STYLE[metric]
    configure_evolution_style()
    figure, axis = plt.subplots(figsize=(90 / 25.4, 60 / 25.4))
    axis.scatter(
        x, y, color=point_color, alpha=0.68, edgecolors="none", linewidths=0, s=8.0, zorder=2
    )
    axis.plot(
        trend["x"],
        trend["fitted"],
        color=line_color,
        linewidth=1.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )
    axis.fill_between(
        trend["x"],
        trend["lower"],
        trend["upper"],
        facecolor=shade_color,
        alpha=0.72,
        edgecolor="none",
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
    axis.set_ylim(bottom=-0.02 * np.ptp(y))
    axis.grid(False)
    figure.subplots_adjust(left=0.18, right=0.96, bottom=0.20, top=0.95)
    stem = project_path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = (stem.with_suffix(".pdf"), stem.with_suffix(".svg"))
    for output in outputs:
        figure.savefig(output, dpi=300)
    plt.close(figure)
    trend.to_csv(stem.with_name(f"{stem.name}_trend.tsv"), sep="\t", index=False)
    return outputs
