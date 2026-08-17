#!/usr/bin/env python3
"""Aggregate species shards and render every generalization plot.

No command-line arguments are required.  The script discovers every completed
dataset written by ``train_species_generalization.py`` and creates four layouts:
from 10 samples and from the shared mean-baseline cutoff, each with and without
the Mean curve.  Every layout is exported as PDF and SVG.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from generalization.fonts import register_fonts

matplotlib.use("Agg", force=True)

############################ FIXED CONFIG ####################################

MODEL_NAMES = ("EN", "Meta EN", "SVM", "Meta SVM", "Mean")
N_TRAINING_SIZES = 30
N_RESAMPLES = 50
MINIMUM_PLOT_SIZE = 10
EXPECTED_CONFIG_ID = (
    "generalization-v2-split105-30x50-without-replacement-scaled-encv5-"
    "scaled-svr-strict-mlp-elu-loo"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "generalization" / "species"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "generalization" / "species"

PANEL_WIDTH_MM = 90.0
PANEL_HEIGHT_MM = 60.0
DIRECT_COLOR = "#B77729"
META_COLOR = "#4286A1"
MEAN_COLOR = "#7A7A7A"
EVOLUTION_RESAMPLES = 200
EVOLUTION_SEED = 42
EVOLUTION_STYLE = {
    "distance": ("#839A8E", "#557264", "#C8D4CD"),
    "maximum_lifespan_difference_years": ("#7895A1", "#466B7A", "#C7D3D8"),
    "gestation_difference_days": ("#AC8782", "#8A625E", "#D8C4C1"),
}
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


######################## SHARD AGGREGATION ###################################


def validate_shard(frame: pd.DataFrame, size_index: int) -> None:
    """Require a complete model x resample Cartesian product."""
    required = {
        "model",
        "training_size",
        "size_index",
        "repeat_id",
        "mae",
        "prediction_sd",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Shard {size_index} is missing {sorted(required - set(frame.columns))}")
    expected = {(model, repeat) for model in MODEL_NAMES for repeat in range(N_RESAMPLES)}
    observed = set(zip(frame["model"], frame["repeat_id"].astype(int), strict=True))
    if len(frame) != len(expected) or observed != expected:
        raise ValueError(f"Shard {size_index} has an incomplete model x resample grid")
    if set(frame["size_index"].astype(int)) != {size_index}:
        raise ValueError(f"Shard {size_index} records a different size index")
    if frame["training_size"].nunique() != 1:
        raise ValueError(f"Shard {size_index} contains multiple training sizes")
    if not np.isfinite(frame[["mae", "prediction_sd"]].to_numpy(float)).all():
        raise ValueError(f"Shard {size_index} contains non-finite metrics")


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load all 30 point shards and their common provenance."""
    frames: list[pd.DataFrame] = []
    provenances: list[dict[str, object]] = []
    for size_index in range(N_TRAINING_SIZES):
        shard = dataset_dir / "shards" / f"size_{size_index:02d}.parquet"
        provenance_path = shard.with_suffix(".provenance.json")
        if not shard.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(f"Missing point shard or provenance: {shard}")
        frame = pd.read_parquet(shard)
        validate_shard(frame, size_index)
        frames.append(frame)
        provenances.append(json.loads(provenance_path.read_text(encoding="utf-8")))
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(["model", "size_index", "repeat_id"]).any():
        raise ValueError(f"{dataset_dir.name}: duplicate model/size/resample rows")
    sizes = merged.groupby("size_index")["training_size"].first().sort_index().astype(int).tolist()
    first = provenances[0]
    if first.get("config_id") != EXPECTED_CONFIG_ID:
        raise ValueError(f"{dataset_dir.name}: training configuration is not current")
    for provenance in provenances:
        if (
            provenance.get("dataset") != first.get("dataset")
            or provenance.get("input_path") != first.get("input_path")
            or provenance.get("config_id") != first.get("config_id")
            or provenance.get("training_sizes") != sizes
        ):
            raise ValueError(f"{dataset_dir.name}: shard provenance is inconsistent")
    return merged, first


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the mean and empirical 95% interval across 50 resamples."""
    summary = (
        frame.groupby(["model", "training_size"], observed=True)["mae"]
        .agg(
            mean="mean",
            q025=lambda values: values.quantile(0.025),
            q975=lambda values: values.quantile(0.975),
            resamples="size",
        )
        .reset_index()
    )
    if not summary["resamples"].eq(N_RESAMPLES).all():
        raise ValueError("A summary cell does not contain exactly 50 resamples")
    return summary.drop(columns="resamples")


def family_cutoff(summary: pd.DataFrame, direct: str, meta: str) -> int:
    """First size >=10 where both members of one family beat Mean MAE."""
    wide = summary.pivot(index="training_size", columns="model", values="mean")
    eligible = wide.index[
        (wide.index >= MINIMUM_PLOT_SIZE)
        & wide[direct].lt(wide["Mean"])
        & wide[meta].lt(wide["Mean"])
    ]
    if eligible.empty:
        raise ValueError(f"{direct} and {meta} never both beat Mean at size >=10")
    return int(eligible.min())


############################# PLOTTING #######################################


def configure_style() -> None:
    """Apply the current 90 x 60 mm manuscript style."""
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


def select(summary: pd.DataFrame, model: str, cutoff: int) -> pd.DataFrame:
    """Select one model curve at or beyond a declared cutoff."""
    result = summary.loc[
        summary["model"].eq(model) & summary["training_size"].ge(cutoff)
    ].sort_values("training_size")
    if result.empty:
        raise ValueError(f"No {model} points remain at cutoff {cutoff}")
    return result


def draw_curve(
    axis: Axes,
    values: pd.DataFrame,
    label: str,
    color: str,
    marker: str,
    *,
    linestyle: str = "-",
    alpha: float = 0.28,
) -> None:
    """Draw one resample-mean curve and its empirical interval."""
    x = values["training_size"].to_numpy(float)
    axis.plot(
        x,
        values["mean"],
        label=label,
        color=color,
        marker=marker,
        markersize=2.8,
        markeredgewidth=0.2,
        linewidth=0.9,
        linestyle=linestyle,
        zorder=3,
    )
    axis.fill_between(
        x,
        values["q025"].to_numpy(float),
        values["q975"].to_numpy(float),
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=2,
    )


def finish_axis(axis: Axes, displayed: list[pd.DataFrame], show_ylabel: bool) -> None:
    """Finish one family subplot using only its displayed range."""
    lower = min(float(values["q025"].min()) for values in displayed)
    upper = max(float(values["q975"].max()) for values in displayed)
    axis.set_ylim(max(0.0, lower * 0.8), upper * 1.2 if upper > 0 else 1.0)
    sizes = displayed[-1]["training_size"].astype(int).tolist()
    ticks = [sizes[index] for index in sorted({0, len(sizes) // 2, len(sizes) - 1})]
    axis.set_xscale("log")
    axis.set_xticks(ticks, labels=[str(value) for value in ticks])
    axis.minorticks_off()
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel("# Training Samples")
    axis.set_ylabel("MAE" if show_ylabel else "")
    axis.grid(True, linestyle="--", alpha=0.35, linewidth=0.4)
    axis.legend(loc="best", frameon=False, handletextpad=0.3)


def render_panel(
    summary: pd.DataFrame,
    title: str,
    output_stem: Path,
    cutoffs: dict[str, int],
    *,
    include_mean: bool,
) -> tuple[Path, Path]:
    """Render paired SVM/ENCV subplots with a common displayed cutoff."""
    configure_style()
    figure, (svm_axis, en_axis) = plt.subplots(
        1, 2, figsize=(PANEL_WIDTH_MM / 25.4, PANEL_HEIGHT_MM / 25.4)
    )
    for axis, family, direct, meta in (
        (svm_axis, "SVM", "SVM", "Meta SVM"),
        (en_axis, "EN", "EN", "Meta EN"),
    ):
        cutoff = cutoffs[family]
        displayed: list[pd.DataFrame] = []
        if include_mean:
            mean = select(summary, "Mean", cutoff)
            draw_curve(axis, mean, "Mean", MEAN_COLOR, "D", linestyle="--", alpha=0.08)
            displayed.append(mean)
        direct_values, meta_values = select(summary, direct, cutoff), select(summary, meta, cutoff)
        draw_curve(axis, direct_values, direct, DIRECT_COLOR, "o")
        draw_curve(axis, meta_values, meta, META_COLOR, "s")
        displayed.extend((direct_values, meta_values))
        axis.set_title(family, fontweight="normal", pad=3.0)
        finish_axis(axis, displayed, show_ylabel=axis is svm_axis)
    figure.text(
        0.575,
        0.89,
        title,
        ha="center",
        va="top",
        fontsize=9.75,
        fontweight="bold",
        fontstyle="italic",
    )
    figure.subplots_adjust(left=0.22, right=0.96, bottom=0.23, top=0.80, wspace=0.48)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = output_stem.with_suffix(".pdf"), output_stem.with_suffix(".svg")
    for output in outputs:
        figure.savefig(output, dpi=300)
    plt.close(figure)
    return outputs


############################### MAIN #########################################


def plot_dataset(dataset_dir: Path) -> list[Path]:
    """Aggregate one dataset and write all source data, cutoffs, and plots."""
    raw, provenance = load_dataset(dataset_dir)
    summary = summarize(raw)
    output_dir = FIGURE_ROOT / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output_dir / "resample_results.parquet", index=False)
    summary.to_csv(output_dir / "generalization_summary.csv", index=False)
    family_starts = {
        "SVM": family_cutoff(summary, "SVM", "Meta SVM"),
        "EN": family_cutoff(summary, "EN", "Meta EN"),
    }
    shared_start = max(family_starts.values())
    cutoffs = {
        "from_10": {"SVM": MINIMUM_PLOT_SIZE, "EN": MINIMUM_PLOT_SIZE},
        "above_mean": {"SVM": shared_start, "EN": shared_start},
    }
    (output_dir / "plot_cutoffs.json").write_text(
        json.dumps(
            {
                "definition": (
                    "The shared start is max(EN-family start, SVM-family start); each "
                    "family start is the first size >=10 where both direct and Meta "
                    "mean MAE are below the resample Mean baseline."
                ),
                "family_thresholds": family_starts,
                "shared_above_mean_start": shared_start,
                **cutoffs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs: list[Path] = []
    title = str(provenance.get("title") or dataset_dir.name)
    for cutoff_name, family_cutoffs in cutoffs.items():
        for suffix, include_mean in (("", False), ("_with_mean_baseline", True)):
            outputs.extend(
                render_panel(
                    summary,
                    title,
                    output_dir / f"{dataset_dir.name}_{cutoff_name}{suffix}",
                    family_cutoffs,
                    include_mean=include_mean,
                )
            )
    return outputs


########################## EVOLUTION PANELS ##################################


def collect_evolution_coefficients() -> pd.DataFrame | None:
    """Collect ten target-species coefficient tables, or report pending work."""
    paths = sorted(RESULT_ROOT.glob("*/evolution_coefficients.tsv"))
    if not paths:
        print(
            "[evolution pending] No *_elasticnetcv inputs were supplied; "
            "few-shot panels are complete."
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
    return frame.loc[:, ["test_species", "component_species", "coefficient"]]


def find_evolution_sidecar(filename: str) -> Path:
    """Find a shared sidecar beside one recorded species input file."""
    candidates: set[Path] = set()
    for provenance_path in RESULT_ROOT.glob("*/shards/size_00.provenance.json"):
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        input_path = Path(str(provenance["input_path"]))
        candidate = input_path.parent / filename
        if candidate.is_file():
            candidates.add(candidate.resolve())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one shared {filename!r} beside the species input files; found "
            f"{sorted(str(path) for path in candidates)}"
        )
    return candidates.pop()


def attach_evolution_metadata(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Attach the 45 pairwise distances and two absolute trait differences."""
    distance_path = find_evolution_sidecar("pairwise_distance.tsv")
    distance_frame = pd.read_csv(distance_path, sep="\t")
    if len(distance_frame) != 45 or set(distance_frame.columns) != {
        "species_a",
        "species_b",
        "distance",
    }:
        raise ValueError(f"{distance_path}: expected 45 species_a/species_b/distance rows")
    distances = {
        frozenset((str(row.species_a), str(row.species_b))): float(row.distance)
        for row in distance_frame.itertuples(index=False)
    }
    if len(distances) != 45:
        raise ValueError(f"{distance_path}: duplicate unordered species pairs")

    life_path = find_evolution_sidecar("life_history.tsv")
    traits = pd.read_csv(life_path, sep="\t").set_index("model_species")
    required_traits = {"maximum_lifespan_years", "gestation_days"}
    species = set(coefficients["test_species"]) | set(coefficients["component_species"])
    if set(traits.index) != species or traits.index.has_duplicates:
        raise ValueError(f"{life_path}: species rows do not match the ten evaluated species")
    if not required_traits.issubset(traits.columns):
        raise ValueError(f"{life_path}: missing {sorted(required_traits - set(traits.columns))}")

    result = coefficients.copy()
    result["distance"] = [
        distances[frozenset((test, component))]
        for test, component in zip(result["test_species"], result["component_species"], strict=True)
    ]
    for source, destination in (
        ("maximum_lifespan_years", "maximum_lifespan_difference_years"),
        ("gestation_days", "gestation_difference_days"),
    ):
        values = pd.to_numeric(traits[source], errors="raise")
        result[destination] = (
            result["test_species"].map(values) - result["component_species"].map(values)
        ).abs()
    numeric = result[
        [
            "coefficient",
            "distance",
            "maximum_lifespan_difference_years",
            "gestation_difference_days",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Evolution relationship table contains non-finite values")
    return result


def spline_model() -> Pipeline:
    """Build the current formal cubic-spline/fixed-alpha trend model."""
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
            resample[index] = spline_model().fit(x[sample, None], y[sample]).predict(grid[:, None])
    lower, upper = np.percentile(resample, (2.5, 97.5), axis=0)
    return pd.DataFrame({"x": grid, "fitted": fitted, "lower": lower, "upper": upper})


def render_evolution_panel(
    source: pd.DataFrame, metric: str, xlabel: str, filename: str, output_dir: Path
) -> list[Path]:
    """Render one of the three formal coefficient/relationship panels."""
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
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / filename
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".svg")]
    for output in outputs:
        figure.savefig(output, dpi=300)
    plt.close(figure)
    trend.to_csv(output_dir / f"{filename}_trend.tsv", sep="\t", index=False)
    return outputs


def plot_evolution_panels() -> list[Path]:
    """Generate all three cross-species panels when all ten species are ready."""
    coefficients = collect_evolution_coefficients()
    if coefficients is None:
        return []
    source = attach_evolution_metadata(coefficients)
    output_dir = FIGURE_ROOT / "evolution"
    output_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(output_dir / "directed_component_coefficients.tsv", sep="\t", index=False)
    panels = (
        (
            "distance",
            "Molecular evolutionary distance",
            "component_en_coef_vs_molecular_evolutionary_distance",
        ),
        (
            "maximum_lifespan_difference_years",
            "|Maximum lifespan difference| (years)",
            "component_en_coef_vs_max_lifespan_difference",
        ),
        (
            "gestation_difference_days",
            "|Gestation time difference| (days)",
            "component_en_coef_vs_gestation_difference",
        ),
    )
    return [
        output
        for metric, xlabel, filename in panels
        for output in render_evolution_panel(source, metric, xlabel, filename, output_dir)
    ]


def main() -> int:
    if not RESULT_ROOT.is_dir():
        raise FileNotFoundError(f"No training results found under {RESULT_ROOT}")
    dataset_dirs = sorted(path for path in RESULT_ROOT.iterdir() if (path / "shards").is_dir())
    if not dataset_dirs:
        raise FileNotFoundError(f"No completed datasets found under {RESULT_ROOT}")
    outputs = [output for dataset_dir in dataset_dirs for output in plot_dataset(dataset_dir)]
    outputs.extend(plot_evolution_panels())
    print(*outputs, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
