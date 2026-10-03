#!/usr/bin/env python3
"""Train Figure-2 chronological-age architectures and draw every comparison plot.

The bundled training and test CSV files contain ``dataset``, ``sample_id``,
``age``, and the 16 component-clock columns listed in ``COMPONENT_CLOCKS``.
Each test dataset becomes one scatter panel.
"""

from __future__ import annotations

from pathlib import Path

from repository_matplotlib import MPLCONFIGDIR
from repository_paths import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from generalization.fonts import register_fonts

# ---------------------------------------------------------------------------
# Bundled inputs and fixed Figure-2 settings
# ---------------------------------------------------------------------------
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINING_DATASET = MODULE_DIR / "training_component_clock_predictions.csv"
DEFAULT_TEST_DATASET = MODULE_DIR / "test_component_clock_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/chronological_age_prediction"

COMPONENT_CLOCKS = (
    "horvath2013",
    "hannum",
    "dnamphenoage",
    "altumage",
    "han",
    "encen40",
    "hrsinchphenoage",
    "lin",
    "skinandblood",
    "yingcausage",
    "zhangblup",
    "zhangen",
    "pchorvath2013",
    "pchannum",
    "pcphenoage",
    "pcskinandblood",
)
DISPLAYED_COMPONENTS = tuple(clock for clock in COMPONENT_CLOCKS if clock != "yingcausage")
COMPONENT_PANEL_LABELS = (*DISPLAYED_COMPONENTS, "MetaClock")
ARCHITECTURES = (
    "ElasticNet",
    "Lasso",
    "Ridge",
    "RF",
    "SVR",
    "LightGBM",
    "MLP",
    "Mean",
    "SpectralEnsemble",
)
SUPERVISED_ARCHITECTURES = ARCHITECTURES[:7]
MODEL_LABELS = {"ElasticNet": "EN", "LightGBM": "LGBM", "SVR": "SVM", "SpectralEnsemble": "SE"}
COMPONENT_LABELS = {
    "altumage": "AltumAge",
    "dnamphenoage": "DNAmPhenoAge",
    "encen40": "EN-Cen40",
    "han": "Han",
    "hannum": "Hannum",
    "horvath2013": "Horvath",
    "hrsinchphenoage": "HRSInCHPhenoAge",
    "lin": "Lin",
    "pchannum": "PC Hannum",
    "pchorvath2013": "PC Horvath",
    "pcphenoage": "PC PhenoAge",
    "pcskinandblood": "PC Skin&Blood",
    "skinandblood": "Skin&Blood",
    "zhangblup": "Zhang BLUP",
    "zhangen": "Zhang EN",
}
TAB20 = tuple(matplotlib.colors.to_hex(color) for color in matplotlib.colormaps["tab20"].colors)
META_COLOR = TAB20[8]
FIGURE_MM = (180.0, 65.0)
PANEL_SIZE_MM = 23.0 * FIGURE_MM[0] / 170.0
PANEL_LEFTS_MM = tuple(value * FIGURE_MM[0] / 170.0 for value in (13.0, 44.0, 75.0, 106.0))
PANEL_BOTTOM_MM = 18.0 * FIGURE_MM[0] / 170.0
ARCHITECTURE_COLORS = {
    name: TAB20[index] for name, index in zip(ARCHITECTURES, range(0, 18, 2), strict=True)
}
ARCHITECTURE_COLORS["SVR"] = META_COLOR
COMPONENT_COLORS = dict(
    zip(
        sorted(DISPLAYED_COMPONENTS),
        (color for index, color in enumerate(TAB20) if index != 8),
        strict=False,
    )
)
COMPONENT_COLORS["MetaClock"] = META_COLOR


# ---------------------------------------------------------------------------
# Input validation and estimators
# ---------------------------------------------------------------------------
def read_dataset(path: Path) -> pd.DataFrame:
    """Read a wide clock-output table and enforce the public input schema."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"dataset", "sample_id", "age", *COMPONENT_CLOCKS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame["sample_id"].isna().any() or frame.duplicated(["dataset", "sample_id"]).any():
        raise ValueError(f"{path} must have unique non-missing dataset/sample_id pairs")
    numeric = frame.loc[:, ["age", *COMPONENT_CLOCKS]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} has missing, non-numeric, or infinite ages/component predictions")
    if frame["dataset"].isna().any():
        raise ValueError(f"{path} has missing dataset labels")
    return pd.concat([frame.loc[:, ["dataset", "sample_id"]], numeric], axis=1)


def _cv(n_samples: int) -> KFold:
    if n_samples < 10:
        raise ValueError("At least ten training samples are required for five-fold model selection")
    return KFold(n_splits=5, shuffle=True, random_state=42)


def sklearn_architectures(n_samples: int) -> dict[str, object]:
    """Return the six non-neural Figure-2 architectures.

    LightGBM is the actual architecture rather than an sklearn substitute.
    """
    alphas = np.logspace(-5, 0, 30)
    cv = _cv(n_samples)
    return {
        "ElasticNet": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNetCV(alphas=alphas, cv=cv, max_iter=20_000, random_state=42)),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LassoCV(alphas=alphas, cv=cv, max_iter=20_000, random_state=42)),
            ]
        ),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=alphas, cv=cv))]),
        "RF": RandomForestRegressor(random_state=42),
        "SVR": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="linear"))]),
        "LightGBM": LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
    }


class TorchMLP(torch.nn.Module):
    """The fixed 16->100->50->1 architecture used for the Figure-2 MLP."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_features, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(100, 50),
            torch.nn.BatchNorm1d(50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def fit_mlp(features: np.ndarray, age: np.ndarray) -> tuple[TorchMLP, StandardScaler]:
    """Fit the documented 128-sample minibatch torch MLP."""
    torch.manual_seed(105)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(105)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler().fit(features)
    x = torch.tensor(scaler.transform(features), dtype=torch.float32)
    y = torch.tensor(age, dtype=torch.float32)
    batch_size = min(128, len(x))
    while batch_size > 2 and len(x) % batch_size == 1:
        batch_size -= 1
    generator = torch.Generator().manual_seed(105)
    batches = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = TorchMLP(x.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(500):
        model.train()
        for batch_x, batch_y in batches:
            optimiser.zero_grad()
            loss = torch.nn.functional.mse_loss(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimiser.step()
    model.eval()
    return model, scaler


def spectral_weights(matrix: np.ndarray) -> np.ndarray:
    """Compute the non-negative leading-eigenvector component ensemble weights."""
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(centered @ centered.T / max(matrix.shape[1] - 1, 1))
    weights = np.abs(vectors[:, np.argmax(values)])
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Training, held-out inference, and standard metrics
# ---------------------------------------------------------------------------
def metric(age: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Calculate standard held-out MAE and two-sided Pearson r."""
    if len(age) < 3 or np.std(age) == 0 or np.std(prediction) == 0:
        raise ValueError("Each test dataset needs at least three non-constant observations")
    return float(mean_absolute_error(age, prediction)), float(pearsonr(age, prediction).statistic)


def train_and_evaluate(
    training: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all architectures once, infer every test dataset, and return Figure-2 tables."""
    x_train = training.loc[:, COMPONENT_CLOCKS].to_numpy(dtype=float)
    y_train = training["age"].to_numpy(dtype=float)
    fitted = sklearn_architectures(len(training))
    for estimator in fitted.values():
        estimator.fit(x_train, y_train)
    mlp, mlp_scaler = fit_mlp(x_train, y_train)

    architecture_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for dataset, panel in test.groupby("dataset", sort=True):
        x = panel.loc[:, COMPONENT_CLOCKS].to_numpy(dtype=float)
        y = panel["age"].to_numpy(dtype=float)
        predictions = {
            name: np.asarray(estimator.predict(x), dtype=float)
            for name, estimator in fitted.items()
        }
        with torch.no_grad():
            predictions["MLP"] = (
                mlp(
                    torch.tensor(mlp_scaler.transform(x), dtype=torch.float32).to(
                        next(mlp.parameters()).device
                    )
                )
                .cpu()
                .numpy()
            )
        predictions["Mean"] = x.mean(axis=1)
        predictions["SpectralEnsemble"] = x @ spectral_weights(x.T)
        for name in ARCHITECTURES:
            mae, correlation = metric(y, predictions[name])
            architecture_rows.append(
                {
                    "dataset": dataset,
                    "model": name,
                    "n": len(y),
                    "mae": mae,
                    "pearson_r": correlation,
                }
            )
        for index, clock in enumerate(COMPONENT_CLOCKS):
            mae, correlation = metric(y, x[:, index])
            component_rows.append(
                {
                    "dataset": dataset,
                    "predictor": clock,
                    "n": len(y),
                    "mae": mae,
                    "pearson_r": correlation,
                }
            )
        meta_mae, meta_correlation = metric(y, predictions["SVR"])
        component_rows.append(
            {
                "dataset": dataset,
                "predictor": "MetaClock",
                "n": len(y),
                "mae": meta_mae,
                "pearson_r": meta_correlation,
            }
        )
    architectures = pd.DataFrame(architecture_rows)
    components = pd.DataFrame(component_rows)
    comparison = pd.concat(
        [
            components.loc[components["predictor"].isin(DISPLAYED_COMPONENTS)]
            .rename(columns={"predictor": "label"})
            .assign(group="Single clock"),
            architectures.loc[architectures["model"].isin(SUPERVISED_ARCHITECTURES)]
            .rename(columns={"model": "label"})
            .assign(group="MetaClock architecture"),
        ],
        ignore_index=True,
    ).loc[:, ["dataset", "group", "label", "n", "mae", "pearson_r"]]
    return architectures, components, comparison


# ---------------------------------------------------------------------------
# Figure-2 scatter plotting (architecture, component, and boundary panel)
# ---------------------------------------------------------------------------
def configure_style() -> None:
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    register_fonts()
    plt.rcParams.update(
        {
            "font.family": ["Liberation Sans"],
            "font.sans-serif": ["Liberation Sans"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.75,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 8.5,
            "axes.linewidth": 0.2,
            "xtick.major.width": 0.2,
            "ytick.major.width": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def axes_for(datasets: list[str]) -> tuple[plt.Figure, np.ndarray]:
    """Create the fixed four-panel layout or one subplot per supplied dataset."""
    figure = plt.figure(figsize=(FIGURE_MM[0] / 25.4, FIGURE_MM[1] / 25.4))
    if len(datasets) == 4:
        axes = [
            figure.add_axes(
                [
                    left / FIGURE_MM[0],
                    PANEL_BOTTOM_MM / FIGURE_MM[1],
                    PANEL_SIZE_MM / FIGURE_MM[0],
                    PANEL_SIZE_MM / FIGURE_MM[1],
                ]
            )
            for left in PANEL_LEFTS_MM
        ]
    else:
        axes = figure.subplots(1, len(datasets), squeeze=False).ravel().tolist()
    return figure, np.asarray(axes)


def finish(figure: plt.Figure, axes: np.ndarray, datasets: list[str]) -> None:
    """Apply dataset titles and shared MAE/Pearson labels to the metric panels."""
    for index, (axis, dataset) in enumerate(zip(axes, datasets, strict=True)):
        axis.set_title(dataset)
        axis.set_ylabel("Pearson r" if index == 0 else "")
        axis.tick_params(axis="both", pad=1.5)
        axis.margins(x=0.10, y=0.14)
    left, right = axes[0].get_position().x0, axes[-1].get_position().x1
    figure.text((left + right) / 2, 3.0 / FIGURE_MM[1], "MAE (years)", ha="center")
    if len(datasets) != 4:
        figure.tight_layout(rect=(0, 0.08, 0.82, 1))


def save_pair(figure: plt.Figure, stem: str) -> None:
    """Save matching PDF/SVG outputs and close the figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{stem}.pdf")
    figure.savefig(OUTPUT_DIR / f"{stem}.svg")
    plt.close(figure)


def scatter_metrics(
    table: pd.DataFrame,
    label_column: str,
    labels: tuple[str, ...],
    colors: dict[str, str],
    stem: str,
    title: str,
    star: str | None = None,
) -> None:
    """Draw one MAE--Pearson panel set from held-out metrics."""
    datasets = sorted(table["dataset"].unique())
    figure, axes = axes_for(datasets)
    for axis, dataset in zip(axes, datasets, strict=True):
        panel = table.loc[table["dataset"].eq(dataset)].set_index(label_column)
        for label in labels:
            is_star = label == star
            axis.scatter(
                panel.at[label, "mae"],
                panel.at[label, "pearson_r"],
                marker="*" if is_star else "o",
                s=85.5 if is_star else (20 if star is None else 17),
                color=colors[label],
                edgecolor="#222",
                linewidth=0.2,
                alpha=1 if is_star else 0.9,
            )
    handles = [
        Line2D(
            [0],
            [0],
            marker="*" if label == star else "o",
            color="none",
            markerfacecolor=colors[label],
            markeredgecolor="#222",
            markersize=9 if label == star else (3.8 if star is None else 3.5),
            label=MODEL_LABELS.get(label, COMPONENT_LABELS.get(label, label)),
        )
        for label in labels
    ]
    figure.legend(
        handles=handles,
        title=title,
        loc="center left",
        bbox_to_anchor=(0.83, 0.5),
        frameon=False,
        fontsize=7.5,
    )
    finish(figure, axes, datasets)
    save_pair(figure, stem)


def plot_boundary(comparison: pd.DataFrame) -> None:
    """Draw the formal borderplot: a dashed box around seven architecture stars."""
    datasets = sorted(comparison["dataset"].unique())
    figure, axes = axes_for(datasets)
    for axis, dataset in zip(axes, datasets, strict=True):
        panel = comparison.loc[comparison["dataset"].eq(dataset)]
        components = panel.loc[panel["group"].eq("Single clock")].set_index("label")
        architectures = panel.loc[panel["group"].eq("MetaClock architecture")].set_index("label")
        for label in DISPLAYED_COMPONENTS:
            axis.scatter(
                components.at[label, "mae"],
                components.at[label, "pearson_r"],
                s=17,
                color=COMPONENT_COLORS[label],
                edgecolor="#222",
                linewidth=0.2,
                alpha=0.55,
            )
        x0, x1 = architectures["mae"].min(), architectures["mae"].max()
        y0, y1 = architectures["pearson_r"].min(), architectures["pearson_r"].max()
        x_span = float(panel["mae"].max() - panel["mae"].min())
        y_span = float(panel["pearson_r"].max() - panel["pearson_r"].min())
        x_pad = max((x1 - x0) * 0.16, 0.04 * x_span)
        y_pad = max((y1 - y0) * 0.16, 0.08 * y_span)
        axis.add_patch(
            Rectangle(
                (x0 - x_pad, y0 - y_pad),
                (x1 - x0) + 2 * x_pad,
                (y1 - y0) + 2 * y_pad,
                facecolor=(0.92, 0.92, 0.92, 0.22),
                edgecolor="#555",
                linestyle=(0, (4, 2.5)),
                linewidth=0.5,
                zorder=0,
            )
        )
        for label in SUPERVISED_ARCHITECTURES:
            axis.scatter(
                architectures.at[label, "mae"],
                architectures.at[label, "pearson_r"],
                marker="*",
                s=85.5,
                color=ARCHITECTURE_COLORS[label],
                edgecolor="#111",
                linewidth=0.2,
            )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=META_COLOR,
                markeredgecolor="#111",
                markersize=9,
                label="MetaClock architecture",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#999",
                markeredgecolor="#222",
                markersize=5,
                label="Component model",
            ),
        ],
        loc="center left",
        bbox_to_anchor=(0.83, 0.5),
        frameon=False,
        fontsize=7,
    )
    finish(figure, axes, datasets)
    save_pair(figure, "pearson_mae_submodel_architectures")


def main() -> None:
    configure_style()
    architectures, components, comparison = train_and_evaluate(
        read_dataset(DEFAULT_TRAINING_DATASET), read_dataset(DEFAULT_TEST_DATASET)
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    architectures.to_csv(OUTPUT_DIR / "architecture_metrics.csv", index=False)
    components.to_csv(OUTPUT_DIR / "component_clock_metrics.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "architecture_comparison_metrics.csv", index=False)
    scatter_metrics(
        architectures,
        "model",
        ARCHITECTURES,
        ARCHITECTURE_COLORS,
        "pearson_mae_model_architecture",
        "Model architecture",
        star=None,
    )
    scatter_metrics(
        components.loc[components["predictor"].isin(COMPONENT_PANEL_LABELS)],
        "predictor",
        COMPONENT_PANEL_LABELS,
        COMPONENT_COLORS,
        "pearson_mae_submodel",
        "Component clock",
        star="MetaClock",
    )
    plot_boundary(comparison)


if __name__ == "__main__":
    main()
