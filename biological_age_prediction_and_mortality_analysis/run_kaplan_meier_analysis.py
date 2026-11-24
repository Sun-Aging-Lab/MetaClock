"""Use published FHS and WHI Kaplan–Meier panels, redraw local estimates, or fit local cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import MORTALITY_MODELS
from .plotting import clean_axis, configure_style, ordered_rows, save_vector_pair

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "kaplan_meier"
GROUP_ORDER = ("bottom_20", "top_20")
GROUP_COLORS = {"bottom_20": "#C96B6B", "top_20": "#5B7FAE"}
PRIMARY_COVARIATES = {
    "fhs": ("age", "female", "batch_umn"),
    "whi": ("age", "race_black", "race_hispanic"),
}


def verify_bundled_panels(output_dir: Path = OUTPUT_DIR) -> None:
    for cohort in MORTALITY_MODELS:
        for suffix in ("curves.pdf", "curves.svg", "logrank_and_deaths.csv"):
            path = output_dir / f"{cohort}_km_{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"Missing published Kaplan–Meier file: {path}")


def _read_published_input(path: Path, checksums: dict[str, str]) -> pd.DataFrame:
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != checksums[path.name]:
            raise ValueError(f"KM input differs from the manuscript analysis: {path.name}")
        handle.seek(0)
        return pd.read_csv(handle)


def plot_km(cohort: str, curves: pd.DataFrame, tests: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
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
    save_vector_pair(figure, output_dir, f"{cohort}_km_curves")


def redraw_private_curves(private_curve_dir: Path, output_dir: Path = OUTPUT_DIR) -> None:
    """Publish both panels once all inputs and rendered files are ready."""
    private_curve_dir = private_curve_dir.expanduser().resolve()
    checksums = json.loads((output_dir / "figure_source.json").read_text())["plot_input_sha256"]
    with TemporaryDirectory(prefix=".km_render_", dir=output_dir) as temporary:
        dest = Path(temporary)
        for cohort in MORTALITY_MODELS:
            curves = _read_published_input(private_curve_dir / f"{cohort}_km_curve_points.csv", checksums)
            stats = _read_published_input(output_dir / f"{cohort}_km_logrank_and_deaths.csv", checksums)
            plot_km(cohort, curves, stats, dest)
        for panel in dest.iterdir():
            panel.replace(output_dir / panel.name)
    print("Redrew the FHS and WHI Kaplan–Meier panels from the local curve estimates.")


def _read_cohort(cohort: str, private_cohort_dir: Path, cohort_source: dict) -> pd.DataFrame:
    pheno_path = private_cohort_dir / "inputs" / f"{cohort}_pheno.csv"
    prediction_path = private_cohort_dir / "predictions" / f"{cohort}_predictions.csv"
    pheno_columns = ["sample_id", "time_years", "event", *PRIMARY_COVARIATES[cohort]]
    score_columns = ["sample_id", *MORTALITY_MODELS[cohort]]
    pheno = pd.read_csv(pheno_path, usecols=pheno_columns, dtype={"sample_id": str})
    predictions = pd.read_csv(prediction_path, usecols=score_columns, dtype={"sample_id": str})
    data = pheno.merge(predictions, on="sample_id", how="inner", validate="one_to_one")
    if len(data) != len(pheno) or len(data) != len(predictions) or data.sample_id.isna().any():
        raise ValueError(f"{cohort.upper()} requires a matching sample ID in both local files.")
    if (not np.isfinite(data.drop(columns="sample_id").to_numpy(float)).all()
            or not data.event.isin((0, 1)).all() or data.time_years.lt(0).any()):
        raise ValueError(f"Invalid score, covariate, follow-up time, or death indicator in {cohort.upper()}.")
    if len(data) != cohort_source["n"] or data.event.sum() != cohort_source["deaths"]:
        raise ValueError(f"{cohort.upper()} cohort size or deaths differ from the published analysis.")
    return data


def _refit_cohort(cohort: str, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the adjusted quintile comparison using the manuscript's sample-ID tie break."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from sklearn.linear_model import LinearRegression

    covariates = PRIMARY_COVARIATES[cohort]
    stats, curves = [], []
    adjustment = data[list(covariates)].astype(float).to_numpy()
    for model in MORTALITY_MODELS[cohort]:
        fit = LinearRegression().fit(adjustment, data[model].to_numpy(float))
        residual = data[model].to_numpy(float) - fit.predict(adjustment)
        ordered = data.assign(_residual=residual).sort_values(
            ["_residual", "sample_id"], kind="mergesort"
        )
        n_per_group = int(round(0.2 * len(ordered)))
        low, high = ordered.head(n_per_group), ordered.tail(n_per_group)
        test = logrank_test(
            low.time_years, high.time_years,
            event_observed_A=low.event, event_observed_B=high.event,
        )
        low_events, high_events = int(low.event.sum()), int(high.event.sum())
        stats.append({
            "model": model, "n_per_group": n_per_group,
            "lowest_20_deaths": low_events, "highest_20_deaths": high_events,
            "delta_deaths_highest_minus_lowest": high_events - low_events,
            "logrank_chi2": float(test.test_statistic), "logrank_p": float(test.p_value),
            "residual_covariates": "+".join(covariates),
        })
        for group, subset in (("bottom_20", low), ("top_20", high)):
            km = KaplanMeierFitter().fit(subset.time_years, subset.event)
            curve = km.survival_function_.join(km.confidence_interval_).reset_index()
            curve.columns = ["time_years", "survival", "ci_low", "ci_high"]
            curves.append(curve.assign(group=group, model=model, cohort=cohort))
    return pd.concat(curves, ignore_index=True), pd.DataFrame(stats)


def _verify_refit(cohort: str, observed: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Check every numerical annotation against the manuscript before rendering."""
    observed = observed.set_index("model").loc[list(MORTALITY_MODELS[cohort])]
    expected = expected.set_index("model").loc[list(MORTALITY_MODELS[cohort])]
    exact = ("n_per_group", "lowest_20_deaths", "highest_20_deaths",
             "delta_deaths_highest_minus_lowest", "residual_covariates")
    for column in exact:
        if not observed[column].equals(expected[column]):
            raise ValueError(f"{cohort.upper()} refit differs in published {column} annotations.")
    for column in ("logrank_chi2", "logrank_p"):
        if not np.allclose(observed[column], expected[column], rtol=1e-9, atol=0.0):
            raise ValueError(f"{cohort.upper()} refit differs in published {column} annotations.")


def refit_private_cohorts(private_cohort_dir: Path, output_dir: Path = OUTPUT_DIR) -> None:
    """Refit authorized cohorts in memory and publish panels only after validation."""
    private_cohort_dir = private_cohort_dir.expanduser().resolve()
    source = json.loads((output_dir / "figure_source.json").read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix=".km_refit_", dir=output_dir) as temporary:
        dest = Path(temporary)
        for cohort in MORTALITY_MODELS:
            data = _read_cohort(cohort, private_cohort_dir, source["cohorts"][cohort])
            curves, stats = _refit_cohort(cohort, data)
            expected = _read_published_input(
                output_dir / f"{cohort}_km_logrank_and_deaths.csv", source["plot_input_sha256"]
            )
            _verify_refit(cohort, stats, expected)
            stats = stats.merge(expected[["model", "title"]], on="model", validate="one_to_one")
            # The manuscript renderer read decimal CSV estimates. Match its rounding in
            # memory so the same step boundaries are drawn at publication resolution.
            plot_curves = pd.read_csv(StringIO(curves.to_csv(index=False)))
            plot_km(cohort, plot_curves, stats, dest)
        for cohort in MORTALITY_MODELS:
            for suffix in ("pdf", "svg"):
                name = f"{cohort}_km_curves.{suffix}"
                (dest / name).replace(output_dir / name)
    print("Refit and redrew the FHS and WHI Kaplan–Meier panels from authorized local cohorts.")


def render_or_use(private_curve_dir: Path | None = None,
                  private_cohort_dir: Path | None = None) -> None:
    """Use bundled panels, or explicitly redraw both from authorized local inputs."""
    if private_curve_dir is not None and private_cohort_dir is not None:
        raise ValueError("Choose either private curve inputs or private cohort inputs, not both.")
    if private_cohort_dir is not None:
        refit_private_cohorts(private_cohort_dir)
    elif private_curve_dir is None:
        verify_bundled_panels()
        print("Using bundled FHS and WHI Kaplan–Meier panels.")
    else:
        redraw_private_curves(private_curve_dir)


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--private-curve-dir", type=Path, metavar="DIR",
        help="Redraw from the two local cohort-specific curve-point CSVs.",
    )
    inputs.add_argument(
        "--private-cohort-dir", type=Path, metavar="DIR",
        help="Fit local inputs/{fhs,whi}_pheno.csv and predictions/{fhs,whi}_predictions.csv.",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    args = parser.parse_args(argv)
    render_or_use(args.private_curve_dir, args.private_cohort_dir)


if __name__ == "__main__":
    main()
