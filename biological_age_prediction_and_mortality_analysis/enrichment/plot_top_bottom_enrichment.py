#!/usr/bin/env python3
"""Plot the current top/bottom-1,000 GREAT enrichment panels."""

from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom, false_discovery_control

from repository_paths import PROJECT_ROOT, repo_path

from ..config import ENRICHMENT
from ..plotting import configure_style as configure_manuscript_style

DEFAULT_RESULTS = PROJECT_ROOT / "results/enrichment/rgreat"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/enrichment/figures"
FDR_THRESHOLD = 0.05
TERMS_PER_PANEL = 5


@dataclass(frozen=True)
class LibrarySpec:
    """Display metadata for one tested gene-set library."""

    key: str
    label: str
    color: str


LIBRARY_SPECS = {
    LibrarySpec("go_bp", "GO Biological Process", "#0072B2"),
    LibrarySpec("kegg_2021_human", "KEGG 2021 Human", "#CC79A7"),
    LibrarySpec("reactome_pathways_2024", "Reactome Pathways 2024", "#E69F00"),
    LibrarySpec("msigdb_hallmark_2020", "MSigDB Hallmark 2020", "#D55E00"),
}
LIBRARIES = tuple(
    next(spec for spec in LIBRARY_SPECS if spec.key == key) for key in ENRICHMENT["libraries"]
)
MODEL_LABELS = {"chrono": "Chronological MetaClock", "bio": "Biological MetaClock"}
DIRECTION_LABELS = {"top": "Top 1,000 CpGs", "bottom": "Bottom 1,000 CpGs"}
GO_TERM_LABELS = {
    label.replace("\n", " "): label
    for label in (
        "oncogene-induced\ncell senescence",
        "auditory receptor\ncell fate commitment",
        "inner ear receptor\ncell fate commitment",
        "regulation of\ngap junction assembly",
        "regulation of interleukin-6-mediated\nsignaling pathway",
    )
}


def configure_style() -> None:
    """Apply manuscript vector-output and typography settings."""
    configure_manuscript_style()
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 6,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.75,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.2,
            "xtick.major.width": 0.2,
            "ytick.major.width": 0.2,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_top_terms(path: Path) -> tuple[pd.DataFrame, int]:
    """Validate one result table and return its five lowest-FDR terms."""
    table = pd.read_csv(path)
    required = {"id", "observed_region_hits", "p_value", "p_adjust"}
    if missing := required.difference(table.columns):
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    table["term"] = table.get("description", table["id"]).fillna(table["id"])
    for column in ("observed_region_hits", "p_value", "p_adjust"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    if table.empty or table["term"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{path} has no enrichment terms or contains a blank term.")
    if not np.isfinite(table[["observed_region_hits", "p_value", "p_adjust"]]).all().all():
        raise ValueError(f"{path} contains non-finite enrichment statistics.")
    if not table[["p_value", "p_adjust"]].apply(lambda column: column.between(0, 1)).all().all():
        raise ValueError(f"{path} contains a P value outside [0, 1].")
    significant = int(table["p_adjust"].lt(FDR_THRESHOLD).sum())
    plotted = (
        table.sort_values(["p_adjust", "p_value", "term"], kind="mergesort")
        .head(TERMS_PER_PANEL)
        .copy()
    )
    plotted["minus_log10_fdr"] = -np.log10(plotted["p_adjust"].clip(lower=np.finfo(float).tiny))
    plotted["significant"] = plotted["p_adjust"].lt(FDR_THRESHOLD)
    return plotted.reset_index(drop=True), significant


def result_path(root: Path, model: str, library: str, direction: str) -> Path:
    """Return one rGREAT all-terms table path."""
    return root / model / library / direction / f"{library}_all.csv"


def plot_pair(
    model: str,
    library: LibrarySpec,
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Plot top and bottom foreground results for one clock/library combination."""
    go_bp = library.key == "go_bp"
    model_name = {"chrono": "chronological", "bio": "biological"}[model]
    tables: dict[str, pd.DataFrame] = {}
    significant: dict[str, int] = {}
    for direction in ENRICHMENT["directions"]:
        tables[direction], significant[direction] = read_top_terms(
            result_path(results_dir, model, library.key, direction)
        )
    max_x = max(
        1.55,
        max(float(table["minus_log10_fdr"].max()) for table in tables.values()) + 0.22,
    )
    if not go_bp:
        max_hits = max(float(table["observed_region_hits"].max()) for table in tables.values())
    figure, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 75 / 25.4), sharex=True)
    if go_bp:
        figure.subplots_adjust(left=47 / 150, right=146 / 150, bottom=20 / 75, top=63 / 75, wspace=1)
    else:
        figure.subplots_adjust(left=0.335, right=0.985, bottom=0.25, top=0.78, wspace=0.90)
    threshold = -math.log10(FDR_THRESHOLD)
    for axis, direction in zip(axes, ENRICHMENT["directions"], strict=True):
        table = tables[direction]
        y = np.arange(len(table))
        axis.axvspan(threshold, max_x, color="#009E73", alpha=0.07, zorder=0)
        cutoff_line = axis.axvline(threshold, color="#009E73", linestyle="--", linewidth=0.8,
                                  zorder=1, label=f"FDR = {FDR_THRESHOLD:g}")
        sizes = 32 if go_bp else 14 + 48 * np.sqrt(table["observed_region_hits"] / max_hits)
        colors = [
            to_rgba(library.color, 0.95 if is_significant else 0.38)
            for is_significant in table["p_adjust"].lt(FDR_THRESHOLD)
        ]
        axis.scatter(
            table["minus_log10_fdr"],
            y,
            s=sizes,
            color=colors,
            edgecolor="#222222",
            linewidth=0.2,
            zorder=3,
        )
        terms = table["term"].astype(str)
        if go_bp:
            labels = [
                GO_TERM_LABELS.get(term)
                or textwrap.fill(term, 24, break_long_words=False, break_on_hyphens=False)
                for term in terms
            ]
        else:
            labels = [textwrap.fill(term, 32, break_long_words=False) for term in terms]
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlim(-0.04 * max_x, max_x)
        if go_bp:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            axis.set_xlabel(r"$-\log_{10}$(BH-adjusted $P$)", labelpad=2.0)
            title = {"top": "Top 1000 CpGs", "bottom": "bottom 1000"}[direction]
        else:
            axis.set_xlabel("−log₁₀(BH-adjusted P)", labelpad=2.0)
            title = f"{DIRECTION_LABELS[direction]}\nFDR < 0.05: {significant[direction]}"
        axis.set_title(title, fontweight="bold", pad=4.0)
        axis.grid(axis="x", color="#D0D0D0", linewidth=0.35, alpha=0.75)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", pad=1.5)
        axis.tick_params(axis="y", length=0, pad=2.0)
        if not go_bp and significant[direction] == 0:
            axis.text(
                0.5,
                -0.20,
                "Top-ranked terms shown",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=6,
                color="#666666",
                clip_on=False,
            )
    handles = [cutoff_line]
    if not go_bp:
        hit_values = sorted(
            {int(value) for table in tables.values() for value in table["observed_region_hits"]}
        )
        size_examples = sorted({hit_values[0], hit_values[len(hit_values) // 2], hit_values[-1]})
        handles.extend(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=library.color,
                markeredgecolor="#222222",
                markeredgewidth=0.2,
                markersize=math.sqrt(14 + 48 * math.sqrt(value / max_hits)),
                label=f"Region hits: {value}",
            )
            for value in size_examples
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(handles),
        handletextpad=0.5,
        columnspacing=1.0,
    )
    figure.suptitle(
        f"MetaClock-{model_name}" if go_bp else f"{MODEL_LABELS[model]}: {library.label}",
        fontsize=10 if go_bp else 7,
        fontweight="bold",
        y=0.955,
    )
    library_name = {
        "go_bp": "go_biological_process",
        "kegg_2021_human": "kegg_2021_human",
        "reactome_pathways_2024": "reactome_pathways_2024",
        "msigdb_hallmark_2020": "msigdb_hallmark_2020",
    }[library.key]
    stem = f"enrichment_{model_name}_{library_name}_top_bottom_1000"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    figure.savefig(output_dir / f"{stem}.svg", facecolor="white")
    plt.close(figure)


def summarize_go(results_dir: Path, output_dir: Path) -> None:
    summary, significant = [], []
    for model in ENRICHMENT["models"]:
        for direction in ENRICHMENT["directions"]:
            terms = pd.read_csv(result_path(results_dir, model, "go_bp", direction))
            p = binom.sf(terms.observed_region_hits - 1, ENRICHMENT["top_n"], terms.genome_fraction)
            np.testing.assert_allclose(terms.p_value, p, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(terms.p_adjust, false_discovery_control(terms.p_value, method="bh"),
                                       rtol=1e-10, atol=1e-12)
            hits = terms.loc[terms.p_adjust < FDR_THRESHOLD].copy()
            hits.insert(0, "direction", direction)
            hits.insert(0, "model", model)
            significant.append(hits)
            top = terms.sort_values(["p_adjust", "p_value", "id"]).iloc[0]
            summary.append({"model": model, "direction": direction, "terms": len(terms),
                            "significant_fdr05": len(hits), "top_id": top.id,
                            "top_description": top.description, "top_p_adjust": top.p_adjust})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(significant, ignore_index=True).to_csv(output_dir / "significant_go_bp.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=repo_path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=repo_path, default=DEFAULT_OUTPUT)
    parser.add_argument("--library", choices=["all", *ENRICHMENT["libraries"]], default="go_bp")
    args = parser.parse_args()
    configure_style()
    for library in LIBRARIES:
        if args.library not in ("all", library.key):
            continue
        for model in ENRICHMENT["models"]:
            plot_pair(model, library, args.results_dir, args.output_dir)
    if args.library in ("go_bp", "all"):
        summarize_go(args.results_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
