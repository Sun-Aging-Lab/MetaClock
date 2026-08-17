# MetaClock analysis workflows

This repository contains compact, reproducible workflows for chronological-age
prediction, cross-domain generalization, biological-age and disease analysis,
functional enrichment, and intervention analysis. Raw methylation matrices and
cohort metadata are not bundled. Every external dataset input
uses the explicit placeholder
`Path("/path/to/your/file")  # fill in your path`.

## Project overview

MetaClock is a modular DNA-methylation analysis framework that combines
predictions from multiple established epigenetic clocks into compact meta
models. This repository organizes the reproducible downstream workflows used
to evaluate chronological-age prediction, quantify generalization across array
platforms, tissues, and species, and study biological-age associations with
disease and interventions.

## Project layout

```text
chronological_age_prediction_performance/          chronological-age prediction metrics and scatter plots
generalization/
├── array/                                         array generalization
├── tissue/                                        tissue generalization
└── species/                                       species generalization and three evolutionary plots
biological_age_prediction_and_mortality_analysis/  biological-age, disease, survival, and enrichment
intervention_analysis/                             intervention effects and FDR volcano plot
```


## Environment

Install [uv](https://docs.astral.sh/uv/) and create the locked Python 3.11
environment from the repository root:

```bash
uv sync --frozen
uv run --frozen pytest
uv run --frozen ruff check .
```

Python dependency versions are pinned in `pyproject.toml` and fully resolved in
`uv.lock`. The GREAT enrichment workflow additionally uses R 4.4, Bioconductor
3.20, `rGREAT` 2.8.0, `data.table`, `GenomicRanges`, `GenomeInfoDb`, `minfi`,
`rtracklayer`, `AnnotationDbi`, `org.Hs.eg.db`, and the 27K v1.2, 450K v1.2,
and EPIC v1.0 B4 hg19 annotation packages documented in the enrichment README;
R dependencies are therefore recorded separately from the Python lockfile.

