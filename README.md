# MetaClock analysis workflows

This repository contains compact, reproducible workflows for chronological-age
prediction, cross-domain generalization, biological-age and mortality analysis,
functional enrichment, and intervention analysis. Raw methylation matrices,
fitted model artifacts, and cohort metadata are intentionally not bundled.
Every dataset/model input is a project-relative placeholder marked
`# fill in your path`; it can be replaced directly and, where exposed by an
entry point, overridden with a command-line option.

## Project overview

MetaClock is a modular DNA-methylation analysis framework that combines
predictions from multiple established epigenetic clocks into compact meta
models. This repository organizes the reproducible downstream workflows used
to evaluate chronological-age prediction, quantify generalization across array
platforms, tissues, and species, and study biological-age associations with
disease, mortality, and interventions. It also includes CpG-weight enrichment
analysis for interpreting the biological processes represented by the component
models. Training, aggregation, statistical testing, plotting, and validation
are separated into focused scripts so that each stage can be rerun or audited
independently.

## Project layout

```text
chronological_age_prediction_performance/          chronological-age prediction metrics and scatter plots
generalization/
├── array/                                         array generalization
├── tissue/                                        tissue generalization
└── species/                                       species generalization and three evolutionary plots
biological_age_prediction_and_mortality_analysis/  prediction, disease, mortality, and enrichment
intervention_analysis/                             intervention effects and FDR volcano plot
```

Each directory has a focused README describing input columns, outputs, fixed
analysis parameters, and example commands. Scripts expose typed functions as
well as a command-line entry point and create output directories when needed.

## Environment

Install [uv](https://docs.astral.sh/uv/) and create the locked Python 3.11
environment from the repository root:

```bash
uv sync --frozen
uv run --frozen python validate_project.py
uv run --frozen pytest
uv run --frozen ruff check .
```

Python dependency versions are pinned in `pyproject.toml` and fully resolved in
`uv.lock`. The GREAT enrichment workflow additionally uses R 4.4, Bioconductor
3.20, `rGREAT` 2.8.0, `data.table`, `GenomicRanges`, `GenomeInfoDb`, `minfi`,
`rtracklayer`, `AnnotationDbi`, `org.Hs.eg.db`, and the 27K v1.2, 450K v1.2,
and EPIC v1.0 B4 hg19 annotation packages documented in the enrichment README;
R dependencies are therefore recorded separately from the Python lockfile.

## Reproducibility conventions

- Training-size generalization uses `StandardScaler` within every resample.
  Elastic Net uses five-fold `ElasticNetCV`, 30 log-spaced alphas from `1` to
  `1e-5`, `max_iter=20000`, 30 log-spaced training sizes beginning at at least
  10, and 50 resamples per size. Species component predictions are MLP-ELU
  predictions. Evolution relationships reproduce their formal alpha range of
  `10` to `1e-5`; their spline basis is not scaled a second time.
- Plotting scripts write matching PDF and SVG vector outputs with consistent
  typography and fixed canvas sizes.
- Biological and intervention alpha selection wrap the complete
  `Pipeline(StandardScaler, ElasticNet)` in `GridSearchCV`, so scaling is fit
  independently in each fold. Any convergence warning rejects the fit.
- Biological component lists reject `dnamfitage` explicitly and validate the
  exact ordered feature manifests.
- Enrichment uses the 27K + 450K + EPIC v1.0 B4 union as the GREAT universe and
  top/bottom 1,000 CpGs without a foreground-array restriction. The manifest
  records 16 chronological and 19 biological candidate clocks; only the 8 and
  17 clocks that pass the fixed-affine audit contribute CpG weights.
- Statistical tests, scaling, model selection, survival analysis, and multiple
  testing correction use maintained scientific packages rather than local
  reimplementations.

Run any entry point with `--help` to inspect its expected relative inputs. The
repository validator checks directory structure, syntax, path portability, and
the required input-path annotations without executing long model fits.
