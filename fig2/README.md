# Figure 2 — generalization

This directory contains the compact code and result tables for the manuscript
generalization panels.  Panels are ordered as array, tissue, and species:

1. GSE234461 (EPIC array)
2. GSE184221 (Mammalian MethylChip40 array)
3. GSE111629 (whole blood)
4. *Bos taurus* (cross-species)

`results/01_*.tsv` through `results/03_*.tsv` contain the plotted means,
between-repeat SDs, and empirical 95% intervals.  They are compact derivatives
of 50 repeated few-shot fits and are sufficient to redraw every data-driven
panel.  The `Mean` rows are the repeated training-subset mean baseline.

## Run

Plot all four panels from the retained compact tables:

```bash
python fig2/scripts/plot_generalization.py
```

Prepare or rerun one dataset after placing your data in the relative directory
listed in `datasets/manifest.tsv`:

```bash
python fig2/scripts/run_generalization.py prepare \
  --dataset-dir fig2/datasets/array/GSE234461
python fig2/scripts/run_generalization.py evaluate \
  --dataset-dir fig2/datasets/array/GSE234461 \
  --output fig2/results/01_array_generalization.tsv
```

`evaluate` uses `datasets/manifest.tsv` to add panel metadata and apply the
declared minimum displayed training size. Its output has the same compact
schema consumed by `plot_generalization.py`. When two datasets share one table
(the array panels), rerun both against the same output; each run replaces only
its own dataset rows.

The species panel uses a distinct legacy pipeline and must be run separately:

```bash
python fig2/scripts/run_species_generalization.py \
  --dataset-dir fig2/datasets/species/BosTaurus \
  --species BosTaurus \
  --output fig2/results/03_species_generalization.tsv
```

The commands expose the appropriate `--dataset-dir`, `--output`,
`--results-dir`, or `--output-dir` overrides. No machine-specific source path
is embedded in the code.

## Fixed analysis parameters

- Array/tissue CpG ranking: `mutual_info_regression(n_neighbors=3, random_state=0,
  n_jobs=-1)`, top 10,000.
- Array/tissue redundancy reduction: `mrmr-selection` v0.2.8
  `mrmr_regression(K=1000)`, using
  F-statistic relevance (`relevance="f"`), correlation redundancy
  (`redundancy="c"`), `denominator="mean"`, `only_same_domain=False`,
  `return_scores=False`, and `n_jobs=-1`.
- Split: one fixed 80/20 train-pool/test split, `random_state=105`.
- Training sizes: 30 log-spaced sizes from 10 to `n_train - 10`.
- Repeats: 50 subsets per size; seed `42 + repeat + size * 1000`.
- Array/tissue CpG EN and Meta EN: standardized inputs and
  `ElasticNet(alpha=0.001, l1_ratio=0.5, selection="cyclic")`.
- Array/tissue CpG SVM and Meta SVM: standardized inputs and
  `SVR(kernel="linear", C=1.0, epsilon=0.1)`.
- Species direct models: all zero-imputed conserved CpGs; no human MI/mRMR and
  no StandardScaler.  Species Meta models: only per-species MLP-ELU prediction
  columns ending in `_mlp_elu`; no StandardScaler.  Both branches directly fit
  `ElasticNet(alpha=0.001, l1_ratio=0.5)` and
  `SVR(kernel="linear", C=1.0, epsilon=0.1)`.
- Missing values: constant-zero imputation, matching the retained workflow.
- Uncertainty ribbon: empirical 2.5th and 97.5th percentiles across repeats.

Array/tissue feature selection was performed on the full healthy-only cohort
in the legacy analysis.  That behavior is retained for exact comparability and
should be described as such; a new benchmark should move selection inside each
training split.  It does not apply to the species panel.

Additional eligible tissue cohorts (GSE232891, GSE287331 and GSE175458) and the
other nine species can be added as new rows in `datasets/manifest.tsv`.  They
are intentionally not included in these minimal manuscript tables.
