# Intervention analysis

This directory contains the compact, reproducible workflow used to train the
biological MetaClock, predict intervention samples, estimate donor-level
effects, apply the curated one-intervention/one-comparison rule, and draw the
FDR volcano plot.

## Scripts

- `train_intervention_biometaclock.py`: reproduces the canonical two-stage
  fit used for the current intervention results. It fits
  `Pipeline(StandardScaler, ElasticNetCV)` with the scaler outside the five
  shuffled folds, searches 30 alphas from `1` to `1e-5`, uses the default
  `l1_ratio=0.5`, records (but does not reject) CV convergence warnings, and
  performs a fresh all-participant `Pipeline(StandardScaler, ElasticNet)`
  refit at the selected alpha. Refit warnings and reaching `100000` iterations
  are fatal. `dnamfitage` is rejected.
- `predict_intervention_biological_age.py`: applies the trained model to an
  ordered component-prediction table.
- `summarize_intervention_effects.py`: performs one chronological-age fit per
  GEO dataset/correction stratum, collapses repeated samples to donor means,
  and uses a paired t-test for paired donor sets (including the shared-donor
  subset of partial pairs) or Welch's t-test for fully unpaired sets. The
  current workflow requires at least four donors per arm; comparisons without
  chronological age must be fully paired.
- `biometaclock_modeling.py`: canonical five-fold alpha selection, warning
  auditing, and the checked all-participant Elastic Net refit.
- `select_curated_intervention_results.py`: applies the audited representative
  selection, excludes the four datasets used by the current intervention
  analysis (`GSE276988`, `GSE330869`, `GSE168628`, `GSE130028`), and recomputes
  Benjamini-Hochberg FDR across that final family. The exclusion list can be
  overridden with repeated `--exclude-gse-id` arguments.
- `plot_intervention_fdr_volcano.py`: generates the current 110 × 60 mm PDF/SVG
  volcano with FDR 5% and 20% categories. It registers the bundled Liberation
  Sans assets, uses the formal 0.2 pt outline styling, and validates the exact
  five-label annotation map and offsets used by the current figure.
- `validate_intervention_volcano.py`: checks the canvas, editable text, axis
  labels, and the complete curated annotation set.

## Input layout

Every original dataset/feature-table input defaults to
`Path("/path/to/your/file")  # fill in your path`; replace it or pass a
command-line path before running. Workflow outputs remain under `results/`.

```text
datasets/intervention/
├── training_component_predictions.csv
├── biometaclock_features.csv
├── component_predictions.csv
├── sample_manifest.csv
├── intervention_selection_audit.csv
└── volcano_annotations.csv
```

The training table contains `clinical_phenoage` plus the component columns in
`biometaclock_features.csv`. The sample manifest requires `sample_id`,
`comparison_id`, `gse_id`, `comparison_name`, `group`, and `donor_id`; optional
`chronological_age` and `correction_stratum` columns control age adjustment.
The annotation table requires `comparison_id,short_label,dx,dy,ha,va`.
Within each `comparison_id`, `gse_id` and `comparison_name` must be constant.
If a sample occurs in more than one comparison, its `predicted_age`, optional
`chronological_age`, and `correction_stratum` must agree before de-duplication.
The selection audit's `selected` column accepts only explicit Boolean values
(`true` or `false`); ambiguous spreadsheet values are rejected.

## Example

```bash
uv run python -m intervention_analysis.train_intervention_biometaclock
uv run python -m intervention_analysis.predict_intervention_biological_age
uv run python -m intervention_analysis.summarize_intervention_effects
uv run python -m intervention_analysis.select_curated_intervention_results
uv run python -m intervention_analysis.plot_intervention_fdr_volcano
uv run python -m intervention_analysis.validate_intervention_volcano
```

Raw methylation parsing and component-clock inference are intentionally kept
upstream: this workflow starts from a validated component-prediction
matrix so that the statistical and plotting logic remains small and auditable.
Synthetic donor-design and audit-parser tests are in
`tests/test_intervention_effects.py` and can be run with the project environment.
