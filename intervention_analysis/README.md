# Intervention analysis

This directory contains the compact, reproducible workflow used to train the
biological MetaClock, predict intervention samples, estimate donor-level
effects, apply the curated one-intervention/one-comparison rule, and draw the
FDR volcano plot.

## Scripts

- `train_intervention_biometaclock.py`: tunes the complete
  `Pipeline(StandardScaler, ElasticNet)` with `GridSearchCV`, so scaling is fit
  independently inside each of five shuffled folds. It searches 30 alphas from
  `1` to `1e-5`, uses `l1_ratio=0.5`, rejects any convergence warning, and
  performs an all-participant selected-alpha refit. `dnamfitage` is rejected.
- `predict_intervention_biological_age.py`: applies the frozen model to an
  ordered component-prediction table.
- `summarize_intervention_effects.py`: performs one chronological-age fit per
  GEO dataset/correction stratum, collapses repeated samples to donor means,
  and uses a paired t-test only for fully paired donor sets or Welch's t-test
  only for fully unpaired sets. Partially overlapping designs fail rather than
  silently dropping unmatched donors; comparisons with no chronological age
  must be fully paired with at least two donors per group.
- `biometaclock_modeling.py`: leakage-safe fold-local scaling, five-fold alpha
  selection, convergence auditing, and the all-participant Elastic Net refit.
- `select_curated_intervention_results.py`: applies the audited representative
  selection and recomputes Benjamini-Hochberg FDR across the final family.
- `plot_intervention_fdr_volcano.py`: generates the current 110 × 60 mm PDF/SVG
  volcano with FDR 5% and 20% categories.
- `validate_intervention_volcano.py`: checks the canvas, editable text, axis
  labels, and the complete curated annotation set.

## Relative input layout

Replace the project-relative placeholder files below or pass corresponding
project-relative or absolute command-line paths. Each dataset/model input
default is marked `# fill in your path`.

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
uv run python intervention_analysis/train_intervention_biometaclock.py
uv run python intervention_analysis/predict_intervention_biological_age.py
uv run python intervention_analysis/summarize_intervention_effects.py
uv run python intervention_analysis/select_curated_intervention_results.py
uv run python intervention_analysis/plot_intervention_fdr_volcano.py
uv run python intervention_analysis/validate_intervention_volcano.py
```

Raw methylation parsing and component-clock inference are intentionally kept
upstream: this workflow starts from a validated component-prediction
matrix so that the statistical and plotting logic remains small and auditable.
Synthetic donor-design and audit-parser tests are in
`tests/test_intervention_effects.py` and can be run with the project environment.
