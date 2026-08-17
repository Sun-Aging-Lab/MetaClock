# Biological-age weights, external disease, and enrichment analysis

This directory publishes the unrestricted analysis needed to apply BioMetaClock,
analyze external disease cohorts, render the external-disease heatmap, and run
CpG-weight enrichment. InCHIANTI is restricted: participant-level inputs,
training/OOF code, sample identifiers, performance/reliability barplots,
mortality analysis, and Cox/Kaplan–Meier plotting code are intentionally absent.

## Published BioMetaClock weights

`model_weights/biometaclock_672_participants_weights.csv` contains the fitted
17-component BioMetaClock trained on one sample from each of 672 participants.
It records feature order, scaler mean and scale, standardized coefficients, and
equivalent raw-scale coefficients. The accompanying JSON records both
intercepts, selected alpha, convergence count, estimator definition, software
versions, and the SHA-256 digest of the restricted fitted artifact. It contains
no participant data or internal filesystem path.

The published raw-scale form is

```text
prediction = raw_intercept + component_predictions @ raw_coefficient
```

Apply it to a CSV containing the exact 17 component-clock columns:

```bash
python -m biological_age_prediction_and_mortality_analysis.apply_biometaclock_weights \
  --components /path/to/your/file
```

The ordered components are `dnamphenoage`, `hrsinchphenoage`, `dnamic`,
`zhangmortality`, `systemsage`, `systemsageblood`, `systemsagebrain`,
`systemsageheart`, `systemsagehormone`, `systemsageimmune`,
`systemsageinflammation`, `systemsagekidney`, `systemsageliver`,
`systemsagelung`, `systemsagemetabolic`, `systemsagemusculoskeletal`, and
`dunedinpace`. `dnamfitage` is globally excluded.

## Unrestricted result workflows

- `analyze_external_disease_differences.py` performs within-dataset age
  correction and maintained SciPy Welch tests on an analysis-ready prediction
  table.
- `plot_external_disease_panel.py` renders the external disease heatmap.
- `render_external_disease_panel.py` is the external-disease rendering entry point.
- `validate_external_disease_outputs.py` validates its schema,
  counts, vector dimensions, fonts, and global component exclusion.

External inputs use `Path("/path/to/your/file")  # fill in your path` where they
enter the workflow. Generated intermediate and output paths are project-relative.
There is no mortality source-table or participant-level interface.

Typical rendering and validation commands are:

```bash
python -m biological_age_prediction_and_mortality_analysis.analyze_external_disease_differences
python -m biological_age_prediction_and_mortality_analysis.render_external_disease_panel
python -m biological_age_prediction_and_mortality_analysis.validate_external_disease_outputs
```

## Enrichment

The enrichment subdirectory contains the chronological and biological
top/bottom-1,000 GREAT workflow. The publication table
`enrichment/top_bottom_1000_cpg_weights.csv` has 4,000 rows with columns
`group`, `direction`, `rank`, `cpg`, and `weight`; every group/direction has
exactly 1,000 deterministic entries. See `enrichment/README.md` for the union
background, ranking definition, rGREAT parameters, and commands.
