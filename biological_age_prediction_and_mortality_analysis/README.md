# Biological-age prediction and mortality analysis

This directory contains the portable biological-age and mortality workflow: InCHIANTI out-of-fold
prediction, duplicate-sample reliability, all-participant external models,
external disease differences, mortality analyses, plotting, validation, and the
CpG-weight enrichment subworkflow. Input defaults are project-relative; replace
paths marked `# fill in your path` or pass their command-line options.

## Reproducibility contract

`analysis_manifest.json` is the single source of truth. BioMeta has exactly 17
ordered component predictions:

`dnamphenoage`, `hrsinchphenoage`, `dnamic`, `zhangmortality`, `systemsage`,
`systemsageblood`, `systemsagebrain`, `systemsageheart`, `systemsagehormone`,
`systemsageimmune`, `systemsageinflammation`, `systemsagekidney`,
`systemsageliver`, `systemsagelung`, `systemsagemetabolic`,
`systemsagemusculoskeletal`, and `dunedinpace`.

`dnamfitage` is globally excluded. The direct baseline requires an exact,
unique 485,577-CpG feature manifest. The external artifacts are refit on one
sample from each of 672 participants. `training_summary.json` binds normalized
inputs, exact feature manifests, fitted artifacts, software versions, and
SHA-256 digests; validators reject missing or inconsistent evidence.

Elastic Net alpha selection uses scikit-learn's maintained `GridSearchCV` over
the complete `Pipeline(StandardScaler(), ElasticNet())`. Thus the scaler is fit
inside each CV fold, not once before CV. It uses five shuffled folds (seed 0),
30 log-spaced alphas from 1 through 1e-5, `l1_ratio=0.5`, negative MSE, and
`max_iter=100000`. Any `ConvergenceWarning`, or a selected fit reaching
`max_iter`, fails the run. Each alpha's mean/SD CV error and rank is retained.
Previously generated artifacts with a convergence warning do not satisfy this
contract and must be regenerated.

## Portable input boundary

The original cohort-specific RData/SAS/GEO extraction, methylation QC, probe
harmonization, and component-clock inference require source-controlled data not
shipped here. This workflow deliberately does not guess those transformations.
Its executable boundary is normalized, analysis-ready files with strict schema,
counts, exact alignment, feature order, and provenance:

- `participant_targets.csv`: `ID`, `sampleID`, `age_years`,
  `clinical_phenoage`; exactly 672 unique participant/sample rows.
- `component_clock_predictions.csv`: `sampleID` followed by the exact 17
  manifest columns.
- `cpg_matrix.parquet`: `sampleID` followed by exactly 485,577 unique CpGs.
- duplicate inputs: labels identify exactly 44 IDs with two samples each, plus
  aligned component and CpG matrices.
- external metadata: `dataset`, `disease`, `sample_id`, `age`, `group`; exactly
  3,325 samples, 15 dataset/disease comparisons, and `control`/`disease` groups.
- external component/CpG matrices are aligned by `sample_id`; DNAmPhenoAge is a
  separate normalized prediction input.
- survival covariates: `ID`, `sampleID`, `age`, `sex`, `survival_time`, `event`,
  joined to the generated OOF predictions and clinical PhenoAge.

The normalized boundary, generated artifacts, and their hashes are auditable;
claims about unavailable raw-data preprocessing are intentionally out of scope.

## Script and output map

| Script | Main output |
|---|---|
| `generate_inchianti_oof_predictions.py` | nested five-fold OOF predictions, fold assignments, selected alphas, fold models, run audit |
| `evaluate_biological_age_prediction.py` | MAE/Pearson source data |
| `generate_duplicate_holdout_predictions.py` | leave-all-replicates-out paired predictions and audit |
| `analyze_duplicate_sample_reliability.py` | paired absolute-difference source data (not ICC) |
| `train_external_biological_age_models.py` | clean all-672 BioMeta/baseline pipelines, exact features, alpha searches, hashes |
| `predict_external_disease_models.py` | aligned three-model external predictions and audit |
| `analyze_external_disease_differences.py` | within-dataset/model age correction, Welch tests, three-model Bonferroni |
| `prepare_mortality_analysis_table.py` | aligned 672-person OOF/clinical/survival table and audit |
| `analyze_mortality_associations.py` | standardized Cox and formal KM/log-rank source tables |
| `render_biological_age_mortality_panels.py` | five semantic PDF/SVG pairs |
| `validate_biological_age_mortality_outputs.py` | artifact, source-schema, count, font, exclusion, and vector report |

The plot stems are `bioage_prediction_mae_pearson`,
`duplicate_sample_reliability`, `external_disease_heatmap`,
`cox_coefficients_standardized`, and `kaplan_meier_survival`. Plotting is split
across small semantic modules and shares typography/vector helpers in
`plotting.py`.

Mortality uses lifelines `CoxPHFitter` with Breslow baseline and no penalizer.
Standardization is computed on each model's complete-case fit sample. KM groups
are the deterministic lowest/highest `floor(n * 0.20)` age-corrected scores;
source tables retain confidence intervals, risk/event counts, and expected-event
log-rank audit terms.

## Typical execution

From the project root (module execution uses the regular package imports):

```bash
python -m biological_age_prediction_and_mortality_analysis.generate_inchianti_oof_predictions --n-jobs -1
python -m biological_age_prediction_and_mortality_analysis.evaluate_biological_age_prediction
python -m biological_age_prediction_and_mortality_analysis.generate_duplicate_holdout_predictions --n-jobs -1
python -m biological_age_prediction_and_mortality_analysis.analyze_duplicate_sample_reliability
python -m biological_age_prediction_and_mortality_analysis.train_external_biological_age_models --n-jobs -1
python -m biological_age_prediction_and_mortality_analysis.predict_external_disease_models
python -m biological_age_prediction_and_mortality_analysis.analyze_external_disease_differences
python -m biological_age_prediction_and_mortality_analysis.prepare_mortality_analysis_table
python -m biological_age_prediction_and_mortality_analysis.analyze_mortality_associations
python -m biological_age_prediction_and_mortality_analysis.render_biological_age_mortality_panels
python -m biological_age_prediction_and_mortality_analysis.validate_biological_age_mortality_outputs
python -m unittest biological_age_prediction_and_mortality_analysis.tests.test_synthetic_workflow -v
```

scikit-learn, SciPy, lifelines, pandas, and NumPy provide the maintained
statistical implementations. Custom code is limited to orchestration, input
adaptation, validation, plotting, and provenance. Liberation Sans is required
for formal vector output. See `enrichment/README.md` for the rGREAT workflow.
