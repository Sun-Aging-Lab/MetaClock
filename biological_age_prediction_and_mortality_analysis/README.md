# Biological-age prediction and mortality analysis

The directory has five analysis entry points.  For each entry point, edit the
single line

```python
DATASET_PATH = Path(__file__).resolve().parent / "outputs" / "preprocessing" / "inchianti_analysis_ready.parquet"
```

and run the script from the project root.  Output tables, fitted artifacts, and
all PDF/SVG plots are written under
`biological_age_prediction_and_mortality_analysis/outputs/`.

## Prepare the shared InCHIANTI dataset

The raw inputs are three separate files: the DNAm RData containing
b.baseline and pheno.baseline, the baseline laboratory SAS table, and the
Vital Status SAS table. Set DNAM_RDATA_PATH, LAB_SAS_PATH, and VITAL_SAS_PATH
at the top of prepare_inchianti_dataset.py, then build the table once:

    python -m biological_age_prediction_and_mortality_analysis.prepare_inchianti_dataset

The producer computes Levine Clinical PhenoAge from the baseline laboratory
measurements, derives Cox and Kaplan-Meier fields from Vital Status, aligns
participant data by ID, aligns beta values by sampleID, and writes
outputs/preprocessing/inchianti_analysis_ready.parquet. The biological-age,
duplicate, Cox, and Kaplan-Meier entry points all read this file directly.
Component-clock predictions are not stored by the producer; each analysis
entry point calls PyAging when those columns are absent.

The table retains all 485,577 `b.baseline` rows in their original RData order:
482,421 `cg*`, 3,091 `ch.*`/`ch.X.*`, and 65 `rs*` features. The direct baseline
does not use a pre-training feature-selection filter; sparsity is introduced
only by Elastic Net fitting.

## 1. Biological-age prediction

Input: one sample per row with `ID`, `sampleID`, `age_years`,
`clinical_phenoage`, `clinical_phenoage_complete_case`, and the complete ordered
485,577-feature DNAm manifest. The script computes the 19-component OOF BioMeta
after excluding DNAmFitAge. One lowest-suffix DNAm sample per participant enters
training; every complete-case replicate is predicted by that participant's
held-out model, and MAE/Pearson are calculated after averaging predictions to
672 participants. The donor-to-fold mapping is pinned in
`inchianti_oof_fold_assignment.csv`, validated and mapped by donor ID
rather than regenerated from input row order. The external full refit separately
uses the 17 components available across all external datasets.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_biological_age_prediction
```

## 2. Duplicate-sample reliability

Run script 1 first. This entry point reuses its participant-grouped OOF
predictions, selects participants with exactly two complete-case DNAm samples,
and summarizes the 44 BioMeta and direct-baseline replicate pairs. It does not
fit a separate singleton-only model.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_duplicate_sample_reliability
```

## 3. External disease analysis

Input: one external DNAm dataset with `sampleID`, `age`, `disease_state`, and
CpG columns; optional `dataset` and `disease` columns control heatmap labels.
Run script 1 first.  This entry point loads its complete BioMeta and direct-CpG
baseline artifacts, calls the PyAging API for the 17 BioMeta components and
DNAmPhenoAge, performs within-dataset age correction and Welch tests, and draws
the three-model heatmap. The large baseline artifact is generated locally by
script 1.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_external_disease_analysis
```

## 4. Cox analysis

Input: the script-1 columns plus `survival_time_years`, `event`, and
`sex_male`.  The script selects one baseline sample per participant, assigns
the fixed Figure 4 donor folds, trains the 19-component BioMeta and the
full 485,577-feature direct baseline in five folds,
uses the exact Figure 4 panel-1 OOF ElasticNetCV configuration, and fits the
three pooled-OOF Cox specifications after z-scoring every included covariate
with population SD (`ddof=0`). Cox models use `CoxPHFitter(alpha=0.05,
baseline_estimation_method="breslow", penalizer=0.0)` with `robust=False` and
`show_progress=False`. Its donor folds are read from the same fixed Figure 4
assignment used by the biological-age panel.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_cox_analysis
```

## 5. Kaplan–Meier analysis

Input: the script-1 columns plus `INTV_MOR` (`0` for alive/censored; `4`, `5`,
or `6` for death at that follow-up wave).  The script selects one baseline
sample per participant, reads the fixed Figure 4 donor folds, trains the
19-component BioMeta and full 485,577-feature baseline, age-corrects pooled OOF scores, and
plots bottom/top-20% Kaplan–Meier curves and log-rank results.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_kaplan_meier_analysis
```

## CpG-weight enrichment

The enrichment workflow and fixed GREAT parameters are documented in
`enrichment/README.md`.  Its Python entry points are:

```bash
python -m biological_age_prediction_and_mortality_analysis.enrichment.select_top_bottom_cpg_foregrounds
python -m biological_age_prediction_and_mortality_analysis.enrichment.plot_top_bottom_enrichment
```

The two R entry points are:

```bash
Rscript biological_age_prediction_and_mortality_analysis/enrichment/build_27k_450k_epic_union_background.R results/enrichment/array_union
Rscript biological_age_prediction_and_mortality_analysis/enrichment/run_rgreat_top_bottom_enrichment.R chrono go_bp
```
