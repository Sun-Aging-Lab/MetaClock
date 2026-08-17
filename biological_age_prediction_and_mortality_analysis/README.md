# Biological-age prediction and mortality analysis

The directory has five analysis entry points.  For each entry point, edit the
single line

```python
DATASET_PATH = Path("/path/to/your/file")  # fill in your path
```

and run the script from the project root.  Output tables, fitted artifacts, and
all PDF/SVG plots are written under
`biological_age_prediction_and_mortality_analysis/outputs/`.

## 1. Biological-age prediction

Input: one sample per row with `ID`, `sampleID`, `age_years`,
`clinical_phenoage`, `clinical_phenoage_complete_case`, and CpG beta columns
named `cg...`.  The 17 canonical component columns may be supplied; otherwise
the script computes them with PyAging.  It applies the complete-case flag,
keeps every sample from a participant in the same outer fold, runs five-fold
OOF BioMeta/direct-CpG training, draws MAE/Pearson bars, and refits both
artifacts on one baseline sample per participant for external use.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_biological_age_prediction
```

## 2. Duplicate-sample reliability

Input: the same row-oriented InCHIANTI dataset.  Every participant with exactly
two DNAm samples is held out in full.  The direct-CpG baseline and BioMeta are
trained on all valid participants with exactly one DNAm sample, while
DNAmPhenoAge is obtained by direct PyAging inference.  All three comparators are
then summarized as paired prediction differences.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_duplicate_sample_reliability
```

## 3. External disease analysis

Input: one external DNAm dataset with `sampleID`, `age`, `disease_state`, and
CpG columns; optional `dataset` and `disease` columns control heatmap labels.
Run script 1 first.  This entry point loads its complete BioMeta and direct-CpG
baseline artifacts, calls the PyAging API for the 17 BioMeta components and
DNAmPhenoAge, performs within-dataset age correction and Welch tests, and draws
the three-model heatmap.  The large baseline artifact is generated locally and
is not replaced by partial or invented weights in this repository.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_external_disease_analysis
```

## 4. Cox analysis

Input: the script-1 columns plus `survival_time_years`, `event`, and
`sex_male`.  The script selects one baseline sample per participant, assigns
participant-level age-stratified folds, trains both predictors in five folds,
fits the three pooled-OOF Cox specifications on raw covariates with the formal
`1e-6` penalizer, converts prediction coefficients to population-SD units for
display, and draws the forest plot.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_cox_analysis
```

## 5. Kaplan–Meier analysis

Input: the script-1 columns plus `INTV_MOR` (`0` for alive/censored; `4`, `5`,
or `6` for death at that follow-up wave).  The script selects one baseline
sample per participant, assigns 10-year-age-bin folds, trains both predictors,
age-corrects pooled OOF scores, and plots bottom/top-20% Kaplan–Meier curves and
log-rank results.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_kaplan_meier_analysis
```

## CpG-weight enrichment

The enrichment workflow and fixed GREAT parameters are documented in
`enrichment/README.md`.  Its Python entry points are:

```bash
python -m biological_age_prediction_and_mortality_analysis.enrichment.build_component_weight_rankings chrono
python -m biological_age_prediction_and_mortality_analysis.enrichment.build_component_weight_rankings bio
python -m biological_age_prediction_and_mortality_analysis.enrichment.select_top_bottom_cpg_foregrounds
python -m biological_age_prediction_and_mortality_analysis.enrichment.plot_top_bottom_enrichment
```

The two R entry points are:

```bash
Rscript biological_age_prediction_and_mortality_analysis/enrichment/build_27k_450k_epic_union_background.R results/enrichment/array_union
Rscript biological_age_prediction_and_mortality_analysis/enrichment/run_rgreat_top_bottom_enrichment.R chrono go_bp
```
