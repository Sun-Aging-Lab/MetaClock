# Biological-age prediction and mortality analysis

## Figure 4 panels

Render the two InCHIANTI bar plots, the external disease heatmap and the separate
FHS and WHI Cox and Kaplan–Meier plots from the included group-level summaries.

```bash
python -m biological_age_prediction_and_mortality_analysis.plot_figure4
```

The heatmap compares the 19-component MetaClock-biological, the full-CpG
Baseline and DNAmPhenoAge. The clock predictions use zero-filled missing CpGs;
recorded sex is used when available and Biolearn SexEstimation supplies missing
sex for the GrimAge components. Each cell is the mean age-residualized score
in cases minus the mean in controls (years). Within each of the 14 eligible
datasets, the three two-sided Welch p-values are Bonferroni-adjusted. The seven
datasets with any adjusted p < 0.05 are shown. GSE146917 (Huntington’s disease)
was excluded because its methylation records pool multiple participants.

The Cox plots report the log hazard ratio and 95% CI per SD of predicted score.
The FHS model uses five clocks trained outside FHS and adjusts for age, sex and
DNAm set. WHI uses the 19-component MetaClock, with adjustment
for age and race. Outputs are in `outputs/biological_age_prediction/`,
`outputs/duplicate_sample_reliability/`, `outputs/figure4_external_disease/`,
`outputs/cox/` and `outputs/kaplan_meier/`.

## Prepare the shared InCHIANTI dataset

The prediction and duplicate-sample analyses use the DNAm RData containing
`b.baseline` and `pheno.baseline`, and the baseline laboratory SAS table.
Set both input paths in `prepare_inchianti_dataset.py`.

```python
DNAM_RDATA_PATH = Path("path/to/your/file")  # DNAm RData
LAB_SAS_PATH = Path("path/to/your/file")  # baseline laboratory SAS table
```

These inputs may be outside the repository. Build the shared table once:

    python -m biological_age_prediction_and_mortality_analysis.prepare_inchianti_dataset

## 1. Biological-age prediction

Input: one sample per row with `ID`, `sampleID`, `age_years`,
`clinical_phenoage`, `clinical_phenoage_complete_case`, and DNAm features. The primary-profile donor-to-fold mapping is pinned in
`inchianti_primary_oof_folds.csv`. Participants are sorted by numeric ID and
sample suffix; the lowest-suffix sample is used for each participant. The
19-component MetaClock and full-CpG baseline share five outer folds stratified
by age decade and sex. Both use shuffled five-fold inner CV with seed 0,
zero imputation and scaling inside each training fold, 30 alphas from 1 to
1e-5, and cyclic elastic net with l1_ratio=0.5, max_iter=20000 and tol=1e-4.
The baseline uses all 485,577 DNAm features without feature selection.
The workflow saves the OOF predictions and configuration manifest.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_biological_age_prediction
```

## 2. Duplicate-sample reliability

Run script 1 first. It also trains on 628 participants without duplicate
profiles and predicts all 88 profiles from the 44 held-out participants with
exactly two complete-case samples. This entry point uses those independent
holdout predictions for duplicate-sample reliability.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_duplicate_sample_reliability
```

## 3. External disease analysis with the published 17-component clock

Input: one external DNAm dataset with `sampleID`, `age`, `disease_state`, and
CpG columns; optional `dataset` and `disease` columns control heatmap labels.
Replace the placeholder in `run_external_disease_analysis.py` with the local
CSV, CSV.GZ, or Parquet file:

```python
DATASET_PATH = Path("path/to/your/file")
```


```bash
python -m biological_age_prediction_and_mortality_analysis.run_external_disease_analysis
```

## 4. Mortality in FHS and WHI

The two mortality entry points render Figures 4d–g from the included cohort
summaries. FHS follow-up starts at examination 8 (2,606 participants, 929
deaths). WHI follow-up starts at enrollment (2,127 participants, 894 deaths).
Each panel compares MetaClock-biological (5 components in FHS and 19 in WHI),
the full-CpG baseline, and DNAmPhenoAge.

The Cox source table contains the unpenalized lifelines estimates and Wald
95% confidence intervals per within-cohort SD. The Kaplan–Meier source tables
contain survival probabilities, 95% confidence intervals, death counts, and
two-sided log-rank tests. The curves compare the highest and lowest quintiles
of each score after regression on age, sex, and DNAm laboratory in FHS or
age and race in WHI. Each tail contains 521 FHS or 425 WHI participants.
ΔDeaths is the difference in death counts over full follow-up (highest minus
lowest quintile).

```bash
python -m biological_age_prediction_and_mortality_analysis.run_cox_analysis
python -m biological_age_prediction_and_mortality_analysis.run_kaplan_meier_analysis
```

## CpG-weight enrichment

The [GO BP workflow](enrichment/README.md) includes commands for the published
top/bottom-1,000 foregrounds, the 8- and 17-component Elastic Net fits, and
the region-level enrichment tables and figures.
