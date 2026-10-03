# Biological-age prediction and mortality analysis


## Prepare the shared InCHIANTI dataset

The raw inputs are three separate files: the DNAm RData containing
b.baseline and pheno.baseline, the baseline laboratory SAS table, and the
Vital Status SAS table. Replace each placeholder at the top of
`prepare_inchianti_dataset.py` with the corresponding local file path:

```python
DNAM_RDATA_PATH = Path("path/to/your/file")  # DNAm RData
LAB_SAS_PATH = Path("path/to/your/file")  # baseline laboratory SAS table
VITAL_SAS_PATH = Path("path/to/your/file")  # Vital Status SAS table
```

These inputs may be outside the repository. Build the shared table once:

    python -m biological_age_prediction_and_mortality_analysis.prepare_inchianti_dataset

## 1. Biological-age prediction

Input: one sample per row with `ID`, `sampleID`, `age_years`,
`clinical_phenoage`, `clinical_phenoage_complete_case`, and the complete ordered
485,577-feature DNAm manifest. The donor-to-fold mapping is pinned in
`inchianti_oof_fold_assignment.csv`, validated and mapped by donor ID
rather than regenerated from input row order.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_biological_age_prediction
```

## 2. Duplicate-sample reliability

Run script 1 first. This entry point reuses its participant-grouped OOF
predictions, selects participants with exactly two complete-case DNAm samples,
and summarizes the 44 BioMeta and direct-baseline replicate pairs.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_duplicate_sample_reliability
```

## 3. External disease analysis

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

## 4. Cox analysis

Input: the script-1 columns plus `survival_time_years`, `event`, and
`sex_male`.  The script selects one baseline sample per participant, assigns
the fixed Figure donor folds, trains the 19-component BioMeta and the
full 485,577-feature direct baseline in five folds,
uses Figure 4 panel-1 OOF ElasticNetCV configuration, and fits the
three pooled-OOF Cox specifications after z-scoring every included covariate.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_cox_analysis
```

## 5. Kaplan–Meier analysis

Input: the script-1 columns plus `INTV_MOR` (`0` for alive/censored; `4`, `5`,
or `6` for death at that follow-up wave).

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

The two R entry points are (replace `path/to/your/file` with the UCSC
hg18-to-hg19 chain file; see the enrichment README for external GMT inputs):

```bash
Rscript biological_age_prediction_and_mortality_analysis/enrichment/build_27k_450k_epic_union_background.R results/enrichment/array_union path/to/your/file
Rscript biological_age_prediction_and_mortality_analysis/enrichment/run_rgreat_top_bottom_enrichment.R chrono go_bp
```
