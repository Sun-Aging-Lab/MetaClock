# Biological-age prediction and mortality analysis

## Figure 4 panels

The default command renders the two InCHIANTI bar plots, the external disease
heatmap, and the FHS and WHI Cox plots from the group summaries. It uses the
bundled Kaplan–Meier panels for Figure 4f–g.

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
Supply both local files to build the shared analysis table. Replace each example
path with the corresponding file on your machine.

```bash
python -m biological_age_prediction_and_mortality_analysis.prepare_inchianti_dataset \
  --dnam-rdata path/to/your/methylation.RData \
  --lab-sas path/to/your/baseline_labs.sas7bdat
```

The two downstream commands use the generated table. Existing configurations
can still set `DNAM_RDATA_PATH` and `LAB_SAS_PATH` in the preparation script.

## 1. Biological-age prediction

Input: one sample per row with `ID`, `sampleID`, `age_years`,
`clinical_phenoage`, `clinical_phenoage_complete_case`, and DNAm features. For each participant the workflow selects the
complete-case sample with the lowest numeric sample-ID suffix. It sorts primary
samples by numeric participant ID, then reconstructs the original five outer
folds from age decade and sex with scikit-learn’s shuffled StratifiedKFold
(seed 0). A checksum verifies the full assignment. The 19-component MetaClock
and the full-CpG baseline share these folds. Both use shuffled five-fold inner CV with seed 0,
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

Figures 4d–g report external cohort mortality. FHS follow-up starts at examination 8 (2,606 participants, 929
deaths). WHI follow-up starts at enrollment (2,127 participants, 894 deaths).
Each panel compares MetaClock-biological (5 components in FHS and 19 in WHI),
the full-CpG baseline, and DNAmPhenoAge.

The Cox source table contains the unpenalized lifelines estimates and Wald
95% confidence intervals per within-cohort SD. The Kaplan–Meier summary tables
contain group sizes, death counts, and two-sided log-rank tests. The published panels show
the survival curves and 95% confidence bands. The curves compare the highest and lowest quintiles
of each score after regression on age, sex, and DNAm laboratory in FHS or
age and race in WHI. Each tail contains 521 FHS or 425 WHI participants.
ΔDeaths is the difference in death counts over full follow-up (highest minus
lowest quintile).

```bash
python -m biological_age_prediction_and_mortality_analysis.run_cox_analysis
python -m biological_age_prediction_and_mortality_analysis.run_kaplan_meier_analysis
```

To reproduce Figure 4f–g from authorized participant follow-up and clock
scores, supply the directory containing the four local CSVs.

```bash
python -m biological_age_prediction_and_mortality_analysis.run_kaplan_meier_analysis \
  --private-cohort-dir path/to/your/mortality_data
```

| File under the supplied directory | Required columns |
| --- | --- |
| `inputs/fhs_pheno.csv` | `sample_id`, `time_years`, `event`, `age`, `female`, `batch_umn` |
| `inputs/whi_pheno.csv` | `sample_id`, `time_years`, `event`, `age`, `race_black`, `race_hispanic` |
| `predictions/fhs_predictions.csv` | `sample_id`, `metaclock_fhs5`, `baseline`, `dnamphenoage` |
| `predictions/whi_predictions.csv` | `sample_id`, `metaclock_full19`, `baseline`, `dnamphenoage` |

Each cohort has one row per sample in each file, with matching IDs. Follow-up
is in years. The death indicator is 1 for a death and 0 for censoring. The
command fits the adjusted quintile comparisons, verifies group counts and
log-rank results against the manuscript, and writes the two panels as PDF
and SVG. Detailed event times and curve estimates stay in memory.

If you already have the two manuscript curve tables, you can redraw the same
panels with `--private-curve-dir path/to/your/curve_tables`. That directory
supplies `fhs_km_curve_points.csv` and `whi_km_curve_points.csv`. This mode
uses the published log-rank and death-count annotations. The main Figure 4
command accepts either local input option. With neither option it uses the
bundled Kaplan–Meier panels.

## CpG-weight enrichment

The [GO BP workflow](enrichment/README.md) includes commands for the published
top/bottom-1,000 foregrounds, the 8- and 17-component Elastic Net fits, and
the region-level enrichment tables and figures.
