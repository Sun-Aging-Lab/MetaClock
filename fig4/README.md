# Figure 4: InCHIANTI and disease validation

This directory contains the analysis and plotting code for the Figure 4 panels,
followed by the exact compact result tables used for the manuscript figures.
Scripts are numbered in panel order and use paths relative to this directory.
No source-machine path is embedded in the code.

## Panel order

1. InCHIANTI prediction MAE/Pearson *r* barplot.
2. InCHIANTI duplicate-sample reliability barplot.
3. External disease-control heatmap.
4. Standardized Cox regression coefficients and 95% confidence intervals.
5. Kaplan-Meier curves and log-rank tests for the top/bottom 20% of each
   age-corrected score.

## Dataset layout

Place your data under `datasets/` using the filenames in
[`datasets/README.md`](datasets/README.md). If your files live elsewhere, pass
the documented command-line path options instead. In manuscript methods,
replace every placeholder with the location of your own dataset, for example:

```text
fig4/datasets/inchianti/oof_participant_predictions.csv
    # Replace with your dataset path.
```

## Run

From the project root containing `fig4/`:

```bash
python fig4/scripts/01_inchianti_barplots.py
python fig4/scripts/02_external_disease_heatmap.py
python fig4/scripts/03_cox_model.py
python fig4/scripts/04_kaplan_meier.py
```

Each script accepts `--help` and path overrides. Analysis tables and PDF/SVG
figures are written to `fig4/results/` by default. The checked-in tables are
the authoritative values underlying the manuscript panels; see
`results/manifest.tsv` for their order and purpose.

Checked-in row counts are 2 (prediction metrics), 2 (duplicate reliability),
18 (disease heatmap cells), 9 (Cox coefficients), 3 (KM log-rank tests), and
24 (KM curve points), in that order.

## Statistical settings

- Prediction: participant-level five-fold out-of-fold predictions; MAE from
  `sklearn.metrics.mean_absolute_error`; Pearson correlation from
  `scipy.stats.pearsonr`.
- Duplicate reliability: two repeated DNAm records per participant evaluated
  in the same held-out fold; absolute-agreement ICC(A,1), Pearson correlation,
  and absolute prediction differences.
- Disease: prediction is residualized on chronological age separately for each
  model with `sklearn.linear_model.LinearRegression`; disease versus control is
  compared with a two-sided Welch t-test (`equal_var=False`). Bonferroni
  correction is applied across the models tested within each dataset/disease.
- Cox: `lifelines` v0.30.3 `CoxPHFitter`, Breslow baseline, Efron tie handling, no
  penalization (`penalizer=0`), non-robust variance, 95% model-based intervals.
  Each covariate is standardized using the pooled mean and population SD
  (`ddof=0`).
- KM: scores are first residualized on age. Stable ranking selects the top and
  bottom 20%; `lifelines` v0.30.3 `KaplanMeierFitter(alpha=0.05)` supplies curves and
  intervals on the study follow-up-wave scale, and
  `lifelines.statistics.logrank_test` supplies the two-sided log-rank p-value.

Required Python packages: `numpy`, `pandas`, `matplotlib`, `scipy`,
`scikit-learn`, and `lifelines`.
