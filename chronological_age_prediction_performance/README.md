# Chronological-age prediction performance

This directory reproduces the chronological-age comparisons from bundled
sample-level component-clock predictions. It contains two runnable workflows:
an all-architecture/component comparison and a cumulative component-clock
ablation study. Both use `training_component_clock_predictions.csv` for model
fitting, use `test_component_clock_predictions.csv` for held-out evaluation,
and write figures to `results/chronological_age_prediction/`.

## Bundled inputs

The training CSV contains nine GEO cohorts and the test CSV contains four
held-out GEO cohorts. Every row is one sample with the following columns:

```text
dataset, sample_id, age,
horvath2013, hannum, dnamphenoage, altumage, han, encen40,
hrsinchphenoage, lin, skinandblood, yingcausage, zhangblup, zhangen,
pchorvath2013, pchannum, pcphenoage, pcskinandblood
```

The two tables contain only the 16 component clocks used by MetaClock;
`dnamfitage`, unrelated cohorts, and duplicate legacy clock columns from the
larger source table are excluded. `dataset, sample_id` is unique, and `age`
and every component-clock prediction are finite numeric values. The
architecture workflow uses the original
estimators (scikit-learn, PyTorch, and LightGBM), so the runtime environment
must include `lightgbm` in addition to the project's standard dependencies.

## Exported chronological MetaClock weights

The canonical 16-component chronological MetaClock is a
`StandardScaler` followed by a linear-kernel `SVR`, fitted on all 3,353 bundled
training samples from nine GEO cohorts. Its portable raw coefficients and
training metadata are stored in
`model_weights/metaclock_chronological_3353_samples_weights.csv` and
`model_weights/metaclock_chronological_3353_samples_metadata.json`.
Regenerate them with:

```bash
python -m chronological_age_prediction_performance.export_metaclock_weights
```

## Run all architecture, component, and border plots

Run:

```bash
python -m chronological_age_prediction_performance.train_test_architectures
```

This writes architecture and component MAE--Pearson scatter plots, plus the
dashed-boundary comparison plot, as PDF and editable-text SVG files.
It also writes the corresponding metric CSV files.

## Run all ablations

Run:

```bash
python -m chronological_age_prediction_performance.train_test_ablation
```

The script refits the linear-SVR MetaClock after cumulatively removing Zhang
BLUP, Han, and Skin&Blood, and writes the three resulting PDF/SVG scatter
plots and their source metric tables.
