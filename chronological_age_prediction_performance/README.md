# Chronological-age prediction performance

This directory reproduces the chronological-age comparisons from a
wide table of component-clock predictions.  It intentionally contains only
two runnable workflows: an all-architecture/component comparison and a
cumulative component-clock ablation study.  Both train on the supplied
training table, test on the supplied test table, and write all figures to the
fixed relative directory `results/chronological_age_prediction/`.

## Input schema

Supply one training file and one test file, each as CSV (or Parquet).  Every
row is one sample and must contain the following columns:

```text
dataset, sample_id, age,
horvath2013, hannum, dnamphenoage, altumage, han, encen40,
hrsinchphenoage, lin, skinandblood, yingcausage, zhangblup, zhangen,
pchorvath2013, pchannum, pcphenoage, pcskinandblood
```

`dataset` labels the held-out panel in the test file; `dataset, sample_id`
must be unique; `age` and every component-clock prediction must be finite
numeric values.  The architecture workflow uses the original
estimators (scikit-learn, PyTorch, and LightGBM), so the runtime environment
must include `lightgbm` in addition to the project's standard dependencies.

## Run all architecture, component, and border plots

Edit the two defaults at the top of
`train_test_architectures.py` to your two paths, then run:

```bash
uv run python -m chronological_age_prediction_performance.train_test_architectures
```

Alternatively, pass only the two path arguments:

```bash
uv run python -m chronological_age_prediction_performance.train_test_architectures \
  --training-dataset /path/to/training.csv \
  --test-dataset /path/to/test.csv
```

This writes architecture and component MAE--Pearson scatter plots, plus the
formal dashed-boundary comparison plot, as PDF and editable-text SVG files.
It also writes the corresponding metric CSV files.

## Run all ablations

Edit the same two defaults at the top of `train_test_ablation.py`, then run:

```bash
uv run python -m chronological_age_prediction_performance.train_test_ablation
```

Or provide only the two path arguments:

```bash
uv run python -m chronological_age_prediction_performance.train_test_ablation \
  --training-dataset /path/to/training.csv \
  --test-dataset /path/to/test.csv
```

The script refits the linear-SVR MetaClock after cumulatively removing Zhang
BLUP, Han, and Skin&Blood, and writes the three resulting PDF/SVG scatter
plots and their source metric tables.
