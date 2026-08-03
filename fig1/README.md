# Fig. 1: chronological-age performance and ablation

This directory is self-contained. It covers, in panel order, the model-
architecture scatterplot, component-clock scatterplot, and cumulative ablation
study. Large sample-level datasets and fitted models are deliberately excluded.

## Dataset setup

Place your combined prediction table at:

```text
fig1/datasets/all_predicted_ages_with_id.csv
```

This is a placeholder path. Replace its contents or pass another
project-relative path such as `--predictions fig1/datasets/predictions.csv` to
`run_ablation.py`. The table must
contain `dataset`, `age`, and all component-clock columns listed in
`scripts/run_ablation.py`. One row represents one sample. An `id` column is
recommended for traceability but is not used in the calculation.

The configured training cohorts are GSE40279, GSE51057, GSE42861, GSE41169,
GSE51032, GSE73103, GSE69270, GSE64495, and GSE52588. The external test cohorts
are GSE147221, GSE157131, GSE234461, and GSE72680.

## Scripts

- `scripts/plot_scatterplots.py` validates `results/01...` and `results/02...`
  and regenerates both scatterplots as PDF and SVG.
- `scripts/run_ablation.py` ranks eligible component clocks by mean MAE over
  the four external cohorts, cumulatively removes the top five, and retrains
  MetaClock after each removal.

Key ablation parameters are explicit in the code: complete-case analysis,
`StandardScaler`, and `sklearn.svm.SVR(kernel="linear", C=1.0, epsilon=0.1)`.
The historical clock `yingcausage` remains an input feature but is excluded
from the removal ranking. Performance is reported as MAE in years and Pearson r.

## Ordered result tables

1. `01_model_architecture_scatter.csv` — nine architecture/ensemble points.
2. `02_component_clock_scatter.csv` — 15 component clocks plus MetaClock.
3. `03_ablation_summary.csv` — one summary row per cumulative removal.
4. `04_ablation_step_01.csv` through `08_ablation_step_05.csv` — detailed
   per-dataset results used to audit every refitted model.
5. `09_single_clock_ranking.csv` — ranking that determines removal order.

`results/manifest.tsv` gives the machine-readable panel order. Table sizes and
required columns are: 01 (36 rows; `dataset,model,n,mae,pearson_r`), 02 (64
rows; `dataset,predictor,n,mae,pearson_r`), 03 (5 rows; ablation summary),
04--08 (64, 60, 56, 52, and 48 rows; per-dataset ablation metrics), and 09 (15
rows; ranking). The 01/02 tables preserve the validated final scatterplot
values; 03--09 preserve the validated cumulative linear-SVR ablation run.

## Usage

From the new project root:

```bash
python fig1/scripts/plot_scatterplots.py
python fig1/scripts/run_ablation.py \
  --predictions fig1/datasets/all_predicted_ages_with_id.csv
```

Install `numpy`, `pandas`, `matplotlib`, and `scikit-learn` in the execution
environment. Plot outputs are written to `fig1/figures/` by default.
