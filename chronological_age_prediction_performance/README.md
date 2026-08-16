# Chronological-age prediction performance

This directory contains the compact, reproducible workflow for:

- model-architecture MAE--Pearson-r scatter;
- component-clock MAE--Pearson-r scatter;
- three cumulative component-clock ablations (Zhang BLUP, Han, then Skin&Blood);
- the component-clock versus supervised MetaClock-architecture comparison.

## Input layout

Place inputs under a project-relative `data/chronological_age_prediction/`
directory (or pass paths
explicitly on the command line):

```text
data/chronological_age_prediction/
├── heldout_predictions_long.csv
├── model_architecture_metrics.csv
├── component_clock_metrics.csv
├── component_and_architecture_metrics.csv
└── component_clock_ablation/
    ├── step_01_source_data.csv
    ├── step_02_source_data.csv
    └── step_03_source_data.csv
```

`heldout_predictions_long.csv` has `dataset`, `sample_id`, `age`, `predictor`,
and `prediction`. The metric tables require `dataset`, their label column
(`model`, `predictor`, or `label`), `mae`, and `pearson_r`. The ablation files
also require `ablation_step` and cumulative semicolon-separated
`removed_clocks`. The architecture-comparison file uses `group` values `Single
clock` and `MetaClock architecture`.

The four held-out datasets are `GSE147221`, `GSE157131`, `GSE234461`, and
`GSE72680`. All age-prediction statistics are standard `scikit-learn` MAE and
two-sided `scipy.stats.pearsonr` calculations.

## Commands

Run these commands from the repository root after filling in the relative data
paths at the tops of scripts, or provide alternative project-relative or
absolute `--...` paths:

```bash
uv run python chronological_age_prediction_performance/summarize_heldout_age_prediction_metrics.py --label-column predictor
uv run python chronological_age_prediction_performance/summarize_heldout_age_prediction_metrics.py --label-column model
uv run python chronological_age_prediction_performance/plot_chronological_age_prediction_scatter.py
uv run python chronological_age_prediction_performance/plot_component_clock_ablation_scatter.py
uv run python chronological_age_prediction_performance/plot_metaclock_architecture_comparison_scatter.py
uv run python chronological_age_prediction_performance/validate_chronological_age_prediction_outputs.py
```

Outputs are PDF and editable-text SVG files under
`results/chronological_age_prediction/`. The vector plots use a 180 x 65 mm
print canvas. The `--label-column` argument preserves either `predictor` or
`model`, so the metric summarizer can create both primary input tables. The
comparison table combines the already summarized component and architecture
metrics with its explicit `group` column. This workflow consumes held-out
predictions/metrics; model fitting remains upstream because the scatter plots
should never retrain on the plotted test datasets.

The formal color maps are defined once in `plotting.py`: MetaClock and SVM use
the reserved dark-blue `tab20` color, while component clocks intentionally skip
that color. The validator requires one 180 x 65 mm PDF and one editable-text SVG
for every expected plot, and checks that the PDF embeds Liberation Sans. No
cluster-specific path is embedded in the workflow.
