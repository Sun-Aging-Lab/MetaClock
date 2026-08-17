# Chronological-age prediction performance

This directory contains the compact, reproducible workflow for:

- model-architecture MAE--Pearson-r scatter;
- component-clock MAE--Pearson-r scatter;
- three cumulative component-clock ablations (Zhang BLUP, Han, then Skin&Blood);
- the component-clock versus supervised MetaClock-architecture comparison.

## Input layout

The scripts deliberately do not embed a cluster-specific input location. Fill
each `Path("/path/to/your/file")  # fill in your path` default at the top of
the relevant script, or pass the input explicitly on the command line. The
expected source files are:

```text
your input directory/
├── heldout_predictions_long.csv
├── model_architecture_metrics.csv
├── component_clock_metrics.csv
├── architecture_comparison_metrics.csv
└── component_clock_ablation/
    ├── step_01_source_data.csv
    ├── step_02_source_data.csv
    └── step_03_source_data.csv
```

`heldout_predictions_long.csv` has `dataset`, `sample_id`, `age`, `predictor`,
and `prediction`. The metric tables require `dataset`, their label column
(`model` or `predictor`), `mae`, and `pearson_r`. The ablation files also
require `ablation_step` and cumulative semicolon-separated `removed_clocks`.
The architecture comparison reads a semantic source table with `dataset`,
`group`, `label`, `n`, `mae`, and `pearson_r`; `group` is either `Single clock`
or `MetaClock architecture`.

The four held-out datasets are `GSE147221`, `GSE157131`, `GSE234461`, and
`GSE72680`. All age-prediction statistics are standard `scikit-learn` MAE and
two-sided `scipy.stats.pearsonr` calculations.

## Commands

Run these commands from the repository root after filling in the input paths,
or provide alternative project-relative or absolute `--...` paths:

```bash
uv run python -m chronological_age_prediction_performance.summarize_heldout_age_prediction_metrics --label-column predictor
uv run python -m chronological_age_prediction_performance.summarize_heldout_age_prediction_metrics --label-column model
uv run python -m chronological_age_prediction_performance.plot_chronological_age_prediction_scatter
uv run python -m chronological_age_prediction_performance.plot_component_clock_ablation_scatter
uv run python -m chronological_age_prediction_performance.plot_metaclock_architecture_comparison_scatter \
  --metrics /path/to/your/file
uv run python -m chronological_age_prediction_performance.validate_chronological_age_prediction_outputs
```

Outputs are PDF and editable-text SVG files under
`results/chronological_age_prediction/`. The vector plots use a 180 x 65 mm
print canvas and retain the formal output stems
`pearson_mae_model_architecture`, `pearson_mae_submodel`,
`ablation_step_01_without_zhangblup`, `ablation_step_02_without_han`,
`ablation_step_03_without_skinandblood`, and
`pearson_mae_submodel_architectures`. The `--label-column` argument preserves
either `predictor` or `model`, so the metric summarizer can create both
primary input tables. The architecture-comparison command writes its validated
source table beside the vector output. This workflow consumes held-out
predictions/metrics; model fitting remains upstream because the scatter plots
should never retrain on the plotted test datasets.

The formal color maps are defined once in `plotting.py`: MetaClock and SVM use
the reserved dark-blue `tab20` color, while component clocks intentionally skip
that color. The validator requires one 180 x 65 mm PDF and one editable-text SVG
for every expected plot, and checks that the PDF embeds Liberation Sans. No
cluster-specific path is embedded in the workflow.
