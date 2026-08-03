# MetaClock figure analysis

This directory contains readable, figure-ordered analysis code and compact
result tables for the MetaClock manuscript. Raw methylation matrices, fitted
models, and other large inputs are intentionally not included.

## Layout

```text
fig1/  chronological-age scatterplots and ablation study
fig2/  array, tissue, and species generalization
fig3/  evolutionary distance, maximum lifespan, and gestation time
fig4/  InCHIANTI prediction, disease, Cox, and Kaplan-Meier analyses
results_manifest.tsv  all included result tables in figure/panel order
validate_project.py   lightweight path, syntax, and table checks
```

Every figure directory contains:

- `scripts/`: compact analysis and plotting entry points;
- `results/`: manuscript result tables prefixed by panel order;
- `results/manifest.tsv`: the ordered table index for that figure;
- `datasets/README.md`: the expected input layout and required columns.

## Configure your datasets

The scripts do not contain machine-specific data paths. By default, each script
looks below its own figure directory, for example:

```text
fig1/datasets/all_predicted_ages_with_id.csv
fig2/datasets/array/GSE234461/
fig2/datasets/tissue/GSE111629/
fig2/datasets/species/BosTaurus/
fig3/datasets/loo_predictions/
fig4/datasets/inchianti/
```

Replace these placeholder directories with your datasets, or pass the relevant
`--data-dir`, `--dataset-dir`, `--results-dir`, or `--output-dir` option shown by
`python <script> --help`. A symlink to data stored elsewhere is also sufficient.
The compact tables under `results/` are included for manuscript reporting and
do not need to be regenerated merely to inspect the published values.

## Environment

Create an environment with the packages in `requirements.txt`. The plotting
scripts use editable-text PDF/SVG output and create output directories when
needed.

Run the lightweight repository checks from this directory:

```bash
python validate_project.py
```

Each figure README gives the input schema, analysis parameters, result-table
order, and example commands. `MANUSCRIPT_RESULTS_AND_METHODS.md` consolidates
the main numerical readouts and fixed analysis parameters for manuscript use.

The maintained figure-ordered interface is `fig1/` through `fig4/`.
