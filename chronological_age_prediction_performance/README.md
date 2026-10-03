# Chronological-age prediction performance

This directory reproduces the chronological-age comparisons. It contains two workflows:
an all-architecture/component comparison and a cumulative component-clock
ablation study. 

## Exported MetaClock-chronological weights

MetaClock-chronological raw coefficients and
training metadata are stored in
`model_weights/metaclock_chronological_3353_samples_weights.csv` and
`model_weights/metaclock_chronological_3353_samples_metadata.json`.
Generate them with:

```bash
python -m chronological_age_prediction_performance.export_metaclock_weights
```

## Run all architecture, component, and border plots

Run:

```bash
python -m chronological_age_prediction_performance.train_test_architectures
```

## Run all ablations

Run:

```bash
python -m chronological_age_prediction_performance.train_test_ablation
```

