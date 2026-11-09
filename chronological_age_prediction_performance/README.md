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

ElasticNet, Lasso, and Ridge select alpha with five-fold `GridSearchCV` over a
`StandardScaler`/regressor pipeline. Each split fits the scaler using only that
split's training samples; the selected pipeline is then refit on all training
samples. The alpha grid is 30 logarithmically spaced values from 1e-5 to 1.
ElasticNet uses `l1_ratio=0.5`. Selection uses
negative MSE for ElasticNet/Lasso and R-squared for Ridge.

The MLP uses a 16→100→50→1 architecture with
ReLU, BatchNorm and Dropout(0.5) after each hidden layer, AdamW with learning
rate 1e-3 and weight decay 1e-3, batch size 128, 500 epochs, and seed 105.
`ReduceLROnPlateau(mode="min", factor=0.2, patience=30, threshold=1e-2)` monitors
the training mean MSE each epoch. The threshold belongs to learning-rate
scheduling; it does not enable early stopping. Training defaults to CPU.
Initialization, shuffle, and dropout share the Torch RNG.

## Run all ablations

Run:

```bash
python -m chronological_age_prediction_performance.train_test_ablation
```

