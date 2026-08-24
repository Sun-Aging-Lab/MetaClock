# MetaClock workflow

This repository contains workflows for chronological-age
prediction, cross-domain generalization, biological-age and disease analysis,
functional enrichment, and intervention analysis. Every external dataset input
uses the explicit placeholder
`Path("data/your_file")  # fill in your path`.

Run workflows from the repository root. File and directory arguments are
repository-relative; formal entry points reject absolute paths and paths that
escape the repository. Downloaded data, model caches, generated results, and
plotting caches remain under `data/`, `results/`, or other documented
repository-relative directories.

## Project overview

MetaClock is an integrated framework that combines
predictions from multiple established epigenetic clocks into meta
models. This repository organizes the reproducible downstream workflows used
to evaluate chronological-age prediction, quantify generalization across array
platforms, tissues, and species, and study biological-age associations with
disease and interventions.

## Environment

Install [uv](https://docs.astral.sh/uv/) and create the locked Python 3.11
environment from the repository root:

```bash
uv sync --frozen
```

## MetaClock Python package

The repository also provides the importable `metaclock` package. The standard
locked environment includes scikit-learn, pyaging, and LightGBM. From the
repository root, the existing environment command installs the package in
editable mode together with all runtime dependencies:

```bash
uv sync --frozen
```

### Input contract

`metaclock.fit()` accepts an in-memory `anndata.AnnData` object or an `.h5ad`
file in pyaging's standard orientation:

- rows are samples;
- columns are DNAm or phenotype features;
- feature names are stored in `adata.var_names`;
- the numeric training response is stored in `adata.obs["age"]` by default.

Clocks that use phenotype covariates, such as GrimAge, require those covariates
in the input matrix and `adata.var_names` according to pyaging's input rules.

### Fit and predict

The built-in chronological preset contains the requested 16 component models.
The fitted result is an sklearn-compatible pipeline whose `predict()` method
first runs the same pyaging components and then applies the trained
meta-regressor:

```python
import metaclock

model = metaclock.fit(
    train_adata,
    metaclock.CHRONOLOGICAL_16_COMPONENT_MODELS,
    "elastic_net",
    target="age",
)

predicted_age = model.predict(test_adata)
```

The three positional arguments are the input data, component-model list, and
model architecture. Set `pyaging_dir` when using a specific local pyaging
weight cache. The component ordering used for fitting is retained for every
later prediction.

To run the component clocks without fitting a meta-model:

```python
component_outputs = metaclock.predict_components(
    test_adata,
    metaclock.CHRONOLOGICAL_16_COMPONENT_MODELS,
)
```

Supported architectures are `elastic_net`, `lasso`, `ridge`,
`linear_regression`, `random_forest`, `svr`, and `lightgbm`. LightGBM is part
of the standard locked environment and uses its sklearn-compatible
`LGBMRegressor` interface.

Fitted models can be saved with joblib and loaded for later prediction:

```python
import joblib

joblib.dump(model, model_output_path)
loaded_model = joblib.load(model_output_path)
predicted_age = loaded_model.predict(test_adata)
```

The package delegates component loading, feature alignment, preprocessing,
inference, and postprocessing to pyaging. Regression, scaling, and
cross-validation use maintained scikit-learn or LightGBM implementations.
Non-finite targets or component outputs raise explicit errors rather than
being silently imputed.

The implementation was validated with Python 3.11.14, pyaging 0.1.30,
anndata 0.12.7, numpy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0, and LightGBM
4.6.0. In the InCHIANTI Illumina 450K integration check, all 16 component
outputs matched the existing pyaging reference with a maximum absolute error
of 1.42e-13. A donor-grouped 18-sample holdout produced MAE 1.86 years, RMSE
2.13 years, R2 0.975, and Pearson r 0.989.
