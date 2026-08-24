# MetaClock workflow

This repository contains workflows for chronological-age
prediction, cross-domain generalization, biological-age and disease analysis,
functional enrichment, and intervention analysis. Every external dataset input
uses the explicit placeholder
`Path("data/your_file")  # fill in your path`.

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

The repository also provides the importable `metaclock` package

```
python -m pip install -e /orcd/pool/007/edsun_shared/yncheng/metaclock/
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

Pass any ordered list of pyaging component-model names. The fitted result is an
sklearn-compatible pipeline whose `predict()` method first runs the same
components in the same order and then applies the trained meta-regressor:

```
import metaclock

component_model_list = [
    "horvath2013",
    "hannum",
    "dnamphenoage",
]

model = metaclock.fit(
    train_adata,
    component_model_list,
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
    component_model_list,
)
```

Supported architectures are `elastic_net`, `lasso`, `ridge`,
`linear_regression`, `random_forest`, `svr`, `mlp`, and `lightgbm`. The `mlp`
architecture uses `StandardScaler` followed by sklearn's `MLPRegressor`; its
network parameters can be overridden through `model_params`. LightGBM is part
of the standard locked environment and uses its sklearn-compatible
`LGBMRegressor` interface.

Fitted models can be saved with joblib and loaded for later prediction:

```python
import joblib

joblib.dump(model, model_output_path)
loaded_model = joblib.load(model_output_path)
predicted_age = loaded_model.predict(test_adata)
```

