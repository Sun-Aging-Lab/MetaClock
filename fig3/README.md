# Figure 3 — evolutionary and life-history relationships

This directory contains the code and compact inputs/results for three ordered
panels:

1. molecular evolutionary distance;
2. absolute maximum-lifespan difference;
3. absolute gestation-time difference.

Each result table contains the same 90 directed target-component pairs.  A-to-B
and B-to-A share an undirected distance or absolute trait difference, but have
separate coefficients because each target species is fitted independently.

## Run

The committed compact tables can be plotted directly:

```bash
python fig3/scripts/evolution_relationships.py plot
```

To recompute the coefficients, place the ten leave-one-species-out prediction
tables in `fig3/datasets/loo_predictions/`, then run:

```bash
python fig3/scripts/evolution_relationships.py all
```

Use `--data-dir`, `--loo-dir`, `--results-dir`, and `--output-dir` to override
the project-relative defaults.  If your source data live elsewhere, replace
the corresponding directory or pass it explicitly; no machine-specific path
is encoded in this project.

## Fixed analysis parameters

Upstream leave-one-species predictions use median imputation,
`StandardScaler`, and `ElasticNetCV(cv=5, n_alphas=10, max_iter=10000,
random_state=105, l1_ratio=0.5)`. For each target species, the nine
`<component>_elasticnetcv` predictions are standardized and entered into
`ElasticNet(alpha=0.001, positive=True, l1_ratio=0.5)`. The default
`l1_ratio=0.5` is fixed rather than tuned. This produces 90 non-negative
directed coefficients across ten species.

Each panel shows all 90 points, Pearson's r, and a descriptive nonlinear trend:

- `SplineTransformer(n_knots=3, degree=3, include_bias=False)`;
- `ElasticNet(alpha=0.001, l1_ratio=0.5)` on the spline basis;
- a 200-point x grid;
- 200 row-bootstrap refits with replacement, seed 42;
- pointwise 2.5th and 97.5th percentiles.

The rows repeatedly reuse the same ten species and component models.  Pearson's
r and the bootstrap ribbon are therefore descriptive, not independent-species
inference; no p-value is reported.

The evolutionary-distance canvas is 100 x 75 mm.  The maximum-lifespan and
gestation canvases match the generalization panels at 90 x 60 mm.
