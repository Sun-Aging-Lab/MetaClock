# GEO intervention analysis: frozen 77-comparison workflow

This directory contains the two repository-local entry points for the formal
analysis universe of 77 intervention comparisons from 64 GEO series. Run both
commands from the repository root. All accepted paths are relative to that root;
absolute paths and paths outside the repository are rejected.

The published 672-participant biological MetaClock is reconstructed directly
from the two tracked artifacts in
`biological_age_prediction_and_mortality_analysis/model_weights/`:

- `biometaclock_672_participants_weights.csv` contains the ordered 17 PyAging
  component-clock coefficients.
- `biometaclock_672_participants_metadata.json` contains the raw intercept and
  model metadata.

No external `joblib` model is needed.

The tracked MetaClock weights above are only the final 17-to-1 linear model.
PyAging 0.1.30 still needs the 17 upstream component-clock `.pt` files to turn
each DNAm beta matrix into those 17 inputs. On first use, PyAging downloads
them into `data/intervention/pyaging_cache`; together they occupy about 22.5
GiB. Later runs reuse that repository-local cache and do not redownload it.

## Main scripts

1. `intervention_analysis/fetch_geo_datasets_and_metadata.py`

   Fetches GEO metadata and the exact 64 DNAm matrix artifacts listed in the
   frozen acquisition manifest. By default it writes metadata to
   `data/intervention/geo_fetch_77/metadata` and matrices to
   `data/intervention/geo_fetch_77/matrices`. The expected matrix download is
   approximately 28.7 GiB.

   ```bash
   conda run --no-capture-output -n metaclock \
     python -m intervention_analysis.fetch_geo_datasets_and_metadata
   ```

2. `intervention_analysis/run_biometaclock_intervention_analysis.py`

   Loads the frozen BioMetaClock weights, predicts every retained GSE with its
   complete frozen branch context, performs the design-aware two-sided paired or
   Welch t-test, applies Benjamini-Hochberg correction across the 77 retained
   comparisons, cross-checks BH with SciPy, and creates PDF/SVG FDR volcano
   plots. With no path arguments, it loads the tracked model weights above,
   reads the matrices produced by step 1, keeps the PyAging cache inside the
   repository, and writes results to
   `results/intervention/biometaclock_77`.

   ```bash
   conda run --no-capture-output -n metaclock \
     python -m intervention_analysis.run_biometaclock_intervention_analysis
   ```
