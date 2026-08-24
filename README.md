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
python -m pytest
python -m ruff check .
```
