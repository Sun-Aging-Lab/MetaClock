#!/bin/bash
#SBATCH --job-name=tissue
#SBATCH --array=0-89
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]}")/../..}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DATASETS=(GSE175458 GSE232891 GSE287331)
GSE_INDEX=$((SLURM_ARRAY_TASK_ID / 30))
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID % 30))
python -m generalization.tissue.train_tissue_generalization \
  --gse "${DATASETS[${GSE_INDEX}]}" --size-index "${SIZE_INDEX}"
