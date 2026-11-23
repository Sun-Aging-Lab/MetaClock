#!/bin/bash
#SBATCH --job-name=GSE184221_methylchip40
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]}")/../..}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m generalization.array.GSE184221_methylchip40 --size-index "${SLURM_ARRAY_TASK_ID}"
