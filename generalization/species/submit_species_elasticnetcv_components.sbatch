#!/bin/bash
#SBATCH --job-name=species_encv
#SBATCH --array=0-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
SOURCES=(BosTaurus CanisLupusFamiliaris EquusCaballus HeterocephalusGlaber HomoSapiens MacacaMulatta MusMusculus OvisAries PapioHamadryas RattusNorvegicus alltissueconcat)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m generalization.species.train_species_submodels --model-type elasticnetcv --source "${SOURCES[$SLURM_ARRAY_TASK_ID]}" --n-jobs "${SLURM_CPUS_PER_TASK}"
