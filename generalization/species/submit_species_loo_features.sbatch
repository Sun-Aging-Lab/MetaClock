#!/bin/bash
#SBATCH --job-name=species_loo
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
SPECIES=(BosTaurus CanisLupusFamiliaris EquusCaballus HeterocephalusGlaber HomoSapiens MacacaMulatta MusMusculus OvisAries PapioHamadryas RattusNorvegicus)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m generalization.species.build_species_loo_features --species "${SPECIES[$SLURM_ARRAY_TASK_ID]}"
