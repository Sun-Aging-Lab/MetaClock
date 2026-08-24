#!/bin/bash
#SBATCH --job-name=species_mlp
#SBATCH --array=0-10%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
SOURCES=(BosTaurus DelphinapterusLeucas EquusCaballus HeterocephalusGlaber HomoSapiens MacacaMulatta MusMusculus OvisAries PapioHamadryas RattusNorvegicus alltissueconcat)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
python -m generalization.species.train_species_submodels --model-type mlp_elu --source "${SOURCES[$SLURM_ARRAY_TASK_ID]}" --device cuda
