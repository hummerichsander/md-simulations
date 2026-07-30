#!/bin/bash
#SBATCH --job-name=ala2-amber14-implicit-330K
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-40:00:00
#SBATCH --partition=a30
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

set -euo pipefail

# Run from the directory the job was submitted from (the project root).
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

md-sim run configs/ala2-amber14-implicit-330K.yaml

echo "Simulation finished."
