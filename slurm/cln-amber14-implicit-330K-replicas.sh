#!/bin/bash
#SBATCH --job-name=cln-amber14-implicit-330K-replicas
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-16:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%A_%a.out
#SBATCH --error=./slurm/output/%x-%A_%a.err

# Clean chignolin production set: 8 x 500 ns of 1UAO in AMBER14 + GBn2 at 330 K, one replica per
# NMR model (configs/cln-amber14-implicit-330K-rNN.yaml), each saving positions and forces every
# 5 ps (~11.5 h per task). Each task writes its own output_subdir, so one failure or requeue costs
# only that replica.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output
source .venv/bin/activate
export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"
TAG="$(printf 'r%02d' "${SLURM_ARRAY_TASK_ID:-0}")"
echo "=== array task ${SLURM_ARRAY_TASK_ID:-0}: 330 K replica ${TAG} ==="
md-sim run "configs/cln-amber14-implicit-330K-${TAG}.yaml"
echo "Finished ${TAG}."
