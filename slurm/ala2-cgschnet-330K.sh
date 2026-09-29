#!/bin/bash
#SBATCH --job-name=ala2-cgschnet-330K
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --partition=gshort
#SBATCH --exclude=gpu05
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# CG MD with the ala2 CGSchNet: 32 replicas x 5 ns at 330 K, one batched forward per step
# (~165 steps/s on an RTX 4060, so ~4 h).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

md-sim run configs/ala2-cgschnet-330K.yaml

echo "Simulation finished."
