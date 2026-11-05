#!/bin/bash
#SBATCH --job-name=cln-cgschnet-330K
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-40:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# CG MD with the chignolin CGSchNet: 32 replicas x 5 ns at 330 K, one batched forward per step.
# Compare its Q / CA-RMSD / TICA free energies against the eight AA replicas it was fit on.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

md-sim run configs/cln-cgschnet-330K.yaml

echo "Simulation finished."
