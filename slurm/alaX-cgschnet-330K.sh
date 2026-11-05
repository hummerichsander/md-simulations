#!/bin/bash
#SBATCH --job-name=alaX-cgschnet-330K
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-40:00:00
#SBATCH --partition=gshort
#SBATCH --exclude=gpu05
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# CG MD with one polyalanine's CGSchNet: 32 replicas x 5 ns at 330 K.
#
#   SYSTEM=ala8 sbatch --job-name=ala8-cgschnet-330K slurm/alaX-cgschnet-330K.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

SYSTEM=${SYSTEM:?set SYSTEM, e.g. SYSTEM=ala8}
md-sim run "configs/${SYSTEM}-cgschnet-330K.yaml"

echo "Simulation finished."
