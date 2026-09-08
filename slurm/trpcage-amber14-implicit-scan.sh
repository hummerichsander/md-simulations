#!/bin/bash
#SBATCH --job-name=trp-cage-amber14-implicit-scan
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%A_%a.out
#SBATCH --error=./slurm/output/%x-%A_%a.err

# Trp-cage GBn2 temperature scan: 300 ns at each of 300/330/360/390 K, one temperature per
# array task on its own GPU (~7.4 h each at the measured 975 ns/day, so ~7.4 h wall clock).
# Purpose: locate a temperature where the folded and unfolded states are both populated. The
# existing 100 ns 300 K run stayed folded (Rg 0.687 +- 0.014 nm) while the CGSchNet ensemble
# spans Rg 0.61-1.00 nm, so no AMBER frame covers the region the CG model visits.
#
# Each task writes its own output_subdir, so tasks are fully independent: one failure or
# requeue costs only that temperature.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

TEMPS=(300 330 360 390)
T="${TEMPS[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "=== array task ${SLURM_ARRAY_TASK_ID:-0}: ${T} K ==="
md-sim run "configs/trpcage-amber14-implicit-scan-${T}K.yaml"

echo "Finished ${T} K."
