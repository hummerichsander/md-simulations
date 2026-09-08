#!/bin/bash
#SBATCH --job-name=trp-cage-amber14-implicit-seeded-330K
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%A_%a.out
#SBATCH --error=./slurm/output/%x-%A_%a.err

# Seeded 330 K replicas, wave 1: 8 x 300 ns from independent starting structures
# (~7.4 h each at the measured 970 ns/day, so ~7.4 h wall clock across 8 GPUs).
#
# Why seeded rather than simply longer: the unseeded 300 ns run at 330 K reaches the
# unfolded core in a single 26 ns episode (237-263 ns), 5.4% of frames, giving roughly one
# independent unfolding event. Extending one trajectory would still explore the unfolded
# basin serially; independent starts spread across the folding coordinate build that
# ensemble in parallel.
#
# 330 K was chosen over 360 K on structure, not statistics: the 360 K unfolded ensemble
# carries 0.68 non-native contacts per native contact (max 2.28) versus 0.41 at 330 K, so it
# is collapsed and misregistered rather than unfolded.
#
# r00-r04 start from 330 K structures spanning Q6 = 0.00 to 0.73; r05-r07 import expanded
# 390 K unfolded structures (screened to non-native <= 0.45, so not the 360 K-type state).
# Every frame they produce is 330 K dynamics and therefore a valid sample of the 330 K mean
# force, whatever the starting point. No folded starts: the unseeded run is already 56%
# folded, so replicas go only where data is scarce.
#
# Output is for the force-matching training set. Basin populations and the melting curve
# stay sourced from the unseeded AMBER14_implicit_scan_330K run: a seeded trajectory begins
# off-equilibrium by construction, so its basin frequencies are not Boltzmann.
#
# Each task writes its own output_subdir, so tasks are fully independent: one failure or
# requeue costs only that replica.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output
source .venv/bin/activate
export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"
TAG="$(printf 'r%02d' "${SLURM_ARRAY_TASK_ID:-0}")"
echo "=== array task ${SLURM_ARRAY_TASK_ID:-0}: seeded 330 K replica ${TAG} ==="
md-sim run "configs/trpcage-amber14-implicit-seeded-330K-${TAG}.yaml"
echo "Finished ${TAG}."
