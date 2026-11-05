#!/bin/bash
#SBATCH --job-name=resample-h-bonds-alaX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00:00
#SBATCH --partition=gshort
#SBATCH --mem=8G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# Resample the X-H bond lengths of every ala-X reference trajectory, by running
# slurm/resample-h-bonds.sh once per chain length and replica against the matching instance of
# configs/alaX-amber14-implicit-unconstrained-330K.yaml. Works as a plain script as well as a job:
#
#   ./slurm/resample-h-bonds-alaX.sh [--dry-run]
#   SYSTEMS="2 5" ./slurm/resample-h-bonds-alaX.sh
#   SYSTEMS=8 REPLICAS="1 2 3" ./slurm/resample-h-bonds-alaX.sh --dry-run
#
# Replica k lives in alaX/ala<n>_AMBER14_implicit_330K-r<k>; replica 0 is the original run, whose
# directory carries no suffix (alaX/ala<n>_AMBER14_implicit_330K). REPLICAS defaults to "0", i.e.
# the behaviour from before the replica axis existed -- note that this rewrites the r0 files
# (identically: same seed, same input), so name the replicas you mean.
#
# Output lands next to each input as trajectory_hresampled.trr (lossless, and unlike dcd fast to
# write over NFS; see resample-h-bonds.sh). Its other knobs (STRIDE, ...) pass through.
#
# SEED: every replica gets its own seed, SEED_BASE + k (SEED_BASE defaults to SEED, else 42, so r0
# keeps the 42 it was resampled with). One seed shared by all replicas would give frame i of every
# replica the same X-H noise draw, correlating, e.g., a training and a holdout frame at the same
# index. Do not pass a per-replica SEED to this script: it is read only as the base.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

SYSTEMS="${SYSTEMS:-2 3 4 5 6 7 8 9 10 11 12 16 24 32}"
REPLICAS="${REPLICAS:-0}"
SEED_BASE="${SEED_BASE:-${SEED:-42}}"
TEMPLATE="${TEMPLATE:-configs/alaX-amber14-implicit-unconstrained-330K.yaml}"
DATA_DIR="${MD_DATA_ROOT:-$PWD/data}/alaX"
export OUT_EXT="${OUT_EXT:-trr}"

DRY_RUN=0
[[ ${1:-} == --dry-run ]] && DRY_RUN=1

CONFIG_DIR="$(mktemp -d)"
trap 'rm -rf "$CONFIG_DIR"' EXIT

for n in $SYSTEMS; do
    config="${CONFIG_DIR}/ala${n}.yaml"
    # the template is written for ala12; every occurrence names the system
    sed "s/ala12/ala${n}/g" "$TEMPLATE" > "$config"

    for k in $REPLICAS; do
        run="${DATA_DIR}/ala${n}_AMBER14_implicit_330K"
        (( k > 0 )) && run="${run}-r${k}"

        cmd=(env "SEED=$((SEED_BASE + k))" bash slurm/resample-h-bonds.sh "$config" "${run}/trajectory.dcd" "${run}/initial_structure.pdb")
        if (( DRY_RUN )); then
            echo "${cmd[*]}"
        else
            echo "=== ala${n} r${k} (seed $((SEED_BASE + k))) ==="
            "${cmd[@]}"
        fi
    done
done
