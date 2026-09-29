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
# slurm/resample-h-bonds.sh once per chain length against the matching instance of
# configs/alaX-amber14-implicit-unconstrained-330K.yaml. Works as a plain script as well as a job:
#
#   ./slurm/resample-h-bonds-alaX.sh [--dry-run]
#   SYSTEMS="2 5" ./slurm/resample-h-bonds-alaX.sh
#
# Output lands next to each input as trajectory_hresampled.trr (lossless, and unlike dcd fast to
# write over NFS; see resample-h-bonds.sh). Its other knobs (STRIDE, SEED, ...) pass through.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

SYSTEMS="${SYSTEMS:-2 3 4 5 6 7 8 9 10 11 12 16 24 32}"
TEMPLATE="${TEMPLATE:-configs/alaX-amber14-implicit-unconstrained-330K.yaml}"
DATA_DIR="${MD_DATA_ROOT:-$PWD/data}/alaX"
export OUT_EXT="${OUT_EXT:-trr}"

DRY_RUN=0
[[ ${1:-} == --dry-run ]] && DRY_RUN=1

CONFIG_DIR="$(mktemp -d)"
trap 'rm -rf "$CONFIG_DIR"' EXIT

for n in $SYSTEMS; do
    config="${CONFIG_DIR}/ala${n}.yaml"
    run="${DATA_DIR}/ala${n}_AMBER14_implicit_330K"
    # the template is written for ala12; every occurrence names the system
    sed "s/ala12/ala${n}/g" "$TEMPLATE" > "$config"

    cmd=(bash slurm/resample-h-bonds.sh "$config" "${run}/trajectory.dcd" "${run}/initial_structure.pdb")
    if (( DRY_RUN )); then
        echo "${cmd[*]}"
    else
        "${cmd[@]}"
    fi
done
