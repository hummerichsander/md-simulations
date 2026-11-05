#!/bin/bash
#SBATCH --job-name=alaX-amber14-implicit-330K-replica
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# Independent replicas r1-r3 of the polyalanine AMBER14 + GBn2 330 K runs (ala2, ala3, ala5, ala8):
# 10 ns at 2 fs, every frame saved, from configs/ala<n>-amber14-implicit-330K-r<k>.yaml. Together
# with the original r0 runs they are the reference set of the split-flows Boltzmann-generator
# benchmark (r0-r2 train, r3 is the holdout). Each replica is its own job with the walltime of its
# config's `slurm:` block (r0 needed 4.7-5.5 h); each writes its own output_subdir and Beamtime run
# (group alaX/ala<n>_AMBER14_implicit_330K, run r<k>), so a failure costs only that replica.
#
# Two modes in one file, as in resample-h-bonds-alaX.sh:
#
#   Submit (run it directly, from a GPU node; one sbatch per length and replica, then one
#   H-resampling job per length that waits for that length's replicas with afterok):
#
#     ./slurm/alaX-amber14-implicit-330K-replicas.sh --dry-run
#     SYSTEMS="2 3" REPLICAS="1 2 3" ./slurm/alaX-amber14-implicit-330K-replicas.sh
#     NO_RESAMPLE=1 ./slurm/alaX-amber14-implicit-330K-replicas.sh     # replicas only
#
#   Job (what sbatch runs; CONFIG names the config, set by the submit mode):
#
#     CONFIG=configs/ala2-amber14-implicit-330K-r1.yaml sbatch slurm/alaX-amber14-implicit-330K-replicas.sh
#
# Every axis is env-overridable and space-separated: SYSTEMS (default "2 3 5 8"), REPLICAS
# (default "1 2 3"). The H-resampling follow-up seeds replica k with 42 + k (see
# resample-h-bonds-alaX.sh). r0 is never part of this script.

set -euo pipefail

if [[ -z ${CONFIG:-} ]]; then
    # ---- submit mode -------------------------------------------------------------------------
    if [[ ${SLURM_JOB_NAME:-} == alaX-amber14-implicit-330K-replica ]]; then
        echo "error: submitted without CONFIG; run this script directly to submit, not through sbatch" >&2
        exit 2
    fi
    cd "$(dirname "${BASH_SOURCE[0]}")/.."

    SYSTEMS="${SYSTEMS:-2 3 5 8}"
    REPLICAS="${REPLICAS:-1 2 3}"
    NO_RESAMPLE="${NO_RESAMPLE:-0}"
    SELF="slurm/alaX-amber14-implicit-330K-replicas.sh"

    DRY_RUN=0
    [[ ${1:-} == --dry-run ]] && DRY_RUN=1

    # A value of a top-level `slurm:` key in a config (the keys are indented two spaces).
    slurm_key() { awk -v key="$2:" '$1 == key { print $2 }' "$1"; }

    for n in $SYSTEMS; do
        ids=()
        for k in $REPLICAS; do
            config="configs/ala${n}-amber14-implicit-330K-r${k}.yaml"
            [[ -f $config ]] || { echo "error: no such config: $config" >&2; exit 2; }
            job="$(slurm_key "$config" job_name)"
            time="$(slurm_key "$config" time)"
            cmd=(sbatch --parsable "--job-name=${job}" "--time=${time}" "--export=ALL,CONFIG=${config}" "$SELF")
            if (( DRY_RUN )); then
                echo "${cmd[*]}   # -> <jobid:ala${n}-r${k}>"
                ids+=("<jobid:ala${n}-r${k}>")
            else
                id="$("${cmd[@]}")"
                echo "ala${n} r${k}: job ${id}"
                ids+=("$id")
            fi
        done

        (( NO_RESAMPLE )) && continue
        # One resampling job per length, after all of that length's replicas succeeded. The
        # environment (SYSTEMS, REPLICAS) travels with sbatch's default --export=ALL.
        dep="afterok:$(IFS=:; echo "${ids[*]}")"
        resample="SYSTEMS=${n} REPLICAS=\"${REPLICAS}\" sbatch --dependency=${dep} slurm/resample-h-bonds-alaX.sh"
        if (( DRY_RUN )); then
            echo "${resample}"
        else
            SYSTEMS="$n" REPLICAS="$REPLICAS" sbatch --dependency="$dep" slurm/resample-h-bonds-alaX.sh
        fi
    done
    exit 0
fi

# ---- job mode ------------------------------------------------------------------------------
# Run from the directory the job was submitted from (the project root).
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

# Optional: this repo has no .env today; allexport so console scripts see its variables.
if [[ -f ./.env ]]; then
    set -o allexport
    source ./.env
    set +o allexport
fi

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

echo "=== ${CONFIG} ==="
md-sim run "$CONFIG"

echo "Finished ${CONFIG}."
