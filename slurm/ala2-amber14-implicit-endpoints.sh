#!/bin/bash
#SBATCH --job-name=ala2-amber14-implicit-endpoints
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# Classical (non-exchanging) MD of ala2, AMBER14 + GBn2 implicit solvent, at the two endpoint
# temperatures a T-REMD ladder for this system would span: 300 K and 600 K. One job per
# temperature, from configs/ala2-amber14-implicit-endpoints-<T>K.yaml. 350 ns each at 1 ps frames.
#
# Two modes in one file, as in slurm/alaX-amber14-implicit-remd-330K.sh:
#
#   Submit (run it directly):
#
#     ./slurm/ala2-amber14-implicit-endpoints.sh --dry-run
#     ./slurm/ala2-amber14-implicit-endpoints.sh                 # both temperatures
#     TEMPS=600 ./slurm/ala2-amber14-implicit-endpoints.sh       # just one
#
#   Job (what sbatch runs; CONFIG names the config, set by the submit mode):
#
#     CONFIG=configs/ala2-amber14-implicit-endpoints-300K.yaml sbatch slurm/ala2-amber14-implicit-endpoints.sh
#
# TEMPS (default "300 600") is space-separated. The two jobs are independent and run concurrently;
# set SEQUENTIAL=1 to chain the second after the first with afterany, which keeps the pair to one
# GPU. NO_RESAMPLE=1 skips the H-resampling job that otherwise follows both runs with afterok.
#
# NOTE: these are plain-MD runs and `md-sim run --resume` only works for replica-exchange configs,
# so a job that hits the wall limit cannot be continued -- it leaves a truncated (but readable)
# trajectory.dcd. 350 ns is sized for ~12 h at the ~40 ns/h this system reaches on an a30
# (3 x the 13.9 ns/h per replica measured for the 3-replica REMD train run).

set -euo pipefail

if [[ -z ${CONFIG:-} ]]; then
    # ---- submit mode -------------------------------------------------------------------------
    if [[ ${SLURM_JOB_NAME:-} == ala2-amber14-implicit-endpoints ]]; then
        echo "error: submitted without CONFIG; run this script directly to submit, not through sbatch" >&2
        exit 2
    fi
    cd "$(dirname "${BASH_SOURCE[0]}")/.."

    TEMPS="${TEMPS:-300 600}"
    SEQUENTIAL="${SEQUENTIAL:-0}"
    NO_RESAMPLE="${NO_RESAMPLE:-0}"
    SELF="slurm/ala2-amber14-implicit-endpoints.sh"

    DRY_RUN=0
    [[ ${1:-} == --dry-run ]] && DRY_RUN=1

    # A value of a top-level `slurm:` key in a config (the keys are indented two spaces).
    slurm_key() { awk -v key="$2:" '$1 == key { print $2 }' "$1"; }

    ids=()
    last=""
    for t in $TEMPS; do
        config="configs/ala2-amber14-implicit-endpoints-${t}K.yaml"
        [[ -f $config ]] || { echo "error: no such config: $config" >&2; exit 2; }
        job="$(slurm_key "$config" job_name)"
        time="$(slurm_key "$config" time)"
        cmd=(sbatch --parsable "--job-name=${job}" "--time=${time}" "--export=ALL,CONFIG=${config}")
        [[ $SEQUENTIAL == 1 && -n $last ]] && cmd+=("--dependency=afterany:${last}")
        cmd+=("$SELF")
        if (( DRY_RUN )); then
            id="<jobid:${t}K>"
            echo "${cmd[*]}"
        else
            id="$("${cmd[@]}")"
            echo "ala2 ${t} K: job ${id}"
        fi
        ids+=("$id")
        last="$id"
    done

    if (( ! NO_RESAMPLE )) && (( ${#ids[@]} )); then
        # X-H resampling, after both runs succeeded: `constraints: hbonds` removes the X-H bond
        # energy term, so the frames are not samples of the unconstrained potential.
        dep="afterok:$(IFS=:; echo "${ids[*]}")"
        if (( DRY_RUN )); then
            echo "SYSTEMS=2 REPLICAS= sbatch --dependency=${dep} slurm/resample-h-bonds-alaX.sh   # check its axes first"
        else
            echo "note: run H-resampling yourself once both jobs are done; see slurm/resample-h-bonds-alaX.sh" >&2
        fi
    fi
    exit 0
fi

# ---- job mode ------------------------------------------------------------------------------
# Run from the directory the job was submitted from (the project root).
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

echo "=== ${CONFIG} ==="
md-sim run "$CONFIG"

echo "Finished ${CONFIG}."
