#!/bin/bash
#SBATCH --job-name=alaX-amber14-implicit-remd-330K
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# T-REMD runs of the polyalanine AMBER14 + GBn2 systems with a 330 K target (ala2, ala3, ala5, ala8),
# from configs/ala<n>-amber14-implicit-remd-330K-<run>.yaml. They replace the plain-MD replica set
# r0-r3, which did not converge the alpha_L basin. <run> is
#
#   pilot     wide 330-600 K ladder, 2 ns, full output at every temperature: calibrates the others
#   train     production run, helix start (pdbs/ala<n>.pdb)       -> Boltzmann-generator training
#   holdout   production run, PPII start (pdbs/ppii/ala<n>.pdb)   -> independent holdout
#
# Each run is one job on one GPU: the replicas step one after another inside it. It writes its own
# output_subdir and Beamtime group (alaX/ala<n>_AMBER14_implicit_remd_330K/<run>, one run per
# temperature), so a failure costs only that run.
#
# Two modes in one file, as in alaX-amber14-implicit-330K-replicas.sh:
#
#   Submit (run it directly; one sbatch per length and run, then, for train/holdout, one
#   X-H equilibration array per length that waits for that length's runs with afterok):
#
#     ./slurm/alaX-amber14-implicit-remd-330K.sh --dry-run
#     SYSTEMS="2 3" ./slurm/alaX-amber14-implicit-remd-330K.sh                 # pilots
#     RUNS="train holdout" ./slurm/alaX-amber14-implicit-remd-330K.sh          # production
#     RESUME=1 SYSTEMS=8 RUNS=train ./slurm/alaX-amber14-implicit-remd-330K.sh # continue from checkpoint
#
#   Job (what sbatch runs; CONFIG names the config, set by the submit mode):
#
#     CONFIG=configs/ala2-amber14-implicit-remd-330K-pilot.yaml sbatch slurm/alaX-amber14-implicit-remd-330K.sh
#
# Every axis is env-overridable and space-separated: SYSTEMS (default "2 3 5 8"), RUNS (default
# "pilot"). RESUME=1 passes --resume, which continues each run from its checkpoint/ (same platform
# as the run started on); without it, a run whose output dir holds a checkpoint fails rather than
# overwrites. X-H equilibration never follows a pilot, and is skipped entirely with
# NO_EQUILIBRATE=1 (NO_RESAMPLE=1 is still honoured, since it used to mean this step).
#
# That step is slurm/equilibrate-h-bonds-alaX.sh, not the analytic resampler it used to be:
# it samples p(x_H | x_heavy) in the real unconstrained potential instead of drawing each
# length from an isolated-harmonic Gaussian. It costs dynamics per frame -- reach for STRIDE
# on a long run -- and writes <trajectory>_hequilibrated_sNN.trr, one shard per array task,
# where the resampler wrote a single trajectory_hresampled.trr.
#
# At most MAX_PARALLEL (default 3) of these GPU jobs run at once: jobs are dealt round-robin onto
# MAX_PARALLEL lanes, each waiting for the previous job of its lane with afterany, so a failed run
# does not stall its lane. Only these jobs are counted, not other jobs you have in the queue.

set -euo pipefail

if [[ -z ${CONFIG:-} ]]; then
    # ---- submit mode -------------------------------------------------------------------------
    if [[ ${SLURM_JOB_NAME:-} == alaX-amber14-implicit-remd-330K ]]; then
        echo "error: submitted without CONFIG; run this script directly to submit, not through sbatch" >&2
        exit 2
    fi
    cd "$(dirname "${BASH_SOURCE[0]}")/.."

    SYSTEMS="${SYSTEMS:-2 3 5 8}"
    RUNS="${RUNS:-pilot}"
    RESUME="${RESUME:-0}"
    NO_EQUILIBRATE="${NO_EQUILIBRATE:-${NO_RESAMPLE:-0}}"
    MAX_PARALLEL="${MAX_PARALLEL:-3}"
    SELF="slurm/alaX-amber14-implicit-remd-330K.sh"

    DRY_RUN=0
    [[ ${1:-} == --dry-run ]] && DRY_RUN=1

    # A value of a top-level `slurm:` key in a config (the keys are indented two spaces).
    slurm_key() { awk -v key="$2:" '$1 == key { print $2 }' "$1"; }

    lane_last=()   # per lane, the id of the job submitted last
    count=0
    for n in $SYSTEMS; do
        ids=()
        postprocess_runs=()
        for run in $RUNS; do
            config="configs/ala${n}-amber14-implicit-remd-330K-${run}.yaml"
            [[ -f $config ]] || { echo "error: no such config: $config" >&2; exit 2; }
            job="$(slurm_key "$config" job_name)"
            time="$(slurm_key "$config" time)"
            lane=$(( count % MAX_PARALLEL ))
            count=$(( count + 1 ))
            cmd=(sbatch --parsable "--job-name=${job}" "--time=${time}" "--export=ALL,CONFIG=${config},RESUME=${RESUME}")
            [[ -n ${lane_last[$lane]:-} ]] && cmd+=("--dependency=afterany:${lane_last[$lane]}")
            cmd+=("$SELF")
            if (( DRY_RUN )); then
                id="<jobid:ala${n}-${run}>"
                echo "${cmd[*]}   # lane ${lane} -> ${id}"
            else
                id="$("${cmd[@]}")"
                echo "ala${n} ${run}: job ${id} (lane ${lane})"
            fi
            lane_last[$lane]="$id"
            if [[ $run != pilot ]]; then
                ids+=("$id")
                postprocess_runs+=("$run")
            fi
        done

        (( NO_EQUILIBRATE )) && continue
        (( ${#ids[@]} )) || continue
        # X-H equilibration per length, after all of that length's production runs succeeded. The
        # dispatcher submits one array per run and needs the submit node, so it is called here
        # rather than sbatched: the dependency goes to the arrays themselves.
        dep="afterok:$(IFS=:; echo "${ids[*]}")"
        equilibrate=(env "SYSTEMS=$n" "REPLICAS=" "REMD_RUNS=${postprocess_runs[*]}" "DEPENDENCY=$dep"
                     ./slurm/equilibrate-h-bonds-alaX.sh)
        if (( DRY_RUN )); then
            echo "${equilibrate[*]} --dry-run"
            "${equilibrate[@]}" --dry-run
        else
            "${equilibrate[@]}"
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

args=("$CONFIG")
[[ ${RESUME:-0} == 1 ]] && args+=(--resume)

echo "=== ${CONFIG} ${args[*]:1} ==="
md-sim run "${args[@]}"

echo "Finished ${CONFIG}."
