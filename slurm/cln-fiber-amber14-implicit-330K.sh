#!/bin/bash
#SBATCH --job-name=cln-fiber-amber14-implicit-330K
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=a30
#SBATCH --exclude=gpu01
#SBATCH --mem=20G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# Fiber baseline for the split-flows backmapping comparison: six chignolin runs, one per
# representative of data/cln-ms/representatives.npz, from
# configs/cln-fiber-amber14-implicit-330K-rep<k>.yaml.
#
# Each run freezes the 47 slow atoms (N CA CB C O per residue) of its representative and samples
# the 91 fast dofs conditioned on them -- the same fiber the normalizing flow backmaps into. The
# MD question is how long that takes to decorrelate; the flow draws independent samples.
#
# Freezing by zero mass forbids constraints that touch a frozen atom, so these runs use the
# *unconstrained* force field (`constraints: none`) at a 0.5 fs timestep -- the same potential the
# chignolin reference dataset is being re-created in, by equilibrating the constrained trajectory
# in it. Both sides of the comparison therefore live in one force field and nothing is
# post-processed here: no X-H resampling follows these runs.
#
# Two modes in one file, as in slurm/alaX-amber14-implicit-remd-330K.sh:
#
#   Submit (run it directly; one sbatch per representative):
#
#     ./slurm/cln-fiber-amber14-implicit-330K.sh --dry-run
#     ./slurm/cln-fiber-amber14-implicit-330K.sh              # all six
#     REPS="0" ./slurm/cln-fiber-amber14-implicit-330K.sh     # just the first mode
#
#   Job (what sbatch runs; CONFIG names the config, set by the submit mode):
#
#     CONFIG=configs/cln-fiber-amber14-implicit-330K-rep0.yaml sbatch slurm/cln-fiber-amber14-implicit-330K.sh
#
# REPS (default "0 1 2 3 4 5") selects the representatives. At most MAX_PARALLEL (default 2) of
# these GPU jobs run at once: jobs are dealt round-robin onto MAX_PARALLEL lanes, each waiting for
# the previous job of its lane with afterany, so a failed run does not stall its lane.
#
# The venv needs the `openmm-cuda` extra, or OpenMM has no CUDA platform and the job runs on the
# CPU without saying so: `uv sync --extra openmm-cuda`. The job mode checks and refuses.

set -euo pipefail

if [[ -z ${CONFIG:-} ]]; then
    # ---- submit mode -------------------------------------------------------------------------
    if [[ ${SLURM_JOB_NAME:-} == cln-fiber-amber14-implicit-330K ]]; then
        echo "error: submitted without CONFIG; run this script directly to submit, not through sbatch" >&2
        exit 2
    fi
    cd "$(dirname "${BASH_SOURCE[0]}")/.."

    REPS="${REPS:-0 1 2 3 4 5}"
    MAX_PARALLEL="${MAX_PARALLEL:-2}"
    SELF="slurm/cln-fiber-amber14-implicit-330K.sh"

    DRY_RUN=0
    [[ ${1:-} == --dry-run ]] && DRY_RUN=1

    # A value of a top-level key of a config, or of a `slurm:` key (those are indented two spaces).
    config_key() { awk -v key="$2:" '$1 == key { print $2 }' "$1"; }

    lane_last=()   # per lane, the id of the job submitted last
    count=0
    for k in $REPS; do
        config="configs/cln-fiber-amber14-implicit-330K-rep${k}.yaml"
        [[ -f $config ]] || { echo "error: no such config: $config" >&2; exit 2; }
        job="$(config_key "$config" job_name)"
        time="$(config_key "$config" time)"

        lane=$(( count % MAX_PARALLEL ))
        count=$(( count + 1 ))
        cmd=(sbatch --parsable "--job-name=${job}" "--time=${time}" "--export=ALL,CONFIG=${config}")
        [[ -n ${lane_last[$lane]:-} ]] && cmd+=("--dependency=afterany:${lane_last[$lane]}")
        cmd+=("$SELF")
        if (( DRY_RUN )); then
            id="<jobid:rep${k}>"
            echo "${cmd[*]}   # lane ${lane} -> ${id}"
        else
            id="$("${cmd[@]}")"
            echo "rep${k}: job ${id} (lane ${lane})"
        fi
        lane_last[$lane]="$id"
    done
    exit 0
fi

# ---- job mode ------------------------------------------------------------------------------
# Run from the directory the job was submitted from (the project root).
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

# The plain openmm wheel ships Reference/CPU/OpenCL only, so without the `openmm-cuda` extra a
# --gres=gpu:1 job would quietly integrate on the CPU at a fraction of the speed.
python - <<'EOF' || exit 1
import sys
import openmm

names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print(f"OpenMM {openmm.version.version} platforms: {', '.join(names)}")
if "CUDA" not in names:
    sys.exit("error: no CUDA platform; run `uv sync --extra openmm-cuda`")
EOF

echo "=== ${CONFIG} ==="
md-sim run "$CONFIG"

echo "Finished ${CONFIG}."
