#!/bin/bash
#SBATCH --job-name=relax-unconstrained
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-08:00:00
#SBATCH --partition=gshort
#SBATCH --mem=4G
#SBATCH --output=./slurm/output/%x-%A_%a.out
#SBATCH --error=./slurm/output/%x-%A_%a.err

# Re-thermalise the bonds an existing trajectory had constrained, by relaxing every frame briefly
# in an unconstrained potential (see src/md_simulations/sampling/relax.py for why this is needed at
# all). System-agnostic: it takes the potential config and the trajectory to relax as arguments.
#
#   sbatch slurm/relax-unconstrained.sh <config> <trajectory> <topology>
#
# The config must set `constraints: none` (or `allbonds`); relax.py refuses a constrained system,
# since relaxing under the constraint leaves the constrained bond lengths exactly where they were.
# It must otherwise be the *same* potential the output will later be scored against.
#
# Every knob is an env-overridable with a default, so re-staging a run needs no edit:
#
#   RELAX_TIME   ps of dynamics per frame                     (default 2.0)
#   TIMESTEP     ps; must resolve the now-unconstrained bond  (default 0.0005)
#   STRIDE       keep every STRIDE-th input frame overall      (default 5)
#   PLATFORM     OpenMM platform                              (default CPU)
#   OUT_DIR      where the shards land        (default: alongside the input trajectory)
#   OUT_PREFIX   shard basename              (default: <trajectory name>_relaxed)
#   OUT_EXT      shard format, any mdtraj writes           (default xtc; see the note below)
#   SEED_BASE    first RNG seed; task i uses SEED_BASE + i    (default 42)
#
# Why CPU and no GPU by default: these are single molecules of order 10^2 atoms, where an OpenMM
# GPU kernel launch costs more than the step it performs and one card sits at a few percent
# utilisation. Relaxation is embarrassingly parallel over frames instead, so N single-core CPU
# tasks beat one A30 and leave the GPU queues free. Set PLATFORM=CUDA and an appropriate
# --partition/--gres for a system large enough to fill a device.
#
# Each task takes a strided slice offset by its array index, so the slices are disjoint and their
# union is exactly every STRIDE-th frame: task i reads frames i*STRIDE, i*STRIDE + S, ... with
# S = STRIDE * n_tasks. A failed or requeued task therefore costs only its own shard. Concatenate
# the shards afterwards -- the frames are then ordered by (shard, frame), so sort by the stored
# time if the original order matters.
#
# Sizing, worked for a 10 000-frame input over the default 16 tasks: 625 frames per task, 2 ps each
# at 0.5 fs = 4000 steps, so 2.5e6 steps per task. Scale --array and --time together.
#
# The shards are written as .xtc, whose 1e-3 nm grid is itself 0.41 pm of quantization noise -- the
# same grid that made the constrained bonds look merely narrow rather than frozen. Added in
# quadrature to a 2.86 pm thermal width that is a ~1% inflation, worth ~0.2 nats of log-weight
# spread and negligible next to everything else -- but set OUT_EXT=dcd for lossless float32 if you
# would rather not put the widths you just created back through that grid.
#
# RELAX_TIME is the one number to check rather than assume. Too short a burst leaves the bonds
# narrower than kT: a harmonic mode started at its minimum with a thermal velocity has width
# sigma*|sin(omega t)|, so a fraction of a period gives a phase-dependent width averaging
# sigma/sqrt(2). Measure the relaxed per-bond width against the force field's own sqrt(kT/k) before
# committing to a full run. Measured on trp-cage (X-H, period ~11 fs, friction 1.0 ps^-1), as the
# ratio to that analytic width: 0.1 ps -> 0.76, 0.5 -> 0.90, 1.0 -> 0.96, 2.0 -> 1.00. Hence the
# default; a stiffer or better-insulated mode may need longer.

set -euo pipefail

# Run from the directory the job was submitted from (the project root).
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

if [[ $# -lt 3 ]]; then
    echo "usage: sbatch $0 <config> <trajectory> <topology>" >&2
    echo '  the config must set `constraints: none`; see the header for the env knobs.' >&2
    exit 2
fi

CONFIG="$1"
TRAJECTORY="$2"
TOPOLOGY="$3"

RELAX_TIME="${RELAX_TIME:-2.0}"
TIMESTEP="${TIMESTEP:-0.0005}"
STRIDE="${STRIDE:-5}"
PLATFORM="${PLATFORM:-CPU}"
SEED_BASE="${SEED_BASE:-42}"
OUT_EXT="${OUT_EXT:-xtc}"
# Default the output next to the input, named after it, so two systems cannot collide.
OUT_DIR="${OUT_DIR:-$(dirname "$TRAJECTORY")}"
OUT_PREFIX="${OUT_PREFIX:-$(basename "${TRAJECTORY%.*}")_relaxed}"

N_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_STRIDE=$((STRIDE * N_TASKS))
SHARD_START=$((TASK * STRIDE))
TAG="$(printf 's%02d' "$TASK")"
SHARD_OUT="${OUT_DIR}/${OUT_PREFIX}_${TAG}.${OUT_EXT}"

mkdir -p "$OUT_DIR"

echo "=== shard ${TAG}: start=${SHARD_START} stride=${SHARD_STRIDE} (of ${N_TASKS} tasks) ==="
echo "    config     = ${CONFIG}"
echo "    trajectory = ${TRAJECTORY}"
echo "    out        = ${SHARD_OUT}"
echo "    relax_time = ${RELAX_TIME} ps, timestep = ${TIMESTEP} ps, platform = ${PLATFORM}"

# Invoked as a module, not through the `md-sim-relax` console script. Console-script wrappers are
# written at install time, so an editable checkout that gains a new entry point does not get one
# until it is reinstalled -- and `uv sync` here is not a safe way to do that: the venv carries the
# cgschnet/awsem/dev extras, which a plain sync strips (measured: 206 packages uninstalled, 1
# installed). The module path works off the editable install with no reinstall at all.
python -m md_simulations.sampling.relax "$CONFIG" \
    --trajectory "$TRAJECTORY" \
    --topology "$TOPOLOGY" \
    --out "$SHARD_OUT" \
    --start "$SHARD_START" \
    --stride "$SHARD_STRIDE" \
    --relax-time "$RELAX_TIME" \
    --timestep "$TIMESTEP" \
    --platform "$PLATFORM" \
    --seed $((SEED_BASE + TASK))

echo "Finished shard ${TAG}."
