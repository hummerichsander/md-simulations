#!/bin/bash
#SBATCH --job-name=relax-xh-unconstrained
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-08:00:00
#SBATCH --partition=gshort
#SBATCH --mem=4G
#SBATCH --output=./slurm/output/%x-%A_%a.out
#SBATCH --error=./slurm/output/%x-%A_%a.err

# Re-thermalise the X-H bond lengths of an existing all-atom trajectory by relaxing every frame
# briefly in the *unconstrained* potential (see src/md_simulations/sampling/relax.py for why).
#
# No GPU: the systems are 138 (cln) and 284 (trp-cage) atoms, so an OpenMM GPU kernel launch costs
# more than the step it performs and one card would sit at a few percent utilisation. Relaxation is
# embarrassingly parallel over frames instead, so 16 single-core CPU tasks beat one A30 by a wide
# margin and leave the GPU queues free.
#
# Each task takes a strided slice of the input (offset by its array index), so the slices are
# disjoint and a failed or requeued task costs only its own shard. Concatenate the shards afterwards
# -- note the output frames are then ordered by (shard, frame), so sort by the stored time if the
# original order matters.
#
# Budget: 10 000 frames / 16 tasks = 625 frames per task, 2 ps each at 0.5 fs = 4000 steps, so
# 2.5e6 steps per task. Set RELAX_TIME from the width gate before launching a full run: a burst
# that is too short leaves the bonds at sigma/sqrt(2), which is half the problem still present.

set -euo pipefail

# Run from the directory the job was submitted from (the project root).
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"

# Every axis is env-overridable so a re-run needs no edit.
CONFIG="${CONFIG:-configs/trpcage-amber14-implicit-unconstrained-300K.yaml}"
TRAJECTORY="${TRAJECTORY:?set TRAJECTORY to the input trajectory}"
TOPOLOGY="${TOPOLOGY:?set TOPOLOGY to the input topology}"
OUT_DIR="${OUT_DIR:-$MD_DATA_ROOT/relaxed}"
OUT_PREFIX="${OUT_PREFIX:-traj_relaxed}"
RELAX_TIME="${RELAX_TIME:-2.0}"
TIMESTEP="${TIMESTEP:-0.0005}"
# Total subsampling of the input, spread across the array: task i takes frames i, i+S, i+2S, ...
# with S = STRIDE * n_tasks, so the union of all tasks is exactly every STRIDE-th frame.
STRIDE="${STRIDE:-5}"

N_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_STRIDE=$((STRIDE * N_TASKS))
SHARD_START=$((TASK * STRIDE))
TAG="$(printf 's%02d' "$TASK")"

mkdir -p "$OUT_DIR"

echo "=== shard ${TAG}: start=${SHARD_START} stride=${SHARD_STRIDE} (of ${N_TASKS} tasks) ==="
echo "    config=${CONFIG} relax_time=${RELAX_TIME} ps timestep=${TIMESTEP} ps"

# Invoked as a module, not through the `md-sim-relax` console script. Console-script wrappers are
# written at install time, so an editable checkout that gains a new entry point does not get one
# until it is reinstalled -- and `uv sync` here is not a safe way to do that: the venv carries the
# cgschnet/awsem/dev extras, which a plain sync strips (measured: 206 packages uninstalled, 1
# installed). The module path works off the editable install with no reinstall at all.
python -m md_simulations.sampling.relax "$CONFIG" \
    --trajectory "$TRAJECTORY" \
    --topology "$TOPOLOGY" \
    --out "${OUT_DIR}/${OUT_PREFIX}_${TAG}.xtc" \
    --start "$SHARD_START" \
    --stride "$SHARD_STRIDE" \
    --relax-time "$RELAX_TIME" \
    --timestep "$TIMESTEP" \
    --platform CPU \
    --seed $((42 + TASK))

echo "Finished shard ${TAG}."
