#!/bin/bash
#SBATCH --job-name=equilibrate-h-bonds
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-24:00:00
#SBATCH --partition=gshort
#SBATCH --mem=2G
#SBATCH --output=./slurm/output/%x-%A_%a.out
#SBATCH --error=./slurm/output/%x-%A_%a.err

# Equilibrate the bonds an existing trajectory had constrained, by integrating the free end of
# each of them in an unconstrained potential while every other atom is frozen at zero mass (see
# src/md_simulations/sampling/equilibrate_h_bonds.py for why this is the right conditional and why
# the analytic draw of md-sim-resample-hbonds is not quite it). System-agnostic: the potential
# config and the trajectory come in as arguments.
#
#   sbatch slurm/equilibrate-h-bonds.sh <config> <trajectory> <topology>
#
# The config must set `constraints: none`: it is both the only build carrying the bond terms the
# freed ends are to be scored by, and the only one OpenMM will freeze atoms in -- a constraint
# that touches a massless particle is refused at context creation. It must otherwise be the *same*
# potential the output will later be scored against.
#
# Every knob is an env-overridable with a default, so re-staging a run needs no edit:
#
#   EQUIL_TIME      ps of dynamics per frame                       (default 2.0)
#   FRICTION        Langevin friction, 1/ps; empty = the config's  (default empty)
#   TIMESTEP        ps; must resolve the now-unconstrained bond    (default 0.0005)
#   CONSTRAINED_AS  scheme the input was produced with             (default hbonds)
#   STRIDE          keep every STRIDE-th input frame               (default 1, i.e. all)
#   START           first input frame the array covers             (default 0)
#   STOP            stop before this input frame; empty = the end  (default empty)
#   RANGES_FILE     one explicit `<start> <stop>` per task; resume  (default empty; see below)
#   CHUNK           frames per read/write flush                    (default 2000; see below)
#   PLATFORM        OpenMM platform                                (default CPU)
#   OUT_DIR         where the shards land      (default: alongside the input trajectory)
#   OUT_PREFIX      shard basename            (default: <trajectory name>_hequilibrated)
#   OUT_EXT         shard format, any mdtraj writes                (default trr; see below)
#   SEED            base for the velocity draws                    (default 42)
#   BEAMTIME_GROUP  group the runs land in; empty = the config's   (default empty)
#   NO_BEAMTIME     set to 1 to not register the job at all        (default unset)
#
# Beamtime gets one run per shard, named after the shard's output file, so a 16-task array is 16
# runs in one group and a requeued task continues only its own. Each logs the running X-H width
# (overall and per element pair: N-H, C-H, O-H separately, since those are the bonds whose width
# the independent-harmonic draw gets wrong) and the mean potential energy before and after, which
# is the number that says what the equilibration bought.
#
# EQUIL_TIME and FRICTION are the two numbers to measure rather than assume, and they trade off
# against each other. Convergence is a *plateau* in EQUIL_TIME -- not agreement with the analytic
# sqrt(kT/k), which is the thing being corrected -- and raising FRICTION buys that plateau at
# fewer steps while biasing nothing, since the stationary distribution of Langevin dynamics does
# not depend on the friction. Measure both on a few hundred frames before committing to a full
# run. Too high a friction overdamps the hydroxyl and methyl rotations instead, so it is scanned,
# not maximised.
#
# Why CPU and no GPU by default: these are single molecules of order 10^2 atoms, where an OpenMM
# GPU kernel launch costs more than the step it performs and one card sits at a few percent
# utilisation. Equilibration is embarrassingly parallel over frames instead, so N few-core CPU
# tasks beat one A30 and leave the GPU queues free. `--cpus-per-task=4` is deliberate and
# measured: on chignolin (138 atoms, GBn2) the CPU platform runs 414 steps/s on one thread, 1124
# on four and *less* again when it takes all 20 cores of a node, so four is where the curve turns
# over. OPENMM_CPU_THREADS is pinned to it below, since OpenMM otherwise helps itself to every
# core on the node regardless of what slurm allocated.
#
# Sharding is by *contiguous block*, one per array task, not by stride. The shards are meant to be
# concatenated back into one trajectory, and only contiguous blocks concatenate in shard order
# into the input's own frame order -- which is what every per-frame array derived from the same
# trajectory (energies, forces, scores, held-out frame indices) is indexed by. The task does not
# need to be told the trajectory length; `--shard i --n-shards n` reads it and takes block i. A
# failed or requeued task therefore costs only its own block, and the velocity draws are keyed to
# the *absolute* frame index, so a rerun of one shard reproduces exactly the frames it owned.
#
# START and STOP narrow what the array covers; the shards then partition *that* selection, so
# `--shard i --n-shards n` still means block i of it. RANGES_FILE is the resume case, where the
# blocks are no longer a partition of anything: it names a file with one `<start> <stop>` per
# line (blank lines and `#` comments ignored), line TASK+1 being this task's block, taken whole
# rather than partitioned again. That is what an array killed by the wall clock leaves behind --
# one hole per task, at the *end* of each task's block, which no `--shard/--n-shards` split can
# express. Give such a run its own OUT_PREFIX: the output is still named after the task (`_s00`,
# `_s01`, ...), and would otherwise overwrite the shards it is meant to complete.
#
# Gap frames are *not* bit-identical to what the interrupted task would have written, and this is
# worth understanding before trusting the joined dataset. The velocity draw per frame is keyed to
# the absolute frame index and does match, but the Langevin thermostat's own random stream is
# seeded once per *process* and runs on across frames, so a task that starts at frame 9995 meets
# that frame with the stream at step 0 rather than 9995 frames in. Measured on chignolin frames
# 9990-9991 re-run under the production settings: the 77 frozen heavy atoms came back
# bit-identical, the 61 hydrogens did not (up to 0.17 nm apart, i.e. a methyl landing in a
# different rotamer). Nothing is biased by this -- every frame is an independent 1 ps run from
# the same frozen heavy-atom configuration with a thermal velocity draw, so a different stream
# offset gives a different draw from the *same* p(x_H | x_heavy) -- but the gap frames are
# another valid realisation, not a continuation. Only a rerun of a whole shard from its own first
# frame reproduces that shard exactly.
#
# CHUNK is the flush granularity, and it is what a wall-clock kill costs: frames of the chunk in
# flight have been integrated but not yet written, and they are lost. Measured on the chignolin
# resume: at CHUNK 2000 the killed tasks each threw away up to ~1800 finished frames. Lower it
# (500 is a good default for runs expected to be cut off) and the loss scales with it; the extra
# writes are negligible against the integration.
#
# Concatenate afterwards, in shard order (mdtraj's join, or `gmx trjcat -cat` for trr):
#
#   python -c "import mdtraj as md, glob; \
#       t = md.join([md.load(p, top=TOP) for p in sorted(glob.glob('PREFIX_s*.trr'))]); \
#       t.save('PREFIX.trr')"
#
# Sizing. Do not size from the 1124 steps/s single-task benchmark above: that is the speed of one
# task with a node to itself, and a full array does not get one. Measured on the 196 000-frame
# chignolin reference over 16 tasks at EQUIL_TIME 1.0 (2000 steps per frame): 0.100 to 0.137
# frames per second per task, i.e. 200-274 steps/s, a factor ~4 below the benchmark, and slowest
# on the tasks that landed last and shared their nodes most. At that rate 12 250 frames per task
# needs ~25-34 h, so the 24 h wall above cut every one of the 16 tasks off at 78 % of the
# trajectory. Size from ~0.10 frames/s per task, leave headroom, and scale --array and --time
# together: 32 tasks at 24 h, or 16 at 48 h, cover that reference with room to spare.
#
# The shards are written as .trr: lossless float32, and unlike .xtc it does not quantize to a
# 1e-3 nm grid -- 0.41 pm, which is a sixth of the thermal width being created here and the very
# grid that made the constrained bonds look merely narrow rather than frozen. Prefer it over .dcd
# on /remote too: mdtraj's DCD writer rewrites the header after every frame, which on NFS is
# ~160x slower (measured: 16 s against 0.1 s for 20k ala2 frames; trr and xtc are unaffected).

set -euo pipefail

# Slurm already starts a job in its --chdir, or in the submit directory when none was given, so
# there is nothing to cd to. The `cd "${SLURM_SUBMIT_DIR:-.}"` that used to be here *overrode*
# --chdir: jobs 76206-76589 were submitted with --chdir at this checkout from a shell sitting in
# /remote/gpu04/hummerich/projects/split-flows-public, landed there instead, sourced that
# project's .venv and ran its non-editable md_simulations -- which has no --resume. Nothing in
# the error said so, because that venv has a perfectly good `md-sim` of its own.
cd "${REPO_ROOT:-$PWD}"
if [[ ! -f ./src/md_simulations/cli.py || ! -x ./.venv/bin/python ]]; then
    echo "error: this is not an md-simulations checkout: $PWD" >&2
    echo "       submit with --chdir=<checkout>, or --export=ALL,REPO_ROOT=<checkout>" >&2
    exit 2
fi

if [[ $# -lt 3 ]]; then
    echo "usage: sbatch $0 <config> <trajectory> <topology>" >&2
    echo '  the config must set `constraints: none`; see the header for the env knobs.' >&2
    exit 2
fi

CONFIG="$1"
TRAJECTORY="$2"
TOPOLOGY="$3"

# Validated first, before the venv and before anything is built: an unset variable in the submit
# line -- a `$SF` that was set in a different shell, say -- otherwise reaches the module as an
# empty argument and surfaces there as an mdtraj complaint about an unsupported "" format, which
# names neither the argument nor this script.
for arg in CONFIG TRAJECTORY TOPOLOGY; do
    path="${!arg}"
    if [[ -z $path ]]; then
        echo "error: $arg is empty -- check for an unset variable in the submit line" >&2
        exit 2
    fi
    if [[ ! -f $path ]]; then
        echo "error: $arg does not exist: $path" >&2
        echo "       (relative paths resolve against $PWD)" >&2
        exit 2
    fi
done

mkdir -p ./slurm/output

source .venv/bin/activate

export MD_DATA_ROOT="${MD_DATA_ROOT:-$PWD/data}"
# OpenMM's CPU platform otherwise spawns one thread per core on the node, not per allocated cpu.
export OPENMM_CPU_THREADS="${OPENMM_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

EQUIL_TIME="${EQUIL_TIME:-2.0}"
FRICTION="${FRICTION:-}"
TIMESTEP="${TIMESTEP:-0.0005}"
CONSTRAINED_AS="${CONSTRAINED_AS:-hbonds}"
STRIDE="${STRIDE:-1}"
START="${START:-0}"
STOP="${STOP:-}"
CHUNK="${CHUNK:-2000}"
RANGES_FILE="${RANGES_FILE:-}"
PLATFORM="${PLATFORM:-CPU}"
SEED="${SEED:-42}"
OUT_EXT="${OUT_EXT:-trr}"
BEAMTIME_GROUP="${BEAMTIME_GROUP:-}"
# Default the output next to the input, named after it, so two systems cannot collide.
OUT_DIR="${OUT_DIR:-$(dirname "$TRAJECTORY")}"
OUT_PREFIX="${OUT_PREFIX:-$(basename "${TRAJECTORY%.*}")_hequilibrated}"

N_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
TAG="$(printf 's%02d' "$TASK")"
SHARD_OUT="${OUT_DIR}/${OUT_PREFIX}_${TAG}.${OUT_EXT}"

# Normally this task is block TASK of N_TASKS within [START, STOP). A resume array instead reads
# its block verbatim from RANGES_FILE and takes it whole -- partitioning it again would make the
# numbers in the file mean something other than what they say.
SHARD_ARGS=(--shard "$TASK" --n-shards "$N_TASKS")
if [[ -n $RANGES_FILE ]]; then
    if [[ ! -f $RANGES_FILE ]]; then
        echo "error: RANGES_FILE does not exist: $RANGES_FILE" >&2
        echo "       (relative paths resolve against $PWD)" >&2
        exit 2
    fi
    # Comments and blank lines are dropped before indexing, so the file can carry its own
    # provenance without the line numbers drifting away from the task numbers.
    mapfile -t RANGES < <(grep -vE '^[[:space:]]*(#|$)' "$RANGES_FILE")
    if (( TASK >= ${#RANGES[@]} )); then
        echo "error: RANGES_FILE holds ${#RANGES[@]} ranges, nothing for task ${TASK}" >&2
        echo "       (--array and the file have to agree; see $RANGES_FILE)" >&2
        exit 2
    fi
    read -r START STOP _ <<<"${RANGES[$TASK]}"
    if [[ -z ${STOP:-} ]]; then
        echo "error: ${RANGES_FILE} entry ${TASK} is not '<start> <stop>': ${RANGES[$TASK]}" >&2
        exit 2
    fi
    SHARD_ARGS=(--shard 0 --n-shards 1)
    N_TASKS="${#RANGES[@]}"
fi

mkdir -p "$OUT_DIR"

echo "=== shard ${TAG} of ${N_TASKS} ==="
echo "    frames     = [${START}, ${STOP:-<end>})${RANGES_FILE:+ from $RANGES_FILE}"
echo "    config     = ${CONFIG} (constrained as ${CONSTRAINED_AS})"
echo "    trajectory = ${TRAJECTORY}"
echo "    out        = ${SHARD_OUT}"
echo "    time       = ${EQUIL_TIME} ps, timestep = ${TIMESTEP} ps, platform = ${PLATFORM}"
echo "    friction   = ${FRICTION:-<config>}, threads = ${OPENMM_CPU_THREADS}"
echo "    beamtime   = ${BEAMTIME_GROUP:-<config>}${NO_BEAMTIME:+ (disabled)}"

# Invoked as a module, not through the `md-sim-equilibrate-hbonds` console script. Console-script
# wrappers are written at install time, so an editable checkout that gains a new entry point does
# not get one until it is reinstalled -- and `uv sync` here is not a safe way to do that: the venv
# carries the cgschnet/awsem/dev extras, which a plain sync strips (measured: 206 packages
# uninstalled, 1 installed). The module path works off the editable install with no reinstall.
python -m md_simulations.sampling.equilibrate_h_bonds "$CONFIG" \
    --trajectory "$TRAJECTORY" \
    --topology "$TOPOLOGY" \
    --out "$SHARD_OUT" \
    --constrained-as "$CONSTRAINED_AS" \
    "${SHARD_ARGS[@]}" \
    --stride "$STRIDE" \
    --start "$START" \
    --chunk "$CHUNK" \
    ${STOP:+--stop "$STOP"} \
    --time "$EQUIL_TIME" \
    --timestep "$TIMESTEP" \
    --platform "$PLATFORM" \
    --seed "$SEED" \
    ${FRICTION:+--friction "$FRICTION"} \
    ${BEAMTIME_GROUP:+--beamtime-group "$BEAMTIME_GROUP"} \
    ${NO_BEAMTIME:+--no-beamtime}

echo "Finished shard ${TAG}."
