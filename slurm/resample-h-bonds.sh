#!/bin/bash
#SBATCH --job-name=resample-h-bonds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-01:00:00
#SBATCH --partition=gshort
#SBATCH --mem=8G
#SBATCH --output=./slurm/output/%x-%j.out
#SBATCH --error=./slurm/output/%x-%j.err

# Give an existing trajectory's constrained bond lengths their thermal width by drawing them from
# N(r0, sqrt(kT/k)) rather than by running dynamics (see
# src/md_simulations/sampling/resample_h_bonds.py for why that draw is the right one).
# System-agnostic: the potential config and the trajectory come in as arguments.
#
#   sbatch slurm/resample-h-bonds.sh <config> <trajectory> <topology>
#
# The config must set `constraints: none` -- that is the only build carrying the equilibrium
# lengths and force constants, since OpenMM deletes the bond term of every constrained bond -- and
# must otherwise be the same potential the output will later be scored against. Which bonds to
# redraw is taken from the constrained twin of that same config, so nothing here assumes "the
# hydrogens".
#
# No `--array`, unlike slurm/relax-unconstrained.sh: this is one numpy call over the whole
# trajectory rather than 4000 integrator steps per frame, so it is seconds instead of hours and
# needs neither sharding nor a concatenation step afterwards. The memory request covers holding the
# trajectory twice (input and output). It is fast enough to run directly rather than queue.
#
# Knobs, all env-overridable:
#
#   CONSTRAINED_AS  scheme the input was produced with, hbonds|allbonds  (default hbonds)
#   STRIDE          keep every STRIDE-th input frame                      (default 1, i.e. all)
#   OUT_DIR         where the output lands       (default: alongside the input trajectory)
#   OUT_PREFIX      output basename             (default: <trajectory name>_hresampled)
#   OUT_EXT         output format, anything mdtraj writes            (default xtc; see below)
#   SEED            seed for the draws                                    (default 42)
#
# OUT_EXT matters more here than for a dynamics run: .xtc quantizes to a 1e-3 nm grid, which is
# 0.41 pm of noise -- the very grid that made these bonds look narrow rather than frozen. Against
# the 2.86 pm width being created that is a ~1% inflation, worth about 0.2 nats of log-weight
# spread. Negligible, but `OUT_EXT=dcd` writes lossless float32 and costs nothing.

set -euo pipefail

# Run from the directory the job was submitted from (the project root), so that relative
# arguments mean what they did on the submit line.
cd "${SLURM_SUBMIT_DIR:-.}"

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
# empty argument and surfaces there as an mdtraj complaint about an unsupported "" format.
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

CONSTRAINED_AS="${CONSTRAINED_AS:-hbonds}"
STRIDE="${STRIDE:-1}"
SEED="${SEED:-42}"
OUT_EXT="${OUT_EXT:-xtc}"
# Default the output next to the input, named after it, so two systems cannot collide.
OUT_DIR="${OUT_DIR:-$(dirname "$TRAJECTORY")}"
OUT_PREFIX="${OUT_PREFIX:-$(basename "${TRAJECTORY%.*}")_hresampled}"
OUT="${OUT_DIR}/${OUT_PREFIX}.${OUT_EXT}"

mkdir -p "$OUT_DIR"

echo "=== resampling constrained bonds ==="
echo "    config     = ${CONFIG} (constrained as ${CONSTRAINED_AS})"
echo "    trajectory = ${TRAJECTORY}"
echo "    out        = ${OUT}"
echo "    stride     = ${STRIDE}, seed = ${SEED}"

# Invoked as a module, not through the `md-sim-resample-hbonds` console script. Console-script
# wrappers are written at install time, so an editable checkout that gains a new entry point does
# not get one until it is reinstalled -- and `uv sync` here is not a safe way to do that: the venv
# carries the cgschnet/awsem/dev extras, which a plain sync strips (measured: 206 packages
# uninstalled, 1 installed). The module path works off the editable install with no reinstall.
python -m md_simulations.sampling.resample_h_bonds "$CONFIG" \
    --trajectory "$TRAJECTORY" \
    --topology "$TOPOLOGY" \
    --out "$OUT" \
    --constrained-as "$CONSTRAINED_AS" \
    --stride "$STRIDE" \
    --seed "$SEED"

echo "Finished."
