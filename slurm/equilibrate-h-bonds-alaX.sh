#!/bin/bash
# Equilibrate the X-H bonds of ala-X reference trajectories, by submitting slurm/equilibrate-h-bonds.sh
# once per chain length and run against the matching instance of
# configs/alaX-amber14-implicit-unconstrained-330K.yaml.
#
# The counterpart of slurm/resample-h-bonds-alaX.sh, and deliberately *not* a drop-in replacement:
#
#   resample-h-bonds-alaX.sh  draws each length from N(r0, sqrt(kT/k)) and slides the hydrogen along
#                             its own bond axis. Instant, one trajectory_hresampled.trr per run, but
#                             it treats the bond as an isolated harmonic oscillator. It is not: the
#                             chignolin fiber pilot measured an X-H width 3 % narrower under flexible
#                             dynamics than the independent-harmonic draw predicts.
#   this script               samples p(x_H | x_heavy) in the real unconstrained potential with every
#                             heavy atom at zero mass. The heavy atoms come back bit-identical, so the
#                             slow coordinates are untouched, and the angles, torsions and nonbonded
#                             terms the per-bond draw drops are all in it.
#
# This is a *submit-only* script -- run it directly, do not sbatch it. Each run becomes one array job
# (equilibrate-h-bonds.sh defaults to 16 shards), so unlike the resampler it cannot do its work inside
# a single waiting job.
#
#   ./slurm/equilibrate-h-bonds-alaX.sh --dry-run
#   SYSTEMS="2 5" ./slurm/equilibrate-h-bonds-alaX.sh
#   SYSTEMS=8 REPLICAS= REMD_RUNS="train holdout" ./slurm/equilibrate-h-bonds-alaX.sh
#
# Axes, matching resample-h-bonds-alaX.sh so the two stay interchangeable on the command line:
#
#   SYSTEMS     chain lengths                                  (default "2 3 4 5 6 7 8 9 10 11 12 16 24 32")
#   REPLICAS    plain-MD replicas; "" for none                 (default "0")
#   REMD_RUNS   T-REMD run names (train, holdout, ...)         (default empty)
#   SEED_BASE   base seed; replica k -> +k, train -> +10,      (default SEED, else 42)
#               holdout -> +11, further names -> +12, 13, ...
#   TEMPLATE    the unconstrained config, templated per system (default the alaX one)
#   CONFIG_OUT  where rendered per-system configs are written  (default configs/generated)
#   DEPENDENCY  passed straight to sbatch --dependency         (default empty)
#   ARRAY       sbatch --array override, i.e. the shard count  (default empty = the script's 0-15)
#
# Every knob of equilibrate-h-bonds.sh itself (EQUIL_TIME, STRIDE, TIMESTEP, PLATFORM, CHUNK, ...)
# is exported through untouched -- see its header. STRIDE is the one to reach for first: equilibration
# integrates dynamics per frame, so a 404,000-frame run at STRIDE=1 is a different proposition from
# the resampler's instant pass over the same file.
#
# Unlike the resampler, the rendered config must outlive this script: the array runs later, and reads
# the path it was handed. They are therefore written to CONFIG_OUT (gitignored), one per chain length,
# deterministically named and overwritten on every re-run -- not to a mktemp dir.
#
# Output lands next to each input as <trajectory>_hequilibrated_sNN.trr, one shard per array task.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

SYSTEMS="${SYSTEMS:-2 3 4 5 6 7 8 9 10 11 12 16 24 32}"
REPLICAS="${REPLICAS-0}"
REMD_RUNS="${REMD_RUNS:-}"
SEED_BASE="${SEED_BASE:-${SEED:-42}}"
TEMPLATE="${TEMPLATE:-configs/alaX-amber14-implicit-unconstrained-330K.yaml}"
CONFIG_OUT="${CONFIG_OUT:-configs/generated}"
DEPENDENCY="${DEPENDENCY:-}"
ARRAY="${ARRAY:-}"
DATA_DIR="${MD_DATA_ROOT:-$ROOT/data}/alaX"

DRY_RUN=0
[[ ${1:-} == --dry-run ]] && DRY_RUN=1

[[ -f $TEMPLATE ]] || { echo "error: no such template: $TEMPLATE" >&2; exit 2; }
mkdir -p "$CONFIG_OUT"

# Submit one array for a single run directory. Absolute paths throughout: equilibrate-h-bonds.sh
# resolves relative arguments against SLURM_SUBMIT_DIR, which is wherever the caller happened to be.
submit_one() {
    local n="$1" run_dir="$2" seed="$3" label="$4" config="$5"
    local traj="${run_dir}/trajectory.dcd"
    local top="${run_dir}/initial_structure.pdb"

    if [[ ! -f $traj || ! -f $top ]]; then
        echo "error: ala${n} ${label}: missing ${traj} or ${top}" >&2
        return 1
    fi

    local cmd=(sbatch --parsable "--job-name=equilibrate-h-bonds-ala${n}-${label}")
    [[ -n $DEPENDENCY ]] && cmd+=("--dependency=${DEPENDENCY}")
    [[ -n $ARRAY ]] && cmd+=("--array=${ARRAY}")
    cmd+=("--export=ALL,SEED=${seed}")
    cmd+=("$ROOT/slurm/equilibrate-h-bonds.sh" "$config" "$traj" "$top")

    if (( DRY_RUN )); then
        echo "${cmd[*]}"
    else
        echo "ala${n} ${label} (seed ${seed}): array $("${cmd[@]}")"
    fi
}

for n in $SYSTEMS; do
    # The template is written for ala12; every occurrence names the system.
    config="${ROOT}/${CONFIG_OUT}/ala${n}-amber14-implicit-unconstrained-330K.yaml"
    sed "s/ala12/ala${n}/g" "$TEMPLATE" > "$config"

    for k in $REPLICAS; do
        run="${DATA_DIR}/ala${n}_AMBER14_implicit_330K"
        (( k > 0 )) && run="${run}-r${k}"
        submit_one "$n" "$run" "$(( SEED_BASE + k ))" "r${k}" "$config"
    done

    other=12
    for name in $REMD_RUNS; do
        case $name in
            train) offset=10 ;;
            holdout) offset=11 ;;
            *) offset=$other; other=$(( other + 1 )) ;;
        esac
        submit_one "$n" "${DATA_DIR}/ala${n}_AMBER14_implicit_remd_330K-${name}" \
            "$(( SEED_BASE + offset ))" "remd-${name}" "$config"
    done
done
