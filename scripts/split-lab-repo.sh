#!/usr/bin/env bash
# Split the lab half of md-simulations into the sibling repo `simulations`.
#
# Clone-then-delete, the pattern claude-instructions/workspace.md records for descriptor-fields:
# a full clone of one branch (history kept), then the package half is removed from the clone.
# This script only ever *reads* the md-simulations checkout, with one exception: the last step
# moves the untracked `beamtime/` run store out of it. The package side is cleaned up afterwards
# (ticket S4), once you have confirmed the clone.
#
#   bash scripts/split-lab-repo.sh --dry-run    # print every command, run nothing
#   bash scripts/split-lab-repo.sh              # do it
#
# Environment (all optional):
#   SPLIT_PARENT      directory that gets the new repo as <parent>/simulations
#                     (default /remote/gpu04/hummerich/projects)
#   SPLIT_BRANCH      branch of this checkout to clone (default: the branch checked out here).
#                     It must carry the W2/P2 work (configs without `slurm:` blocks and the
#                     docs/lab-repo-* drafts); `main` only qualifies once that is merged.
#   SPLIT_DATA_ROOT   the shared data root the new repo's data/ links to
#                     (default /remote/gpu04/hummerich/data)
#
# See docs/split-runbook.md for what to do before and after.

# -E so the ERR trap below also fires for a command that fails inside `run` or `emit`:
# without errtrace a shell function does not inherit it, and the split would die silently.
set -Eeuo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run | -n) DRY_RUN=1 ;;
        -h | --help)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown argument: $arg (only --dry-run is accepted)" >&2
            exit 2
            ;;
    esac
done

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT_PARENT="${SPLIT_PARENT:-/remote/gpu04/hummerich/projects}"
DATA_ROOT="${SPLIT_DATA_ROOT:-/remote/gpu04/hummerich/data}"
DEST="$SPLIT_PARENT/simulations"
GH_OWNER="hummerichsander"

# The 3 trivial scripts kept as templates, and the 14 other trivial scripts (one `md-sim run`
# and nothing else) that are deleted. The 13 orchestration scripts are not named: they stay.
TEMPLATES=(
    ala2-amber14-implicit-330K
    cln-cgschnet-330K
    trpcage-amber14-implicit-seeded-330K
)
DELETE_SCRIPTS=(
    ala2-cgschnet-330K
    alaX-amber14-implicit-330K
    alaX-cgschnet-330K
    cln-amber14-implicit-330K
    cln-amber14-implicit-330K-replicas
    cln-amber14-implicit-obc2-330K
    cln-amber14-implicit-stabilized-300K
    cln-amber14-implicit-stabilized-315K
    cln-amber14-implicit-stabilized-330K
    cln-awsem-330K
    cln-awsem-stabilized-330K
    trpcage-amber14-implicit-300K
    trpcage-cgschnet-300K
    trpcage-cgschnet-finetuned-330K
)

# ---- helpers ---------------------------------------------------------------------------------

say() { printf '\n== %s\n' "$*"; }

# Echo a command, then run it unless this is a dry run.
run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

# Write stdin to a file in the new repo. Dry run shows the content instead.
emit() {
    local dest="$1"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "+ write $dest:"
        sed 's/^/    | /'
    else
        echo "+ write $dest"
        cat >"$dest"
    fi
}

FAILED=0
precondition() { # <description> <command...>
    local what="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "  ok    $what"
    else
        echo "  FAIL  $what"
        FAILED=$((FAILED + 1))
    fi
}

# A draft from the source branch, exactly as committed there (not the working tree).
draft() { git -C "$SRC" show "$BRANCH:$1"; }

# Drop the "delete these lines when pasting" preamble from the two drafts.
strip_readme_preamble() {
    awk '
        /^Draft README for the lab repo/ { skip = 1 }
        skip { if (/Delete this line when pasting\./) { skip = 0; drop_blank = 1 } next }
        drop_blank && /^$/ { drop_blank = 0; next }
        { print }'
}
strip_pyproject_preamble() {
    awk '
        !done { if (/Delete these three comment lines/) { done = 1; drop_hash = 1 } next }
        drop_hash { drop_hash = 0; if ($0 == "#") next }
        { print }'
}

# ---- preconditions ---------------------------------------------------------------------------

BRANCH="${SPLIT_BRANCH:-$(git -C "$SRC" symbolic-ref --short -q HEAD || true)}"

echo "source:       $SRC"
echo "branch:       ${BRANCH:-<none>}"
echo "destination:  $DEST"
echo "data root:    $DATA_ROOT"
[ "$DRY_RUN" -eq 1 ] && echo "mode:         DRY RUN (nothing is executed)"

say "Preconditions"
precondition "$SRC is a git checkout" git -C "$SRC" rev-parse --git-dir
precondition "a branch to clone is known (SPLIT_BRANCH, or not a detached HEAD)" test -n "$BRANCH"
if [ -n "$BRANCH" ]; then
    precondition "branch '$BRANCH' exists" git -C "$SRC" rev-parse --verify -q "refs/heads/$BRANCH"
    precondition "branch carries docs/lab-repo-pyproject.toml, lab-repo-mcp.json and lab-repo-README.md" \
        bash -c 'for f in pyproject.toml mcp.json README.md; do
                     git -C "$1" cat-file -e "$2:docs/lab-repo-$f" || exit 1; done' _ "$SRC" "$BRANCH"
    precondition "branch has no 'slurm:' block left in configs/ (W2 is on it)" \
        bash -c '! git -C "$1" grep -q "^slurm:" "$2" -- configs' _ "$SRC" "$BRANCH"
    if [ "$BRANCH" != main ]; then
        echo "  note  cloning '$BRANCH', not main; the new repo's branch is renamed to main"
    fi
    for name in "${TEMPLATES[@]}" "${DELETE_SCRIPTS[@]}"; do
        precondition "slurm/$name.sh is on the branch" git -C "$SRC" cat-file -e "$BRANCH:slurm/$name.sh"
    done
fi
precondition "md-simulations has no uncommitted or untracked changes (ignored files excepted)" \
    test -z "$(git -C "$SRC" status --porcelain 2>/dev/null)"
precondition "$DEST does not exist" bash -c '! [ -e "$1" ] && ! [ -L "$1" ]' _ "$DEST"
precondition "data root $DATA_ROOT is a directory" test -d "$DATA_ROOT"
precondition "no md-simulations/data/ contents to lose (empty, README.md only, or a symlink)" \
    bash -c '[ -L "$1/data" ] || [ ! -e "$1/data" ] || \
             [ -z "$(find "$1/data" -mindepth 1 -maxdepth 1 ! -name README.md ! -name .claude)" ]' _ "$SRC"
if [ -d "$SRC/beamtime" ]; then
    precondition "$DEST/beamtime can be created by the move" bash -c '! [ -e "$1/beamtime" ]' _ "$DEST"
else
    echo "  note  $SRC/beamtime does not exist, so no run store is moved"
fi

if [ "$FAILED" -ne 0 ]; then
    echo
    if [ -d "$SRC/data" ] && [ ! -L "$SRC/data" ] &&
        [ -n "$(find "$SRC/data" -mindepth 1 -maxdepth 1 ! -name README.md ! -name .claude 2>/dev/null)" ]; then
        echo "md-simulations/data is a real directory with contents. The new repo's data/ is a symlink to"
        echo "$DATA_ROOT, so move the contents there first (docs/split-runbook.md, 'Before you run'):"
        find "$SRC/data" -mindepth 1 -maxdepth 1 ! -name README.md ! -name .claude | sort | sed 's/^/    /'
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "dry run: $FAILED precondition(s) failed; a real run would stop here. The plan below is"
        echo "what it would do once they pass."
    else
        echo "error: $FAILED precondition(s) failed; nothing was changed." >&2
        exit 1
    fi
fi

on_error() {
    echo >&2
    echo "error: the split stopped part-way. $DEST is incomplete." >&2
    echo "Recover by hand, in this order:" >&2
    echo "  1. If $DEST/beamtime exists, the run store was already moved out of md-simulations:" >&2
    echo "     move it back ('mv $DEST/beamtime $SRC/beamtime') before anything else." >&2
    echo "  2. Then remove the half-made clone ('rm -rf $DEST') and rerun." >&2
    echo "Nothing else in md-simulations was touched; beamtime moves in the very last step." >&2
}
trap on_error ERR

# ---- the split -------------------------------------------------------------------------------

say "1. Clone branch $BRANCH (history kept) into $DEST"
run mkdir -p "$SPLIT_PARENT"
run git clone --quiet --no-local --single-branch --branch "$BRANCH" "$SRC" "$DEST"
run git -C "$DEST" remote remove origin
[ "$BRANCH" = main ] || run git -C "$DEST" branch -m "$BRANCH" main

say "2. Remove the package half and install the lab pyproject.toml and README"
run git -C "$DEST" rm -r -q --ignore-unmatch src tests examples docs scripts uv.lock CLAUDE.md
if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ draft docs/lab-repo-pyproject.toml | strip preamble > $DEST/pyproject.toml"
    echo "+ draft docs/lab-repo-README.md      | strip preamble > $DEST/README.md"
else
    draft docs/lab-repo-pyproject.toml | strip_pyproject_preamble >"$DEST/pyproject.toml"
    draft docs/lab-repo-README.md | strip_readme_preamble >"$DEST/README.md"
    if grep -q 'Delete th' "$DEST/pyproject.toml" "$DEST/README.md"; then
        echo "error: a draft preamble survived the strip" >&2
        exit 1
    fi
fi
run git -C "$DEST" add pyproject.toml README.md
run git -C "$DEST" commit --quiet -m "Split from md-simulations: drop the package half

The package (src/, tests/, examples/, docs/, uv.lock) stays in md-simulations. This repo keeps
configs/, slurm/ and the Beamtime log, and installs the package as a dependency."

say "3. Workspace descriptor and site profile"
run mkdir -p "$DEST/sites"
emit "$DEST/md-workspace.yaml" <<'EOF'
config_dir: configs
data_root: ${MD_DATA_ROOT:-./data}
run_store: beamtime
site: hd-gpu-cluster
EOF
emit "$DEST/sites/hd-gpu-cluster.yaml" <<'EOF'
# The cluster this repo's jobs run on. `md-sim check-job` reads it; add partitions by hand.
name: hd-gpu-cluster
partitions: [a30]
venv_activate: .venv/bin/activate
slurm_output: ./slurm/output
modules: []
EOF

say "4. data/ becomes a symlink to the shared data root"
run git -C "$DEST" rm -r -q --ignore-unmatch data
run rm -rf "$DEST/data"
run ln -s "$DATA_ROOT" "$DEST/data"
# `data/*` ignores a directory's contents, not a symlink named data.
run sed -i -e '/^data\/\*$/c\/data' -e '/^!data\/README\.md$/d' \
    -e 's/ (but keep the data\/ README)//' "$DEST/.gitignore"
if [ "$DRY_RUN" -eq 0 ] && ! grep -qx '/data' "$DEST/.gitignore"; then
    echo "error: .gitignore edit did not take" >&2
    exit 1
fi

say "5. .mcp.json (beamtime server and md-sim mcp, pinned to $DEST)"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ draft docs/lab-repo-mcp.json > $DEST/.mcp.json"
else
    draft docs/lab-repo-mcp.json >"$DEST/.mcp.json"
fi
echo "  (.mcp.json is git-ignored, as it was in md-simulations)"
if [ "$DEST" != /remote/gpu04/hummerich/projects/simulations ]; then
    echo "  note: the draft pins /remote/gpu04/hummerich/projects/simulations; edit .mcp.json if the"
    echo "        repo lives elsewhere (only expected for a test run)"
fi

say "6. Template examples, then prune the trivial scripts"
for name in "${TEMPLATES[@]}"; do
    run cp "$DEST/slurm/$name.sh" "$DEST/slurm/$name.sh.example"
done
for name in "${DELETE_SCRIPTS[@]}"; do
    run git -C "$DEST" rm -q "slurm/$name.sh"
done
# slurm/output is git-ignored, so the clone lacks it, and slurm does not create it.
run mkdir -p "$DEST/slurm/output"
run git -C "$DEST" add -A
run git -C "$DEST" commit --quiet -m "Add the workspace descriptor and site profile; keep three job templates

md-workspace.yaml and sites/hd-gpu-cluster.yaml tell the md-sim CLI where configs, data and the
run store are. data/ is now the git-ignored symlink to the shared data root. Three trivial job
scripts are kept as slurm/*.sh.example templates and the other 14 one-liners are removed; the 13
orchestration scripts stay."

say "7. Move the untracked Beamtime run store (last step; the only change to md-simulations)"
if [ -d "$SRC/beamtime" ]; then
    # Refuse rather than let mv nest the store as $DEST/beamtime/beamtime.
    if [ "$DRY_RUN" -eq 0 ] && [ -e "$DEST/beamtime" ]; then
        echo "error: $DEST/beamtime already exists; not moving the run store onto it." >&2
        exit 1
    fi
    run mv "$SRC/beamtime" "$DEST/beamtime"
else
    echo "(nothing to move)"
fi

say "Done: summary"
if [ "$DRY_RUN" -eq 0 ]; then
    git -C "$DEST" log --oneline -3
    git -C "$DEST" status --short
    ls -la "$DEST"
fi

cat <<EOF

The clone has no remote. After checking it (docs/split-runbook.md), publish it with:

    cd $DEST
    gh repo create $GH_OWNER/simulations --private --source=. --remote=origin
    git push -u origin main

and build its environment yourself, outside Claude Code (its sandbox cannot write the uv cache):

    cd $DEST && uv lock && uv sync --extra cu130 --torch-backend=cu130

EOF
if [ "$DRY_RUN" -eq 1 ]; then
    echo "(dry run: no command above was executed)"
    [ "$FAILED" -eq 0 ] || exit 1
fi
