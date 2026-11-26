#!/usr/bin/env bash
# Split the lab half of md-simulations into the sibling repo `simulations`.
#
# Clone-then-delete, the pattern claude-instructions/workspace.md records for descriptor-fields:
# a full clone of one branch (history kept), then the package half is removed from the clone.
# This script only ever *reads* the md-simulations checkout, with one exception: the last two
# steps MOVE (never copy) the untracked `beamtime/` run store and the contents of the untracked
# `data/` out of it, into the clone. The package side is cleaned up afterwards (ticket S4), once
# you have confirmed the clone.
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
#   SPLIT_DATA_LAB    top-level entries of md-simulations/data/ that move into the new repo's own
#                     git-ignored data/ (default: "ala2 alaX cln models pdbs trp-cage")
#   SPLIT_DATA_SHARED entries that instead move into the shared data root, with a symlink left in
#                     the new repo's data/ so config paths still resolve (default: none). Each is
#                     `name` or `name=target-dir-under-the-root`. The root only holds derived
#                     datasets, each needing a datacard.yaml that you write afterwards.
#                     Every entry must be in exactly one of the two lists.
#   SPLIT_DATA_ROOT   the shared data root, used only for SPLIT_DATA_SHARED
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
read -r -a DATA_LAB <<<"${SPLIT_DATA_LAB-ala2 alaX cln models pdbs trp-cage}"
read -r -a DATA_SHARED <<<"${SPLIT_DATA_SHARED-}"
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

# Every move out of md-simulations is irreversible once it happens, so remember how to undo each,
# newest first, for the error note below.
UNDO=()
remember_undo() { UNDO=("$1" ${UNDO[@]+"${UNDO[@]}"}); }
# Move (never copy) a path, and record the command that puts it back. The undo is recorded
# *before* the mv, so a move that fails is listed too and the error note can say "check both
# ends of it"; a `mv` that never happened undoes with a harmless "No such file".
move() { # <from> <to>
    remember_undo "$(printf 'mv %q %q' "$2" "$1")"
    run mv "$1" "$2"
}

# Stop with a message and the recovery note. `exit` does not fire the ERR trap, so every
# deliberate abort after the trap is installed has to go through here or the note is lost --
# and after step 6 the note is the only thing standing between the user and an `rm -rf` of
# the only copy of the data.
die() {
    echo "error: $*" >&2
    on_error
    exit 1
}

# The existing ancestor of a path (a mkdir -p target need not exist yet).
existing_ancestor() {
    local d="$1"
    while [ ! -e "$d" ]; do d="$(dirname "$d")"; done
    printf '%s\n' "$d"
}
same_device() { [ "$(stat -c %d "$1")" = "$(stat -c %d "$(existing_ancestor "$2")")" ]; }
shared_name() { printf '%s\n' "${1%%=*}"; }
shared_target() { case "$1" in *=*) printf '%s\n' "${1#*=}" ;; *) printf '%s\n' "$1" ;; esac; }
# A single directory name: no slash and no `..`, so no entry of SPLIT_DATA_LAB/SPLIT_DATA_SHARED
# can name a path that leaves data/ (or the data root) and get moved.
plain_name() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; }

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

# Drop the "delete these lines when pasting" preamble from the drafts.
strip_readme_preamble() {
    awk '
        /^Draft README for the lab repo/ { skip = 1 }
        skip { if (/Delete this line when pasting\./) { skip = 0; drop_blank = 1 } next }
        drop_blank && /^$/ { drop_blank = 0; next }
        { print }'
}
# CLAUDE.md draft: everything before its own `# simulations` title is preamble.
strip_claude_preamble() { awk '/^# simulations$/ { on = 1 } on { print }'; }
# Skill draft: the file proper starts at the first `---` (the frontmatter fence).
strip_skill_preamble() { awk '/^---$/ { on = 1 } on { print }'; }
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
echo "data -> lab:  ${DATA_LAB[*]:-<nothing>}"
echo "data -> root: ${DATA_SHARED[*]:-<nothing>}${DATA_SHARED[*]:+  (root $DATA_ROOT)}"
[ "$DRY_RUN" -eq 1 ] && echo "mode:         DRY RUN (nothing is executed)"

say "Preconditions"
precondition "$SRC is a git checkout" git -C "$SRC" rev-parse --git-dir
precondition "a branch to clone is known (SPLIT_BRANCH, or not a detached HEAD)" test -n "$BRANCH"
if [ -n "$BRANCH" ]; then
    precondition "branch '$BRANCH' exists" git -C "$SRC" rev-parse --verify -q "refs/heads/$BRANCH"
    precondition "branch carries the docs/lab-repo-* drafts (pyproject.toml, mcp.json, README.md, CLAUDE.md, job-skill.md)" \
        bash -c 'for f in pyproject.toml mcp.json README.md CLAUDE.md job-skill.md; do
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
if [ -L "$SRC/data" ]; then
    precondition "md-simulations/data is a real directory, not a symlink" false
elif [ -d "$SRC/data" ]; then
    # Every top-level entry must be assigned, so nothing is left behind or moved by accident.
    mapfile -t ENTRIES < <(find "$SRC/data" -mindepth 1 -maxdepth 1 ! -name README.md ! -name .claude \
        -printf '%f\n' | sort)
    SHARED_NAMES=()
    for e in ${DATA_SHARED[@]+"${DATA_SHARED[@]}"}; do SHARED_NAMES+=("$(shared_name "$e")"); done
    for e in "${ENTRIES[@]}"; do
        in_lab=0 in_shared=0
        for n in ${DATA_LAB[@]+"${DATA_LAB[@]}"}; do [ "$n" = "$e" ] && in_lab=1; done
        for n in ${SHARED_NAMES[@]+"${SHARED_NAMES[@]}"}; do [ "$n" = "$e" ] && in_shared=1; done
        precondition "data/$e is assigned to exactly one of SPLIT_DATA_LAB, SPLIT_DATA_SHARED" \
            test "$((in_lab + in_shared))" -eq 1
    done
    for n in ${DATA_LAB[@]+"${DATA_LAB[@]}"} ${SHARED_NAMES[@]+"${SHARED_NAMES[@]}"}; do
        precondition "assigned entry '$n' is a plain directory name" plain_name "$n"
        precondition "assigned entry data/$n exists" test -e "$SRC/data/$n"
    done
    precondition "md-simulations/data and $SPLIT_PARENT are on one filesystem (the moves are renames)" \
        same_device "$SRC/data" "$SPLIT_PARENT"
    if [ "${#DATA_SHARED[@]}" -gt 0 ]; then
        precondition "data root $DATA_ROOT is a directory" test -d "$DATA_ROOT"
        # The symlink left in the lab repo's data/ points at $DATA_ROOT/<target>, so a relative
        # root would leave a link that resolves against data/ and dangles.
        precondition "data root $DATA_ROOT is an absolute path" \
            bash -c 'case "$1" in /*) ;; *) exit 1 ;; esac' _ "$DATA_ROOT"
        precondition "md-simulations/data and $DATA_ROOT are on one filesystem" \
            same_device "$SRC/data" "$DATA_ROOT"
        for e in "${DATA_SHARED[@]}"; do
            precondition "shared target '$(shared_target "$e")' is a plain directory name" \
                plain_name "$(shared_target "$e")"
            precondition "$DATA_ROOT/$(shared_target "$e") does not exist" \
                bash -c '! [ -e "$1" ] && ! [ -L "$1" ]' _ "$DATA_ROOT/$(shared_target "$e")"
        done
    fi
else
    echo "  note  $SRC/data does not exist, so no data is moved"
fi
if [ -d "$SRC/beamtime" ]; then
    precondition "$DEST/beamtime can be created by the move" bash -c '! [ -e "$1/beamtime" ]' _ "$DEST"
    precondition "md-simulations/beamtime and $SPLIT_PARENT are on one filesystem" \
        same_device "$SRC/beamtime" "$SPLIT_PARENT"
else
    echo "  note  $SRC/beamtime does not exist, so no run store is moved"
fi

if [ "$FAILED" -ne 0 ]; then
    echo
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
    if [ "${#UNDO[@]}" -eq 0 ]; then
        echo "Nothing has been moved out of md-simulations yet (the moves are the last steps), so" >&2
        echo "'rm -rf $DEST' is safe; then rerun." >&2
    else
        echo "These moves already happened. The only copy of that data is now inside $DEST (or the" >&2
        echo "data root), so do NOT 'rm -rf $DEST' yet. Undo them first, in this order:" >&2
        printf '    %s\n' "${UNDO[@]}" >&2
        echo "If the failing step was itself one of these moves, check both ends of it before" >&2
        echo "running its undo line. Once every line is undone (or you have confirmed the clone" >&2
        echo "holds everything), 'rm -rf $DEST' is safe and you can rerun." >&2
    fi
}
trap on_error ERR

# ---- the split -------------------------------------------------------------------------------

say "1. Clone branch $BRANCH (history kept) into $DEST"
run mkdir -p "$SPLIT_PARENT"
run git clone --quiet --no-local --single-branch --branch "$BRANCH" "$SRC" "$DEST"
run git -C "$DEST" remote remove origin
[ "$BRANCH" = main ] || run git -C "$DEST" branch -m "$BRANCH" main

say "2. Remove the package half; install the lab pyproject.toml, README, CLAUDE.md and write-job skill"
run git -C "$DEST" rm -r -q --ignore-unmatch src tests examples docs scripts uv.lock CLAUDE.md
if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ draft docs/lab-repo-pyproject.toml | strip preamble > $DEST/pyproject.toml"
    echo "+ draft docs/lab-repo-README.md      | strip preamble > $DEST/README.md"
    echo "+ draft docs/lab-repo-CLAUDE.md      | strip preamble > $DEST/CLAUDE.md"
    echo "+ draft docs/lab-repo-job-skill.md   | strip preamble > $DEST/.claude/skills/write-job/SKILL.md"
else
    draft docs/lab-repo-pyproject.toml | strip_pyproject_preamble >"$DEST/pyproject.toml"
    draft docs/lab-repo-README.md | strip_readme_preamble >"$DEST/README.md"
    draft docs/lab-repo-CLAUDE.md | strip_claude_preamble >"$DEST/CLAUDE.md"
    mkdir -p "$DEST/.claude/skills/write-job"
    draft docs/lab-repo-job-skill.md | strip_skill_preamble >"$DEST/.claude/skills/write-job/SKILL.md"
    if grep -q 'Delete th' "$DEST/pyproject.toml" "$DEST/README.md" "$DEST/CLAUDE.md" \
        "$DEST/.claude/skills/write-job/SKILL.md" ||
        [ "$(head -n1 "$DEST/CLAUDE.md")" != "# simulations" ] ||
        [ "$(head -n1 "$DEST/.claude/skills/write-job/SKILL.md")" != "---" ]; then
        die "a draft preamble survived the strip"
    fi
fi
# The package ignores all of .claude. In the lab repo the skill is committed, so ignore only what
# is local to a checkout: everything under .claude/ except skills/ (settings.local.json, the
# harness's .cc-writes droppings, ...).
run sed -i 's|^\.claude$|# Claude Code state, local to each checkout -- except the committed skills. A new\n# tracked path under .claude/ (settings.json, commands/, ...) needs its own "!" line.\n.claude/*\n!.claude/skills/|' "$DEST/.gitignore"
if [ "$DRY_RUN" -eq 0 ] && ! grep -qx '!.claude/skills/' "$DEST/.gitignore"; then
    die "the .gitignore edit for .claude did not take"
fi
run git -C "$DEST" add .gitignore pyproject.toml README.md CLAUDE.md .claude/skills/write-job/SKILL.md
run git -C "$DEST" commit --quiet -m "Split from md-simulations: drop the package half

The package (src/, tests/, examples/, docs/, uv.lock) stays in md-simulations. This repo keeps
configs/, slurm/ and the Beamtime log, and installs the package as a dependency."

say "3. Workspace descriptor, site profile and data/README.md"
run mkdir -p "$DEST/sites" "$DEST/data"
emit "$DEST/data/README.md" <<'EOF'
# Simulation data

This directory is the data root named by `md-workspace.yaml` (`data_root: ${MD_DATA_ROOT:-./data}`,
relative to the repo root). Everything in it except this README is git-ignored: trajectories,
checkpoints, trained models and input structures are never committed.

`md-sim run` resolves a config's input paths and `output_subdir` against the data root, so runs
land in `<system>/<descriptor>/` here. Point a single run elsewhere with `--data-root` or
`MD_DATA_ROOT`.

Datasets derived from these runs for other projects belong in the shared root
`/remote/gpu04/hummerich/data`, each with a `datacard.yaml`, not here.
EOF

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

say "4. .mcp.json (beamtime server and md-sim mcp, pinned to $DEST)"
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

say "5. Template examples, then prune the trivial scripts"
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
run store are. data/ stays the git-ignored directory it was (the data moves in afterwards, not
through git). Three trivial job
scripts are kept as slurm/*.sh.example templates and the other 14 one-liners are removed; the 13
orchestration scripts stay."

say "6. Move (not copy) the untracked Beamtime run store"
if [ -d "$SRC/beamtime" ]; then
    # Refuse rather than let mv nest the store as $DEST/beamtime/beamtime.
    if [ "$DRY_RUN" -eq 0 ] && [ -e "$DEST/beamtime" ]; then
        die "$DEST/beamtime already exists; not moving the run store onto it."
    fi
    move "$SRC/beamtime" "$DEST/beamtime"
else
    echo "(nothing to move)"
fi

say "7. Move (not copy) the simulation data into $DEST/data (last step; both moves are renames)"
if [ -d "$SRC/data" ] && [ ! -L "$SRC/data" ]; then
    run mkdir -p "$DEST/data"
    for e in ${DATA_LAB[@]+"${DATA_LAB[@]}"}; do
        if [ "$DRY_RUN" -eq 0 ] && [ -e "$DEST/data/$e" ]; then
            die "$DEST/data/$e already exists; not moving onto it."
        fi
        move "$SRC/data/$e" "$DEST/data/$e"
    done
    for e in ${DATA_SHARED[@]+"${DATA_SHARED[@]}"}; do
        name="$(shared_name "$e")"
        target="$DATA_ROOT/$(shared_target "$e")"
        move "$SRC/data/$name" "$target"
        run ln -s "$target" "$DEST/data/$name"
        remember_undo "$(printf 'rm %q' "$DEST/data/$name")"
    done
else
    echo "(nothing to move)"
fi

say "Done: summary"
if [ "$DRY_RUN" -eq 0 ]; then
    git -C "$DEST" log --oneline -3
    git -C "$DEST" status --short
    ls -la "$DEST" "$DEST/data"
fi

cat <<EOF

The clone has no remote. After checking it (docs/split-runbook.md), publish it with:

    cd $DEST
    gh repo create $GH_OWNER/simulations --private --source=. --remote=origin
    git push -u origin main

Then re-point $DATA_ROOT/cgschnet-data at $DEST/data (it still targets md-simulations/data):

    ln -sfn $DEST/data $DATA_ROOT/cgschnet-data

and build its environment yourself, outside Claude Code (its sandbox cannot write the uv cache):

    cd $DEST && uv lock && uv sync --extra cu130 --torch-backend=cu130

EOF
if [ "$DRY_RUN" -eq 1 ]; then
    echo "(dry run: no command above was executed)"
    [ "$FAILED" -eq 0 ] || exit 1
fi
