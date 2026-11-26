# Splitting the lab half out of md-simulations

`scripts/split-lab-repo.sh` creates the sibling repo `/remote/gpu04/hummerich/projects/simulations`
from this one: configs, hand-written job scripts, the Beamtime store and the workspace descriptor. It
is run by you, outside Claude Code (the sandbox cannot write outside `md-simulations`). It only reads
this checkout, except for its last step, which moves the untracked `beamtime/` store out of it.

The package side is cleaned up afterwards by ticket S4, once you have confirmed the clone.

## Before you run

1. **Commit everything here.** The script refuses to run with uncommitted or untracked changes
   (ignored files excepted).
2. **Pick the branch.** The script clones one branch of this checkout, by default the one checked out.
   It must carry the W2 and P2 work: configs without `slurm:` blocks and the `docs/lab-repo-*` drafts
   (the script checks both). `main` qualifies only after that work is merged; until then run it from
   `split-package-and-lab`, or set `SPLIT_BRANCH`. The new repo's branch is always named `main`.
3. **Empty `md-simulations/data/`.** It is a real directory with ~589 GB in it (Finding 8), and the new
   repo's `data/` is a symlink to `/remote/gpu04/hummerich/data`, so the script refuses while anything
   but `README.md` is in it. Move the contents into the data root first (same filesystem, so `mv` is a
   rename), keeping the relative layout the configs use (`data/pdbs/...`, `<system>/<descriptor>/...`)
   or fixing the configs' paths. Decide how the dirs map: the root already has `pdbs/`, `ala2-ms/` and
   similar names, and the configs' `output_subdir` values (`cln/...`, `alaX/...`) are relative to
   whatever `data_root` resolves to.
4. **Re-point `/remote/gpu04/hummerich/data/cgschnet-data`.** It is a symlink to
   `md-simulations/data`, which is emptied by step 3 and stops existing in S4.
5. **No running jobs.** `beamtime/` is moved while nothing may be writing to it: no `md-sim run`, no
   `beamtime serve`. The moved store keeps its absolute `output_dir` paths into the data root, so the
   store stays readable; if beamtime objects to the new location, move it back and point `run_store`
   in `md-workspace.yaml` at it instead (plan, "Not verified").
6. The target `/remote/gpu04/hummerich/projects/simulations` must not exist.

## Run it

```bash
cd /remote/gpu04/hummerich/projects/md-simulations
bash scripts/split-lab-repo.sh --dry-run    # prints every precondition and every command; exit 1 if a real run would refuse
bash scripts/split-lab-repo.sh
```

`SPLIT_PARENT`, `SPLIT_BRANCH` and `SPLIT_DATA_ROOT` override the defaults; the script header documents
them. It stops before changing anything if a precondition fails. If a later step fails it says so;
remove the half-made `simulations/` and rerun (the source checkout is untouched until the last step).

What it does, in order: clones the branch (`--single-branch`, history kept, `origin` removed);
removes `src/ tests/ examples/ docs/ scripts/ uv.lock CLAUDE.md` and installs `pyproject.toml` and
`README.md` from the drafts (commit 1); writes `md-workspace.yaml` and `sites/hd-gpu-cluster.yaml`;
replaces `data/` with the symlink and swaps the `.gitignore` rule to `/data`; installs `.mcp.json`;
copies three scripts to `slurm/*.sh.example` and deletes 14 one-line scripts; creates `slurm/output/`
(commit 2); moves `beamtime/` across.

## Afterwards

```bash
cd /remote/gpu04/hummerich/projects/simulations
git log --oneline -3 && git status --short        # two new commits, clean tree
ls -l data                                         # -> /remote/gpu04/hummerich/data
ls beamtime                                        # the moved run store
ls slurm | wc -l                                   # 19: 13 orchestration, 3 kept, 3 *.example
uv lock && uv sync --extra cu130 --torch-backend=cu130
.venv/bin/md-sim config ls | head
.venv/bin/md-sim check-job slurm/alaX-amber14-implicit-remd-330K.sh
```

- `uv lock` and `uv sync` run outside Claude Code. The lab `pyproject.toml` takes
  `md-simulations` from the sibling checkout (editable path source); if it fails to resolve,
  compare the `mlcg` pin with this repo's, byte for byte.
- `md-sim check-job` passes once `.venv` exists; before `uv sync` only its `venv` check
  fails.
- `.mcp.json` pins `/remote/gpu04/hummerich/projects/simulations`; restart Claude Code there to pick up
  the beamtime and `md-sim mcp` servers.

Publish the repo (the script never touches GitHub):

```bash
cd /remote/gpu04/hummerich/projects/simulations
gh repo create hummerichsander/simulations --private --source=. --remote=origin
git push -u origin main
```

## Left behind in md-simulations

Untracked, ignored files stay here and are yours to move or delete: `slurm/output/` (job logs),
`configs/generated/` (per-system configs rendered by the h-bond dispatchers), `.mcp.json`. Two
orchestration scripts end with a hint naming a deleted one-liner (`slurm/ala2-cgschnet-330K.sh` in
`ala2-cgschnet-scratch-330K.sh`, `slurm/alaX-cgschnet-330K.sh` in `alaX-cgschnet-scratch-330K.sh`);
the hint is text only, fix it by hand.

## Then

1. Tell the planning session the clone exists; S4 (delete the lab half from the package) runs next. S4
   removes `.mcp.json` from this checkout, which is untracked, so use `rm`, not `git rm`, for it.
2. Paste `docs/workspace-md-draft.md` into `claude-instructions/workspace.md` (you edit that file).
3. S4 deletes the `docs/lab-repo-*` and workspace drafts once they have landed. D2 adds the lab repo's
   `CLAUDE.md` and job skill as further drafts; the script does not place them, copy them in by hand
   (it removes the package's `CLAUDE.md` from the clone, so the lab repo has none until then).
