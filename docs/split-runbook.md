# Splitting the lab half out of md-simulations

> **Done (2026-11-26), kept as a record.** The split ran and ticket S4 removed the lab half from this
> repo, together with the `docs/lab-repo-*` and `docs/workspace-md-draft.md` drafts the script consumed
> (still in git history before the S4 commit). `scripts/split-lab-repo.sh` therefore cannot run
> again: its preconditions require those drafts. The sections below are what was done and the
> follow-ups that stay yours.

`scripts/split-lab-repo.sh` creates the sibling repo `/remote/gpu04/hummerich/projects/simulations`
from this one: configs, hand-written job scripts, the Beamtime store, the simulation data and the
workspace descriptor. It is run by you, outside Claude Code (the sandbox cannot write outside
`md-simulations`).

It reads this checkout and changes nothing in it except its last two steps, which **move** (rename,
never copy) the untracked `beamtime/` store and the contents of the untracked `data/` into the new
repo. The package side is cleaned up afterwards by ticket S4, once you have confirmed the clone.

The simulation data stays with the lab: the shared root `/remote/gpu04/hummerich/data` holds only
derived datasets (each with a `datacard.yaml`), so nothing is pre-moved there. The new repo's `data/` is
a real, git-ignored directory; `data_root: ${MD_DATA_ROOT:-./data}` in `md-workspace.yaml` resolves to it.

## Before you run

1. **Commit everything here.** The script refuses to run with uncommitted or untracked changes
   (ignored files excepted).
2. **Pick the branch.** The script clones one branch of this checkout, by default the one checked out.
   It must carry the W2 and P2 work: configs without `slurm:` blocks and the `docs/lab-repo-*` drafts
   (the script checks both). `main` qualifies only after that work is merged; until then run it from
   `split-package-and-lab`, or set `SPLIT_BRANCH`. The new repo's branch is always named `main`.
3. **Run the script plain.** The data split is settled: everything in `data/` (`ala2 alaX cln models
   pdbs trp-cage`) goes to the lab repo, `models/` included, and that is the script's default. Set no
   `SPLIT_DATA_*` variable. The script still stops before changing anything if a top-level entry of
   `data/` is not assigned (a new directory appeared since).
4. **No running jobs.** `beamtime/` and `data/` are moved while nothing writes to them: no
   `md-sim run`, no `beamtime serve`, no job in the queue that will start meanwhile. Run records and
   configs hold absolute paths (`.../md-simulations/data/...`, `.../md-simulations/beamtime/...`) from
   before the move; they keep their text, so a finished run's recorded `output_dir` is stale. Check
   whether beamtime objects to the moved store; if so, move it back and set `run_store` in
   `md-workspace.yaml` to point at it (plan, "Not verified").
5. **One filesystem.** The moves must be renames, so `md-simulations/data`, `md-simulations/beamtime`
   and the target `/remote/gpu04/hummerich/projects` have to be on the same filesystem (the script
   checks this): a cross-filesystem `mv` of 589 GB is a copy-then-delete that can fail half-way.
6. The target `/remote/gpu04/hummerich/projects/simulations` must not exist.

## The data split

Contents of `md-simulations/data/` (inspected read-only; sizes from `du`):

| Entry | Size | What it holds | Proposed |
| ----- | ---- | ------------- | -------- |
| `alaX/` | 562 G | one directory per run (`ala<N>_AMBER14_implicit_330K[-rN]`, `..._remd_330K-{pilot,train,holdout}`): trajectories (`dcd`, `trr`), checkpoints (`chk`), `npy`/`csv` energies and forces; also the `*_fm_330K` force-matching datasets and `ala*_CGSchNet_330K` runs | lab |
| `cln/` | 9.4 G | chignolin runs (`AMBER14_implicit_330K_rNN`, `AWSEM_*`, `fiber_*`), `fiber_analysis`, `cln_fm_330K` | lab |
| `trp-cage/` | 13 G | Trp-cage runs (`AMBER14_implicit_seeded_330K_rNN`, scans, `CGSchNet_*`), `analysis_*`, `fm_*`, `configurations.pt` | lab |
| `ala2/` | 4.1 G | the first ala2 run (`AMBER14_implicit_330K`) | lab |
| `pdbs/` | 3 MB | 69 starting and CG-bead structures the configs read (`data/pdbs/...`, 8 config references) | lab (inputs of the configs; the root already has an unrelated `pdbs/`, so it could not go there anyway) |
| `models/` | 1.4 G | `cgschnet/`: trained CGSchNet checkpoints (56 `ckpt`) and the merged model+prior `.pt` files, written by the `*-cgschnet-scratch-*` scripts and read by 18 config references and by split-flows-energetics | lab (confirmed) |

Everything under the four run directories is raw or run-adjacent simulation output, so it belongs to
the lab, and so does `models/`: configs and the scratch-training scripts address it as `data/models/...`
(moving it changes no path), and a dataset in the shared root is not done until it has a
`datacard.yaml`. Derived datasets nested inside the run directories (`alaX/*_fm_330K`, `*/analysis_*`)
stay in place too: the scripts and configs address them by those paths.

(The script can instead send an entry to the shared root with `SPLIT_DATA_SHARED="name=target"`; the
decision is not to, so do not set it.)

## Run it

```bash
cd /remote/gpu04/hummerich/projects/md-simulations
bash scripts/split-lab-repo.sh --dry-run    # prints every precondition and every command; exit 1 if a real run would refuse
bash scripts/split-lab-repo.sh              # plain: no SPLIT_DATA_* variable, the defaults are the decided split
```

`SPLIT_PARENT`, `SPLIT_BRANCH`, `SPLIT_DATA_LAB`, `SPLIT_DATA_SHARED` and `SPLIT_DATA_ROOT` override the
defaults; the script header documents them. All preconditions run before anything changes, and a failed
one stops the script with nothing touched.

What it does, in order: clones the branch (`--single-branch`, history kept, `origin` removed);
removes `src/ tests/ examples/ docs/ scripts/ uv.lock CLAUDE.md`, installs `pyproject.toml`, `README.md`,
`CLAUDE.md` and `.claude/skills/write-job/SKILL.md` from the drafts and narrows `.gitignore`'s `.claude` rule
to `.claude/*` + `!.claude/skills/` so the skill is committed (commit 1); writes `md-workspace.yaml`,
`sites/hd-gpu-cluster.yaml` and a short `data/README.md` of its own;
installs `.mcp.json`; copies three scripts to `slurm/*.sh.example`, deletes 14 one-line scripts and
creates `slurm/output/` (commit 2); then the two moves, last: `beamtime/`, then each `data/` entry.
Commits and clone come first so that a failure there leaves `md-simulations` untouched.

### If it stops part-way

The error note lists exactly the moves already done, as `mv` commands, newest first. **Do not
`rm -rf` the half-made `simulations/` before undoing them**: after the first move the only copy of
that data is inside it. Run the listed commands (checking both ends of a move that itself failed),
and only then remove the clone and rerun. If the note says nothing was moved, `rm -rf` of the clone is
safe.

## Afterwards

```bash
cd /remote/gpu04/hummerich/projects/simulations
git log --oneline -3 && git status --short        # two new commits, clean tree
ls data                                            # the moved entries, plus README.md
ls beamtime                                        # the moved run store
ls slurm | wc -l                                   # 20: 13 orchestration, 3 kept, 3 *.example, output/
uv lock && uv sync --extra cu130
.venv/bin/md-sim config ls | head
.venv/bin/md-sim check-job slurm/alaX-amber14-implicit-remd-330K.sh
```

- **Re-point the shared root's `cgschnet-data` link** (it still targets `md-simulations/data`, which is now
  empty). split-flows-energetics reads `models/` and `pdbs/` through it:
  `ln -sfn /remote/gpu04/hummerich/projects/simulations/data /remote/gpu04/hummerich/data/cgschnet-data`.
  Two places still describe the old target: the shared root's `SCHEMA.md` ("`cgschnet-data` is a symlink
  to md-simulations' own data directory") and `split-flows-energetics/README.md` (`ln -s
  .../md-simulations/data data/cgschnet-data`).
- `uv lock` and `uv sync` run outside Claude Code. The lab `pyproject.toml` takes `md-simulations` from
  the sibling checkout (editable path source); if it fails to resolve, compare the `mlcg` pin with this
  repo's, byte for byte.
- `md-sim check-job` passes once `.venv` exists; before `uv sync` only its `venv` check fails.
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
`configs/generated/` (per-system configs rendered by the h-bond dispatchers), `.mcp.json`, and the
emptied `data/` (its `README.md` stays tracked here until S4; the clone has its own, rewritten one). Two orchestration scripts end with a
hint naming a deleted one-liner (`slurm/ala2-cgschnet-330K.sh` in `ala2-cgschnet-scratch-330K.sh`,
`slurm/alaX-cgschnet-330K.sh` in `alaX-cgschnet-scratch-330K.sh`); the hint is text only, fix it by hand.

## Then

1. Tell the planning session the clone exists; S4 (delete the lab half from the package) runs next. It
   touches only tracked files: `.mcp.json` is untracked here, so it stays as a leftover below.
2. Paste this bullet into the "Workspace" section of `claude-instructions/workspace.md`, replacing the
   existing `md-simulations` bullet (you edit that file; a ticket session does not):

   > - `md-simulations` is the package other projects import: the MD engines, the typed config models,
   >   the `md-sim` command and the torch potentials other projects reweight with. It holds no configs,
   >   job scripts or data. `simulations` holds those: the configs, the hand-written job scripts and the
   >   Beamtime store, split out of `md-simulations` on 2026-11-26 (a full clone with its history, then
   >   each half deleted its other side). It keeps the simulation data in its own git-ignored `data/`
   >   (the shared root holds only derived datasets) and finds its layout through `md-workspace.yaml`.
3. S4 deleted the `docs/lab-repo-*` and workspace drafts; the script had already installed what the lab
   repo needs from them.
