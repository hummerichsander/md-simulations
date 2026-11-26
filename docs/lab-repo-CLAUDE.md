Draft `CLAUDE.md` for the lab repo `/remote/gpu04/hummerich/projects/simulations`, to be copied there
by the user (D2 of `docs/plans/split-package-and-lab.md`; `scripts/split-lab-repo.sh` removes the
package's `CLAUDE.md` from the clone and does not place this one). Delete this paragraph when pasting.
Install the companion pieces too: `docs/lab-repo-job-skill.md` as `.claude/skills/write-job/SKILL.md`,
and the `.gitignore` edit under "Committing the skill" in that file.

# simulations

The lab repo around the `md-simulations` package: one versioned config per simulated system, the
hand-written job scripts that run them on the cluster, and the Beamtime experiment log. The engines,
config models, torch potentials and the `md-sim` command live in the package
(<https://github.com/hummerichsander/md-simulations>); this repo holds what is particular to this lab and
this cluster. Shared conventions come from `../claude-instructions` (loaded via `../CLAUDE.md`).

## Workspace

`md-workspace.yaml` is how `md-sim` finds everything (it walks up from the cwd, or reads
`$MD_WORKSPACE`). Do not hard-code a path the workspace already states.

- `configs/` (`config_dir`): one YAML per system, plus `templates/` (every field commented, one per
  engine) and `mdp/`. A config describes a physical system and carries no queue facts: no partition,
  walltime or `slurm:` block.
- `sites/<site>.yaml`: the cluster profile named by `site:` in the workspace file: the partitions that
  exist, how to activate the venv, where job logs go, modules to load. It is the only place a
  partition or a venv path is named. Read it before writing a job script.
- `slurm/`: hand-written batch scripts; `slurm/output/` holds job logs. `*.sh.example` files are the
  template examples to start from (one per engine family); the other scripts are real campaigns and
  show arrays, chaining and sharding.
- `data/`: git-ignored symlink to the shared data root `/remote/gpu04/hummerich/data`.
- `beamtime/` (`run_store`, set in `beamtime.yaml`): the experiment log, git-ignored. Every `md-sim run`
  logs into it as system group -> simulation run, or replica-set group -> `rNN` runs. It is Beamtime's
  to manage: read it through the beamtime server, never edit it by hand, and do not add a run listing
  of your own.
- `.mcp.json` (git-ignored, local to each checkout) registers two servers: `beamtime` (run state) and
  `md-simulations` (`md-sim mcp`: `list_configs`, `describe_config`, `validate_config`,
  `config_schema`, `new_config`, `check_job`). The `md-simulations` tools author and check; none of
  them runs or submits anything. The user submits.

## The `md-sim` command

Do not learn it from source or from this file: `md-sim --help` lists the commands and groups,
`md-sim <group>` lists a group's verbs, `md-sim <command> --help` gives the flags, and every command
takes `--json`. The package's `docs/agents.md` is the full agent contract; the rules that matter in
this repo are repeated below because they are the ones that cost real data and queue time.

- Author a config against `md-sim config schema`, start from the nearest existing config or template
  (`md-sim config new --from <cfg> -o configs/<name>.yaml`), and run `md-sim config validate` before
  handing anything over. Write configs into `configs/` only.
- Write a job script with the `write-job` skill (`.claude/skills/write-job/`) and verify it with
  `md-sim check-job` before it goes to the user. A failing check names a real mismatch: fix the
  script, never `md-workspace.yaml` or `sites/` to make it pass.

## Data root

- **The data root is read-only to you.** `/remote/gpu04/hummerich/data` (through `data/`) is the user's
  shared store. Read inputs from it. Do not create, move, edit or delete anything in it, and do not
  point a config's `output_subdir` at an existing run: use a fresh subdirectory name.
- Outputs land in the data root only because the user's job puts them there. A smoke test you run
  yourself writes to a scratch directory (`--data-root "$TMPDIR/..."`), never the data root.
- Every dataset directory carries a `datacard.yaml` as specified in
  `/remote/gpu04/hummerich/data/SCHEMA.md`, and a new dataset is not done until `check-datacards.py`
  passes (`../claude-instructions/data.md`). You cannot write the card, so when a config or script
  creates a new dataset directory, say in the handover that it needs a `datacard.yaml` once it exists.
- Never commit trajectories, checkpoints or data, and never commit `data/` (it is a symlink).

## Jobs: you prepare, the user submits

- gpu04 is a workstation with no Slurm client: `sbatch`, `squeue` and `scancel` do not exist here, so
  do not try them. Prepare the script, run `md-sim check-job` on it, and hand the user the script's
  path, the `check-job` result and the submit command, written the way the institute policy requires
  (a standalone command, no `cd ... &&` prefix): `sbatch --chdir=/remote/gpu04/hummerich/projects/simulations slurm/<name>.sh`,
  and `sbatch --export=ALL,VAR=value ...` for variables. To see progress, ask the user for `squeue`
  output.
- Do not start `md-sim run` for anything longer than a handful of steps to smoke-test a config.
- A GPU is needed by every OpenMM job on the CUDA platform and every `cgschnet` job: request it in the
  script (`--gres=gpu:1`) and pass `--require-gpu` to `md-sim check-job`.

## What belongs here and what belongs in the package

- Here: configs, job scripts, `md-workspace.yaml`, `sites/`, Beamtime state, notebooks and one-off
  analyses of this lab's systems, and anything naming a partition, a venv path or a data directory.
- In the package (`md-simulations`): engines, config models, the `md-sim` command, `prep/`, `sampling/`
  and `analysis/` code, the torch potentials. A change to any of those is made there and picked up
  here by `uv lock --upgrade-package md-simulations`, which the user runs outside Claude Code. Do not
  vendor package code into this repo.
- The package knows nothing about this cluster, so a bug that is really "the script is wrong for the
  site" is fixed in the script or the site profile, not by teaching the package a partition.

## Environment

- `uv sync` and `uv lock` run outside Claude Code (the sandbox cannot write the uv cache); the venv is
  `.venv`, so call `.venv/bin/md-sim`, `.venv/bin/python` and `.venv/bin/pytest` directly, never `uv run`.
- The CUDA constraints (`constraint-dependencies`, the cu130 indexes) in `pyproject.toml` are a copy of
  the package's; they are honoured only top-level, so they stay here.
