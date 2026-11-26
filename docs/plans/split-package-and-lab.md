# Split md-simulations into a package and a lab repo, and give the package an agent-facing CLI

Base commit `04a4aa1`, branch `main`. Proposed 2026-11-26, based on the user's request to disentangle the
two jobs this repo does — the importable package other projects consume, and the simulation directory
holding 82 configs, 30 slurm scripts, the beamtime run store and the data root — and to add a thin agent
layer. Sources read: `README.md`, `CLAUDE.md`, `../claude-instructions/{data,workspace,testing,tooling,git}.md`,
`src/md_simulations/{cli.py,slurm/render.py,config/}`, `tests/`, all of `configs/` and `slurm/`.
Breaking changes are allowed and listed in [Breaking changes](#breaking-changes).

Run a ticket with `/run-ticket docs/plans/split-package-and-lab.md <ID>` in a fresh session; review tickets
marked `+ review` with `/review-ticket docs/plans/split-package-and-lab.md <ID>` on a stronger model.
`ticket.py status docs/plans/split-package-and-lab.md` shows what's done and what's next.

## Goal

- **Done means:** `md-simulations` is a package repo only — `src/`, `tests/`, `examples/`, `docs/` — that
  clones, installs and tests green on a machine with no data root and no cluster. A sibling repo
  `/remote/gpu04/hummerich/projects/simulations` holds `configs/`, the orchestration slurm scripts, the
  beamtime store, the `data/` symlink, `md-workspace.yaml` and `sites/`. The package no longer contains any
  knowledge of this cluster: no partition defaults, no `.venv/bin/activate`, no `$PWD/data`. Its two uses
  are separately installable: the package alone pulls three dependencies (`numpy`, `pydantic`, `pyyaml`),
  `[torch-potentials]` adds torch and ASE without openmm or beamtime, and `[simulate]` adds the MD stack;
  `import md_simulations` imports none of openmm, beamtime or torch. `md-sim` is one
  discoverable noun/verb CLI where every command takes `--json`, an agent can obtain the config JSON Schema
  and validate/describe/scaffold configs without reading source, and `md-sim check-job` verifies a
  hand-written or agent-written sbatch script before it is submitted.
- **Confirmed by:** in a fresh clone of the package with `MD_DATA_ROOT` unset,
  `ruff check . && ruff format --check . && pytest -q` passes; `rg -n 'a30|\.venv/bin/activate|\$PWD/data' src/`
  returns nothing; `md-sim --help` lists the noun groups and `md-sim config schema --json | jq .title` works;
  `md-sim check-job simulations/slurm/alaX-amber14-implicit-remd-330K.sh` passes from the lab repo;
  `pytest -q tests/test_packaging.py` pins the eager import set; in a venv built from
  `md-simulations[torch-potentials]` alone, `python -c "import openmm"` fails while
  `from md_simulations.torch_potentials import CGSchNetPotential` resolves; `md-sim mcp` starts and
  answers `tools/list` with the documented tool set; and split-flows resolves `md-simulations` from the unchanged git URL and imports
  `md_simulations.torch_potentials` unchanged.
- **Out of scope:** generating
  slurm scripts or teaching job chaining; any change to `torch_potentials`' public API, `bridge.py` units, or
  the engines; moving `prep/`, `sampling/` or `analysis/` code into the lab repo; a `datacard.yaml` audit of
  the data root.

## Findings

Evidence: **reproduced** (run), **real data** (seen in the user's data), **code** (read, not run).

| # | Finding | Root cause | Evidence | Ticket |
|---|---|---|---|---|
| 1 | The package hardcodes this cluster: `partition: str = "a30"`, `venv_activate=".venv/bin/activate"`, `./slurm/output`, `MD_DATA_ROOT:-$PWD/data` | `src/md_simulations/config/base.py:8-22` (`SlurmConfig`) and `src/md_simulations/slurm/render.py:12,48,52,65` encode site facts as library defaults | code | W2 |
| 2 | `gen-slurm` covers only the jobs that need no help | 17 of 30 scripts in `slurm/` are 23–26 lines with one `md-sim run` and nothing else; the other 13 (100–257 lines) carry REMD chaining, arrays and loops `render_slurm` cannot express | reproduced (per-file `md-sim` call and construct counts) | W2 |
| 3 | 74 of 82 configs carry a `slurm:` block, mixing queue policy into the description of a physical system | `src/md_simulations/config/base.py:150` `slurm: SlurmConfig \| None` | reproduced (`grep -l '^slurm:' configs/*.yaml \| wc -l` → 74) | W2 |
| 4 | Two torch_potentials tests reach into `REPO/configs` and `REPO/data`, so the package suite breaks the moment configs leave | `tests/torch_potentials/test_cgschnet_regression.py:26-30`, `test_cgschnet_mlcg.py:27-31` | code | S1 |
| 5 | The `prep/`/`sampling/`/`analysis/` triage is a no-op for code: every `ala2`/`cln`/`trp-cage` occurrence is in a docstring or usage example; there are no per-system tables and `polyalanine.py` is an `n`-parameterised builder | campaign prose written into module docstrings | reproduced (grep for system names, then read each hit) | S2 |
| 6 | An agent cannot discover what this package can do: 15 flat `md-sim-*` entry points, no schema, no JSON output, no way to validate a config without running it | `pyproject.toml:124-139`, `cli.py` has only `run` and `gen-slurm` | code | C1, C2 |
| 7 | The git-level split is far smaller than it looks: `beamtime/` is gitignored and untracked (0 tracked files) and `data/*` is gitignored except the README, so only `configs/`, `slurm/*.sh` and `.mcp.json` actually move in git | `.gitignore:1-7`, `git ls-files beamtime \| wc -l` → 0 | reproduced | S3, S4 |
| 9 | `import md_simulations` eagerly imports openmm, beamtime and tqdm, so no slim install is possible whatever the extras say | the lazy-torch design in `src/md_simulations/torch_potentials/__init__.py` is defeated one level down: `src/md_simulations/torch_potentials/build.py:41` imports `md_simulations.engines` at module scope, and `src/md_simulations/cli.py:7` does the same | reproduced (`import md_simulations`, then inspect `sys.modules`) | P1 |
| 10 | `parmed>=4.3.1` is a runtime dependency with zero imports anywhere, and `ipykernel>=7.3.0` — a notebook convenience pulling pyzmq, jupyter-client, debugpy and traitlets — sits in the same list | dependencies accumulated while the repo was also the simulation directory | reproduced (`rg -n 'parmed' src/ tests/ examples/` → nothing) | P2 |
| 11 | `networkx` (`prep/olives.py`) and `matplotlib`/`scienceplots` (eager in `analysis/{tica,cvs,basins}.py`) are imported but declared by no extra; they resolve only because something else drags them in | `analysis` declares `deeptime`/`scipy` only; `viz` is a separate opt-in | reproduced | P2 |
| 8 | `data/` is a real directory here, not the symlink to `/remote/gpu04/hummerich/data` that `claude-instructions/data.md` mandates | predates the convention | reproduced (`ls -la data`) | S3 |

## Brief

- **Workspace layer** (W1): new `src/md_simulations/workspace.py` — pydantic `Site` and `Workspace` models
  loaded from `md-workspace.yaml`, found by walking up from the cwd or named by `$MD_WORKSPACE`. It carries
  `config_dir`, `data_root`, `run_store` and `site`; the site profile carries `partitions`, `venv_activate`,
  `slurm_output`, `modules`. Nothing in `src/` may name a partition, a venv path or a data directory again —
  those live in the lab repo's `sites/*.yaml`. Data-root precedence stays CLI > `MD_DATA_ROOT` > workspace >
  config, extending `config/load.py:resolve_data_root`.
- **Slurm: verification replaces generation** (W2, W3): `src/md_simulations/slurm/`, `tests/test_slurm.py`,
  the `gen-slurm` subcommand, `SlurmConfig` and the `slurm:` block in 74 configs are deleted. In their place
  `src/md_simulations/preflight.py` + `md-sim check-job <script.sh>` parses `#SBATCH` directives and the
  `md-sim run <config>` invocations out of a shell script and checks them against the workspace, the site
  profile and the config. The lab repo keeps hand-written scripts and its new template examples.
- **CLI** (C1, C2): `cli.py` becomes a package `src/md_simulations/cli/` with a noun/verb dispatcher
  (`run`, `config`, `prep`, `analysis`, `sampling`, `check-job`), a shared JSON envelope in `src/md_simulations/cli/_output.py`,
  and `--json` on every command. The 15 `md-sim-*` entry points stay as aliases to the same `main()`s, so no
  consumer or slurm script breaks. New introspection: `md-sim config schema|validate|describe|ls|new`.
- **The split** (S1–S4): clone-then-delete, the pattern `../claude-instructions/workspace.md` records for
  `descriptor-fields` out of `split-flows`. The clone and everything under the new repo must be run by the
  user — the sandbox cannot write outside `md-simulations` — so S3 writes a reviewed script and S4 cleans up
  the package side afterwards.
- **MCP** (C3): `md-sim mcp` — a stdio server over the same typed surface, behind an `mcp` extra the lab
  repo installs and a torch consumer never does. Read-and-author only: it lists, describes, validates and
  scaffolds configs, returns the JSON Schema and runs `check-job`. It never starts a simulation and never
  calls `sbatch`, and it defers run state to the beamtime server already in `.mcp.json`.
- **Dependency layering** (P1, P2): `[project.dependencies]` shrinks to `numpy`, `pydantic`, `pyyaml` —
  enough for configs, the workspace, the schema and `check-job`. `simulate` (openmm, mdtraj, beamtime,
  tqdm) carries the MD half; `torch-potentials` (torch, ase, mdtraj) carries the wrapper half with no
  openmm and no beamtime; `openmm-potentials` is both, for building a potential from an OpenMM config.
  P1 makes this physically possible by taking `md_simulations.engines` out of the eager import path, and
  `tests/test_packaging.py` pins the eager import set so it cannot regress.
- **Invariant, checked in S4:** split-flows keeps working untouched — same git URL
  `https://github.com/hummerichsander/md-simulations`, same `md_simulations.torch_potentials` module path,
  same `mlcg` pin byte-for-byte.

Done, for every ticket: its Done line holds; its Check passes; the docs say what the code does; its heading
is marked done in the ticket's own commit.

## How to execute

1. Checks: `.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check . && .venv/bin/python -m pytest -q`
   from the repo root. A subset: `.venv/bin/python -m pytest -q tests/test_config.py`. Call `.venv/bin/python`
   directly, never `uv run` — the sandbox cannot write the uv cache (`claude-instructions/tooling.md`).
2. Generated files: `uv.lock` is only touched if `pyproject.toml` dependencies change (they should not in this
   plan; entry points are not dependencies). If a ticket does need a relock, it stops and asks — the user runs
   `uv lock` outside Claude Code.
3. Rules: ruff line length 100; docstrings use the `:param`/`:return:` format; `src/md_simulations/prep/olives.py`
   is vendored and excluded from lint — never touch it. Torch imports stay out of the eager import path.
   Keep smoke tests tiny. No file outside this repo may be written by a ticket session; work that needs the
   sibling `simulations/` repo is handed to the user as a script (S3).
4. Spec: this plan is the spec. `README.md` and `CLAUDE.md` are updated in the same commit as the code they
   describe; `../claude-instructions/workspace.md` is updated by the user, not by a ticket (D1 drafts the text).
5. Ticket budget: 140k tokens per `/run-ticket` session. Split a ticket that would exceed it.
6. Recording: one commit per ticket, subject `<ID>: <what changed>`, which also appends ` — **done**` to the
   ticket's heading.

## Order

```
S1 ─ S2            independent, land first (unblock the suite, clean the docstrings)
W1 ─ W2 ─ W3       workspace layer, then delete the generator, then the preflight
     └─ C1 ─ C2    CLI restructure depends on gen-slurm being gone
C3                 the MCP server, on top of C2's surface
P1                 break the eager openmm/beamtime import path (before C1 rewrites cli.py)
D1                 docs, after C2
P2                 re-layer pyproject extras — hands a `uv lock` to the user
S3                 writes the split script → THE USER RUNS IT (blocking handoff)
S4 ─ D2            package-side cleanup, then the lab repo's agent instructions
```

What's left runs in this order: S1 S2 W1 W2 W3 P1 C1 C2 C3 D1 P2 S3 S4 D2

## Decisions taken; say if you want otherwise

1. **Two repos, not a uv workspace monorepo** (your decision): the package keeps the GitHub URL consumers
   already pin, so split-flows needs no edit. Rejected: `packages/` + `lab/` in one checkout, because every
   consumer's `[tool.uv.sources]` entry would need a `subdirectory =` added, and those entries must stay
   byte-identical across consumers (`CLAUDE.md`, the mlcg trap).
2. **The lab repo is `simulations`** (your decision), a sibling under `/remote/gpu04/hummerich/projects/`.
3. **Clone-then-delete, keeping history on both sides** (your decision): the house pattern recorded in
   `workspace.md` for `descriptor-fields`. Rejected: `git filter-repo`, more ways to get it wrong for a
   history that only needs `configs/` and `slurm/`.
4. **Slurm generation is deleted, not extended** (your decision): `render_slurm` only ever covered the 17
   jobs that needed no help, and `slurm:` put queue policy inside the description of a physical system. A
   preflight (`md-sim check-job`) replaces it, because the value was never the templating but the invariants —
   a wrong partition queues forever, a wrong data root silently writes a 10 ns trajectory into the wrong tree,
   and `platform: CUDA` without `--gres=gpu:1` "succeeds" at 1/50 speed. Rejected: teaching `render_slurm`
   chaining and arrays, which would roughly double this plan.
5. **`md-sim <noun> <verb>`, old names kept as aliases** (your decision): one entry point an agent can
   enumerate, zero breakage for the 13 orchestration scripts that call `md-sim-*`.
6. **No run listing or run status in `md-sim`** (proposed): beamtime already owns run state and ships its own
   CLI and MCP server (`.mcp.json`), so `md-sim run ls|status` would be a second source of truth over the same
   store. The agent layer covers configs, schema, jobs and preflight; runs stay beamtime's. Say if you want
   them anyway.
7. **The MCP server ships in this plan, behind an `mcp` extra** (your decision): it is built on the typed
   CLI (C1, C2) rather than beside it, so there is one surface with two front ends, and the `mcp`
   dependency is installed by the `simulations` repo only — a torch consumer never pays for it. Rejected:
   deferring it to a later plan (the original proposal), and putting the MCP dependency in core.
8. **`prep/`/`sampling/`/`analysis/` code stays in the package** (proposed, replacing your "move campaign
   scripts" answer): the triage found no system-specific logic to move, only campaign flavour in docstrings
   (finding 5). S2 genericises the prose instead. Say if you want modules moved regardless.
9. **Runtime dependencies are `numpy`, `pydantic`, `pyyaml`, and nothing else** (your decision): the two
   uses become separately installable. `pip install md-simulations` consequently cannot run a simulation —
   `md-sim run` and every engine import raise `ModuleNotFoundError` naming the `simulate` extra, as
   `torch_potentials` already does for `torch-potentials`. Rejected: slimming only the torch side, since
   openmm and beamtime are the two heaviest items and would stay mandatory.
10. **Extra names: `simulate`, `torch-potentials`, `openmm-potentials`, `mcp`** (proposed): `simulate` says what
   it is for rather than naming an engine, so `gromacs_native` fits under it too. `openmm-potentials` is
   the union a consumer needs to call `build_potential_from_file` on an AMBER/AWSEM config; a CGSchNet
   consumer takes `cgschnet-potentials` and gets no openmm at all. Rejected: `md` and `engines`, both
   ambiguous next to `md-sim` and `md_simulations.engines`.
11. **`ipykernel`, `jupyter` and the plotting stack move to the lab repo** (your decision): `simulations`
   gets its own `pyproject.toml` depending on `md-simulations[simulate,analysis,viz,...]` plus the
   notebook tooling. Rejected: a `notebook` extra in the package, which would keep the dependency in the
   distributed artifact for everyone.
12. **The cluster CUDA constraints are repeated, not moved** (proposed): `constraint-dependencies`
   (nvrtc/nvcc < 13.2 for driver 590.48.01) and the cu130 indexes are honoured only top-level, and both
   repos are top-level in turn. Each keeps a copy, the package's marked "for this repo's own dev venv".
   Rejected: moving them to the lab repo alone (breaks CUDA in the package's venv) or dropping them
   (reintroduces `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)`).
14. **The MCP tool set is read-and-author only** (proposed): no `run_simulation` and no `submit` — gpu04
   has no Slurm client, and a 40-hour GPU job is not something an agent starts unattended. The agent
   authors, validates and checks; the user submits. Rejected: a `run` tool for "short" jobs, since short
   is not checkable from the config alone.
15. **The data root stays outside both repos** at `/remote/gpu04/hummerich/data`, reached from the lab repo by
   a gitignored `data/` symlink per `claude-instructions/data.md`. The package never defaults a data root.

16. **`md-workspace.yaml` discovery is the package's own, not beamtime's** (settled in W1's review, from W1's
   "Stop if"): `beamtime.sdk.paths.find_root` finds a `beamtime.yaml` (or the git root) for the run store and
   is a different descriptor; reusing it would also put beamtime — a `simulate`-extra dependency under
   Decision 9 — on the core import path that `workspace.py`, `md-sim config` and `check-job` must stay free
   of. `workspace.py` therefore keeps its own upward walk.

17. **The orchestration scripts carry their own job names and walltimes** (settled in W2's review): four
   submit-mode scripts (`slurm/ala2-amber14-implicit-endpoints.sh`,
   `slurm/alaX-amber14-implicit-330K-replicas.sh`, `slurm/alaX-amber14-implicit-remd-330K.sh`,
   `slurm/cln-fiber-amber14-implicit-330K.sh`) read `job_name`/`time` out of each config's `slurm:` block
   with an `awk` helper and submitted with empty `--job-name=`/`--time=` once W2 deleted the block. The
   values are now literals in the scripts, which is Decision 4 applied consistently — queue facts live in
   the hand-written script, not in the description of a physical system. The scripts move to the lab repo in
   S3 self-contained; nothing there reads a config for queue parameters.

18. **A job's need for a GPU is declared, not inferred** (settled in W3's review, amending Decision 4):
   no config model has a `platform` field, so `check-job` cannot read "this job needs a GPU" out of the
   config. `md-sim check-job --require-gpu` makes the author or the agent writing the script state it;
   without the flag the `gres` check is `ok` when a GPU is requested and `skipped` otherwise, never a
   fail. Decision 4's `platform: CUDA` line stands as the motivating mistake, not as a description of how
   the check works. C3's `check_job` MCP tool therefore takes an optional `require_gpu` argument
   alongside the script path.

19. **`md_simulations.core` resolves its names lazily, so the numpy-only ones stay reachable**
   (settled in P1's review): P1 wrapped `core/__init__`'s eager imports in a `simulate`-extra hint, which
   is right for `setup_logger`/`start_run`, but the package `__init__` runs on any submodule import, so
   `md_simulations.core.forces` — numpy only — is unimportable without openmm. That was already true
   before P1 (the eager imports pulled openmm all the same); P1 only made the message clearer. It blocks
   P2's layering: `prep/migrate_forces.py` is numpy plus `core.forces`, and
   `prep/cgschnet_dataset.py` is mdtraj plus `core.forces`, so the `prep` and `cgschnet-dataset` extras
   as drawn in P2 step 6 would resolve to a package their modules cannot import. The fix is the
   `_LAZY`/`__getattr__` pattern of `torch_potentials/__init__.py` applied to `core/__init__`: the
   openmm/beamtime-backed names resolve on first access with `EXTRA_HINT`, while
   `md_simulations.core.forces` imports on a core install. Rejected: adding `md-simulations[simulate]`
   to `prep`, which puts openmm back under a dataset-preparation extra and contradicts Decision 9.

## S. Split and fixtures

### S1. Free the package suite from `configs/` and `data/` (sonnet) — code — **done**

- **Touches:** `tests/torch_potentials/test_cgschnet_regression.py` (`REPO`/`DATA_ROOT`/`CONFIG` 25–30),
  `tests/torch_potentials/test_cgschnet_mlcg.py` (`REPO`/`DATA_ROOT`/`CONFIG` 26–31), new
  `tests/fixtures/trpcage-cgschnet.yaml`, new `tests/fixtures/2JOF_5B-CG.pdb`, `tests/conftest.py`.
- **Read:** this plan's "Findings" row 4.
- **Check:** `.venv/bin/python -m pytest -q tests/torch_potentials`; then with the data root hidden:
  `env -u MD_DATA_ROOT .venv/bin/python -m pytest -q tests/torch_potentials` — all cgschnet regression tests
  report *skipped*, none error.
- **Done:** neither test file contains the strings `REPO / "configs"` or `REPO / "data"`; with `MD_DATA_ROOT`
  unset the 13 tests in the two files skip with a clear reason; with `MD_DATA_ROOT=/remote/gpu04/hummerich/data`
  (or the current `data/`) they pass exactly as before.
- **Risks:** the vendored config drifts from the real `trpcage-cgschnet-300K.yaml` → the fixture carries a
  comment naming its source and the test asserts the bead count it expects; `MD_DATA_ROOT` set but pointing at
  a tree without `trp-cage/` → the existing per-path `pytest.skip` at `:42`/`:51` must still fire, add
  `test_fixture_config_loads` to prove the config parses with no data root at all.
- **Stop if:** the vendored bead PDB turns out to be needed by anything outside tests.

Both files already skip gracefully when a file is missing; what breaks after the split is that `REPO/configs`
and `REPO/data` will not exist at all. Repoint them at a vendored fixture plus `MD_DATA_ROOT`.

1. Copy `data/pdbs/2JOF_5B-CG.pdb` (8 KB) to `tests/fixtures/2JOF_5B-CG.pdb`, tracked.
2. Copy `configs/trpcage-cgschnet-300K.yaml` to `tests/fixtures/trpcage-cgschnet.yaml`, dropping the `slurm:`
   block, and prepend a comment: the file is a test fixture copied from the `simulations` repo at `04a4aa1`.
3. In `tests/conftest.py` add a `data_root` fixture: `Path(os.environ["MD_DATA_ROOT"])` if set, else the
   repo's `data/` if it exists, else `pytest.skip("needs MD_DATA_ROOT")`. Both test modules take it.
4. In both modules, `CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "trpcage-cgschnet.yaml"` and
   `BEAD_PDB = .../fixtures/2JOF_5B-CG.pdb`; `DATA_ROOT` comes from the fixture. Keep the module-scope
   `pytest.importorskip("mlcg")` where it is.

### S2. Genericise the campaign prose in module docstrings (haiku) — code

- **Touches:** `src/md_simulations/prep/cgschnet_priors.py` (docstrings 5, 26–34, 64, 459),
  `src/md_simulations/prep/extract_frames.py` (16), `src/md_simulations/analysis/basins.py` (14, 20),
  `src/md_simulations/sampling/equilibrate_h_bonds.py` (13, 59, 260),
  `src/md_simulations/sampling/resample_h_bonds.py`, `src/md_simulations/prep/cgschnet_dataset.py`,
  `src/md_simulations/prep/cgschnet_finetune.py`.
- **Read:** this plan's "Findings" row 5 and Decision 8.
- **Check:** `.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .`;
  `.venv/bin/python -m pytest -q`.
- **Done:** `rg -niE 'ala2|alaX|cln|chignolin|trp-?cage|2JOF|1UAO' src/md_simulations/` returns only lines
  explicitly framed as *measurements* ("on trp-cage at 330 K a plain threshold gives 470 crossings"), never
  as usage examples or path examples; no line of executable code changed (`git diff -U0 | grep '^[+-]' |
  grep -v '^[+-][+-]'` shows only comment/docstring lines).
- **Risks:** deleting a measurement that justifies a default → keep every sentence that reports a number, only
  rewrite the ones that read as "here is how I ran this on cln"; rewriting a doctest-shaped example into
  something that no longer parses → none of these are doctests, confirm with
  `rg -n '>>>' src/md_simulations/`.
- **Stop if:** a system name turns out to be load-bearing in code rather than prose — that is a different
  ticket and Decision 8 is wrong.

This is prose only. Usage examples like `md-sim-cgschnet-priors --dataset data/alaX/ala2_fm_330K` become
`--dataset <data-root>/<system>_fm_<T>K`; `-o data/pdbs/cln-rep` becomes `-o <data-root>/pdbs/<system>-rep`.
Measurements stay, because they are evidence for the defaults around them.

## W. Workspace, site and the slurm turnaround

### W1. The workspace descriptor and site profile (sonnet + review) — code — **done**

- **Touches:** new `src/md_simulations/workspace.py`, `src/md_simulations/config/load.py`
  (`resolve_data_root` 23–41), `src/md_simulations/config/__init__.py` (exports 1–30), new
  `tests/test_workspace.py`, `tests/test_config.py` (`test_data_root_precedence` 43–52).
- **Read:** `../claude-instructions/data.md` (the shared root and the `data/` symlink rule); this plan's
  Decision 9.
- **Check:** `.venv/bin/python -m pytest -q tests/test_workspace.py tests/test_config.py`; then the full
  suite `.venv/bin/python -m ruff check . && .venv/bin/python -m pytest -q`.
- **Done:** `load_workspace()` finds `md-workspace.yaml` by walking up from a nested cwd; `$MD_WORKSPACE`
  overrides the walk; a missing descriptor raises `WorkspaceNotFound` with the directories searched;
  `${MD_DATA_ROOT:-./data}` in the descriptor expands from the environment with the default applied;
  `resolve_data_root` honours CLI > `MD_DATA_ROOT` > workspace > config and the existing precedence test
  still passes unchanged.
- **Risks:** a relative `config_dir`/`run_store` resolved against the cwd instead of the descriptor's own
  directory → `test_relative_paths_resolve_against_descriptor`; the upward walk escaping to `/` or into
  `$HOME` → stop at a filesystem boundary or after the git root, `test_walk_stops_at_git_root`; a site name
  with no matching file → `WorkspaceError` naming the searched `sites/` dir, `test_unknown_site`;
  `${VAR}` with no default and no value → explicit error, not an empty path, `test_unset_interpolation`.
- **Stop if:** beamtime's own root discovery (`beamtime.sdk.paths.find_root`) already provides a descriptor
  this should reuse rather than duplicate — check before writing the walk.

Today the package learns where things are by being run from this checkout. One file states it instead, and it
is also what the agent CLI reads to orient itself.

1. `class Site(BaseModel)`: `name: str`, `partitions: list[str] = []`, `venv_activate: str | None = None`,
   `slurm_output: str = "./slurm/output"`, `modules: list[str] = []`. Empty `partitions` means "unknown,
   do not check" — the preflight must treat it that way (W3).
2. `class Workspace(BaseModel)`: `root: Path` (the descriptor's directory, excluded from YAML),
   `config_dir: Path = Path("configs")`, `data_root: str | None`, `run_store: Path = Path("beamtime")`,
   `site: str | None`. Methods: `resolve(p) -> Path` (relative paths against `root`), `load_site() -> Site`
   reading `root / "sites" / f"{site}.yaml"`, `configs() -> list[Path]` (sorted `*.yaml` under `config_dir`,
   excluding `generated/`).
3. `load_workspace(start: Path | None = None) -> Workspace`: `$MD_WORKSPACE` if set (a file or a directory
   containing `md-workspace.yaml`), else walk up from `start or Path.cwd()` to the git root or the filesystem
   root. Raise `WorkspaceNotFound(searched: list[Path])`.
4. `_expand(value: str) -> str` for `${VAR}` and `${VAR:-default}` only — no shell, no command substitution.
5. `resolve_data_root(config, cli_data_root=None, workspace=None)`: insert the workspace between the env var
   and the config value; keep the signature backward compatible (new keyword, defaulted). The error message
   gains "or set `data_root` in md-workspace.yaml".
6. Export `Workspace`, `Site`, `load_workspace`, `WorkspaceError`, `WorkspaceNotFound` from
   `md_simulations.workspace`, not from `config` — the config package stays about configs.

### W2. Delete slurm generation and the `slurm:` config block (sonnet + review) — code — **done**

- **Touches:** `src/md_simulations/slurm/` (whole package removed: `__init__.py`, `render.py`),
  `tests/test_slurm.py` (removed), `src/md_simulations/cli.py` (`_cmd_gen_slurm` 45–56, `p_gen` 82–90,
  import 8), `src/md_simulations/config/base.py` (`SlurmConfig` 8–22 removed, `slurm` field 150 removed),
  `src/md_simulations/config/__init__.py` (`SlurmConfig` export), `configs/*.yaml` (74 files, `slurm:` block
  removed), `tests/conftest.py` (`SAMPLE_SLURM` and its uses), `README.md`, `CLAUDE.md`.
- **Read:** this plan's Decision 4 and "Breaking changes"; `../claude-instructions/cluster.md`.
- **Check:** `.venv/bin/python -m pytest -q`; `rg -n 'slurm' src/ --type py` returns nothing;
  `rg -n '^slurm:' configs/ | wc -l` returns 0; `.venv/bin/python -c "import yaml,pathlib;
  [yaml.safe_load(p.read_text()) for p in pathlib.Path('configs').glob('*.yaml')]"` exits 0;
  `.venv/bin/python -m md_simulations.cli run configs/ala2-amber14-implicit-330K.yaml --help` still works.
- **Done:** `md-sim gen-slurm` is gone from `--help`; every one of the 82 configs loads through
  `load_config` with no `slurm` key and no validation error; the existing 30 scripts in `slurm/` are
  untouched and still reference the same config paths.
- **Risks:** a config where `slurm:` is not a trailing block and the strip eats the next key → strip by
  loading YAML and re-dumping is *not* acceptable (it would reorder and drop comments), so strip textually and
  verify every file still `yaml.safe_load`s and that the key set is unchanged apart from `slurm`, with a
  one-off script asserting both before committing; a config using `slurm:` inside a nested block → `rg -n
  '^\s+slurm:' configs/` first, expect zero; `tests/conftest.py`'s `SAMPLE_SLURM` leaking into another test →
  grep its uses.
- **Stop if:** more than 74 files change, or any config's non-`slurm` keys differ before and after.

The generator only ever covered the 17 jobs that needed no help, and `partition`/`mem`/`time` are facts about
the queue, not about alanine dipeptide. Both go; W3 puts a checker in their place.

1. `git rm -r src/md_simulations/slurm tests/test_slurm.py`.
2. `cli.py`: drop the `render_slurm` import, `_cmd_gen_slurm` and the `p_gen` parser block. Leave `run` as it is.
3. `src/md_simulations/config/base.py`: delete `SlurmConfig` and the `slurm: SlurmConfig | None = None` field; remove the export.
4. Strip the trailing `slurm:` block from the 74 configs textually: delete the `slurm:` line and the indented
   lines that follow it, plus a single preceding blank line. Verify as the Risks line requires.
5. `conftest.py`: delete `SAMPLE_SLURM` and any fixture merging it.
6. README: delete the `gen-slurm` usage; add two sentences saying job scripts are written by hand (or by an
   agent) in the `simulations` repo against its template examples and checked with `md-sim check-job`.

### W3. `md-sim check-job`: preflight an sbatch script (sonnet + review) — code — **done**

- **Touches:** new `src/md_simulations/preflight.py`, `src/md_simulations/cli.py` (new `check-job` parser),
  new `tests/test_preflight.py`.
- **Read:** this plan's Decision 4; `src/md_simulations/workspace.py` (`Site`, from W1);
  `configs/ala2-amber14-implicit-330K.yaml` (`platform`, `input_pdb`).
- **Check:** `.venv/bin/python -m pytest -q tests/test_preflight.py`; then against two real scripts from the
  working tree: `.venv/bin/python -m md_simulations.cli check-job slurm/ala2-amber14-implicit-330K.sh` and
  `... slurm/alaX-amber14-implicit-remd-330K.sh`.
- **Done:** a correct script reports every check `OK` and exits 0; a script whose `--partition` is not in the
  site's `partitions` exits 1 naming the partition and the allowed set; with `--require-gpu`, a script that
  requests no GPU (`--gres=gpu:N` or `--gpus=N`) exits 1, and without the flag the `gres` check is `ok` when a
  GPU is requested and `skipped` otherwise (amended: no config model has a `platform` field, so the need for
  a GPU cannot be read from a config and the author or agent states it); a script whose `md-sim run` config path does not exist
  exits 1; `--json` prints `{"ok": bool, "checks": [{"name", "ok", "detail"}]}`; the REMD orchestration script
  (191 lines, no literal `md-sim run`) is handled, not crashed on.
- **Risks:** a script that builds the config path in a shell variable or a loop → the config check reports
  `skipped: config path not statically determinable`, never a false pass (`test_dynamic_config_path`); a site
  with empty `partitions` → that check is `skipped`, not failed (`test_unknown_partitions_skipped`); `#SBATCH`
  after the first non-comment line, which slurm ignores → warn (`test_sbatch_after_code`); a config whose
  `input_pdb` is absolute → resolve as-is (`test_absolute_input_path`).
- **Stop if:** a check cannot be made both cheap and certain — a preflight that passes a broken script is
  worse than no preflight, so prefer `skipped` over a guess.

A wrong partition queues forever, a wrong data root silently writes a 10 ns trajectory into the wrong tree, and
`platform: CUDA` without a GPU "succeeds" at 1/50 speed. These are the mistakes an agent makes and cannot see.

1. `parse_script(text) -> ScriptFacts`: `#SBATCH` long/short options into a dict (stop at the first
   non-comment, non-blank line and record whether any `#SBATCH` appeared later), plus every statically
   determinable `md-sim run <path>` config path (a bare path, no `$`), plus any `export MD_DATA_ROOT=` value.
2. `check_job(path, workspace, *, data_root=None) -> list[Check]` with `Check(name, status: "ok"|"fail"|
   "skipped"|"warn", detail)`. Checks, in order: `config` (each path exists and `load_config` succeeds),
   `inputs` (each config's `input_pdb`/`input_gro`/`top`/`model_file` resolves under the data root and exists),
   `partition` (in `site.partitions` when non-empty), `gres` (`--gres`/`--gpus` must request a GPU when
   `--require-gpu` is passed; amended, see Done), `output` (the `--output`/`--error` directory is creatable), `venv`
   (`site.venv_activate` exists, when set).
3. `md-sim check-job <script.sh> [--data-root P] [--json]`: exit 0 when no check failed, 1 otherwise. Human
   output is one aligned line per check, as in the Done line.
4. Tests write tiny scripts and configs into `tmp_path` with a `Workspace` pointed at it; no real cluster.

## C. The agent-facing CLI

### C1. `md-sim <noun> <verb>`, `--json`, and aliases (sonnet + review) — code — **done**

- **Touches:** `src/md_simulations/cli.py` (removed, becomes a package), new `src/md_simulations/cli/__init__.py`,
  new `src/md_simulations/cli/main.py`, new `src/md_simulations/cli/_output.py`, new
  `src/md_simulations/cli/_dispatch.py`; `pyproject.toml` (`[project.scripts]` 124–139);
  new `tests/test_cli.py`.
- **Read:** this plan's Decision 5; `pyproject.toml:124-139`; the `main()` signatures (all 18 take no
  arguments and parse `sys.argv` themselves).
- **Check:** `.venv/bin/python -m pytest -q tests/test_cli.py`; `.venv/bin/python -m md_simulations.cli --help`;
  after `.venv/bin/python -m pip install -e . --no-deps`, `md-sim-cgschnet-priors --help` and
  `md-sim prep cgschnet-priors --help` print the same text.
- **Done:** `md-sim --help` lists `run`, `config`, `prep`, `analysis`, `sampling`, `check-job`;
  `md-sim prep <name> --help` is identical to `md-sim-<name> --help` for all 14 aliased scripts; all 15
  `md-sim-*` entry points still exist in `pyproject.toml`; `md-sim run <cfg> --json` prints one JSON object
  on stdout and nothing else, with human logging still going to the log file and stderr; a failure exits
  non-zero with `{"ok": false, "error": {"type", "message"}}` on stdout under `--json`.
- **Risks:** a subcommand's own `--json` colliding with the global one → the dispatcher passes everything
  after the verb through untouched and only `md-sim`'s own commands parse `--json`
  (`test_passthrough_preserves_argv`); `sys.argv[0]` rewriting breaking a script's `prog=` in its usage line →
  set `sys.argv[0]` to the alias name (`test_usage_shows_alias`); importing every prep module at startup
  dragging torch into the eager path → the dispatch table maps names to `module:function` *strings*, imported
  only when that verb is chosen (`test_no_torch_import_on_help`); exit codes swallowed by the dispatcher
  (`test_exit_code_propagates`).
- **Stop if:** any existing `md-sim-*` command's behaviour would change in any way other than its `prog` name.

An agent cannot use 15 names it has to be told. One entry point it can enumerate replaces them, and the old
names stay so the 13 orchestration scripts in the lab repo keep working.

1. `src/md_simulations/cli/_output.py`: `emit(data: dict, *, json_mode: bool)` and `fail(exc, *, json_mode) -> NoReturn`. JSON
   envelope `{"ok": true, ...payload}` / `{"ok": false, "error": {"type", "message"}}`, printed to stdout with
   `sort_keys=False`; human mode keeps today's formatting. Exit codes: 0 ok, 1 expected failure, 2 usage.
2. `src/md_simulations/cli/_dispatch.py`: `GROUPS: dict[str, dict[str, str]]` mapping e.g.
   `"prep": {"cgschnet-dataset": "md_simulations.prep.cgschnet_dataset:main", ...}` — strings only, no imports
   at module level. `dispatch(group, verb, argv) -> int` sets `sys.argv = [f"md-sim-{verb}", *argv]`, imports
   lazily, calls `main()`, and translates `SystemExit` into a return code.
3. `src/md_simulations/cli/main.py`: `run`, `check-job` and the `config` group are parsed by `md-sim` itself; `prep`, `analysis`
   and `sampling` pass through to `dispatch`. `md-sim <group>` with no verb lists the verbs; add `--json` to
   that listing too, so an agent can enumerate the surface.
4. `src/md_simulations/cli/__init__.py` re-exports `main` so `md-sim = "md_simulations.cli:main"` in `pyproject.toml` is unchanged.
5. `pyproject.toml`: keep all 15 existing entries exactly as they are. Add nothing.
6. `md-sim run` gains `--json`: on success emit `{"ok": true, "config", "data_root", "output_dir", "engine",
   "system", "steps"}`; the run's own logging is unaffected.

### C2. `md-sim config schema|validate|describe|ls|new` (sonnet) — code — **done**

- **Touches:** new `src/md_simulations/cli/config_cmds.py`, new-in-C1 `src/md_simulations/cli/main.py` (the
  `config` group), `tests/test_cli.py`.
- **Read:** `src/md_simulations/config/base.py` (`_Common` 123–210, the discriminated union 369),
  `src/md_simulations/workspace.py` (from W1).
- **Check:** `.venv/bin/python -m pytest -q tests/test_cli.py`; by hand from the repo root:
  `.venv/bin/python -m md_simulations.cli config schema --json | .venv/bin/python -c "import json,sys; d=json.load(sys.stdin); print(len(d['payload']['oneOf']))"`,
  `... config validate configs/ala2-amber14-implicit-330K.yaml`,
  `... config describe configs/ala2-amber14-implicit-330K.yaml --json`.
- **Done:** `config schema` emits the JSON Schema of the discriminated `SimulationConfig` union with one
  branch per engine, and `--engine amber` emits just that branch; `config validate` exits 0 with
  `{"ok": true}` on a good file and 1 with every pydantic error as `{"loc", "msg", "type"}` on a bad one;
  `config describe` reports engine, system, resolved `output_dir`, resolved input paths with an `exists` flag
  per path, `beamtime_group`/`beamtime_name`, and the derived `total_time_ns`, `frame_spacing_ps`,
  `n_frames`; `config ls` lists the workspace's configs with engine, system and temperature; `config new
  --from <cfg> --set temperature=350 --set output_subdir=... -o <path>` writes a config that `config validate`
  accepts.
- **Risks:** `describe` requiring a data root that does not exist → resolve paths but never stat-fail; report
  `exists: false` and still exit 0 (`test_describe_without_data_root`); `--set` on a nested key
  (`replica_exchange.n_replicas`) → support dotted paths or reject them explicitly, do not write a literal
  dotted key (`test_set_nested_key`); `--set` coercing `temperature=350` to the string `"350"` → values are
  parsed as YAML scalars (`test_set_type_coercion`); `config ls` with no workspace → a clear
  `WorkspaceNotFound` message, not a traceback (`test_ls_without_workspace`).
- **Stop if:** `model_json_schema()` on the discriminated union produces something an agent cannot use (no
  `oneOf`, unresolved `$ref`s) — say so rather than hand-writing a schema.

This is the layer that lets an agent author a valid config without reading `src/md_simulations/config/base.py`, and check it
without running a simulation.

1. `schema`: `TypeAdapter(SimulationConfig).json_schema(ref_template="#/$defs/{model}")`; `--engine <name>`
   selects the branch by its `engine` literal.
2. `validate <path>`: `load_config`, catching `pydantic.ValidationError` and `yaml.YAMLError` separately.
3. `describe <path> [--data-root]`: the fields in the Done line. Derived values:
   `total_time_ns = num_steps * timestep / 1000`, `frame_spacing_ps = output_freq * timestep`,
   `n_frames = num_steps // output_freq`.
4. `ls [--json]`: `load_workspace().configs()`, each loaded leniently — a config that fails to parse is listed
   with its error, not skipped silently.
5. `new --from <cfg> [--set k=v ...] -o <path>`: load the source YAML as text, apply `--set` by re-dumping
   (comments are lost — say so in `--help`), validate the result before writing, refuse to overwrite without
   `--force`.

### C3. `md-sim mcp`: the same surface over MCP (sonnet + review) — code — **done**

- **Touches:** new `src/md_simulations/mcp_server.py`, `src/md_simulations/cli/main.py` (an `mcp`
  subcommand), `pyproject.toml` (an `mcp` extra), new `tests/test_mcp_server.py`,
  new `docs/lab-repo-mcp.json` (a draft `.mcp.json` for the `simulations` repo).
- **Read:** this plan's Decisions 6 and 7; `.mcp.json` (the beamtime server already registered here);
  `src/md_simulations/cli/config_cmds.py` (from C2) — the tools call those functions, never a subprocess.
- **Check:** `.venv/bin/python -m pytest -q tests/test_mcp_server.py`; then by hand,
  `.venv/bin/python -m md_simulations.cli mcp` answering a `tools/list` request on stdin.
- **Done:** `md-sim mcp` serves exactly `list_configs`, `describe_config`, `validate_config`,
  `config_schema`, `new_config` and `check_job`; every tool's result is the same payload the matching
  `--json` command prints; each tool has a description and a JSON input schema; without the `mcp` extra,
  `md-sim mcp` exits 1 with a message naming it, and `import md_simulations` still imports no `mcp`
  module; `tests/test_packaging.py`'s eager-import assertion still passes.
- **Risks:** a tool that mutates outside the workspace — `new_config` must refuse a path outside
  `workspace.config_dir` (`test_new_config_path_escape`); an exception escaping as a transport error
  instead of a tool result (`test_tool_error_is_a_result`); a tool returning a `Path` or a pydantic model
  that will not serialise (`test_results_are_json_serialisable`); the server importing engines or torch at
  startup and dragging openmm back in (`test_server_import_is_slim`).
- **Stop if:** a tool would need to start a simulation or submit a job — that is deliberately not in the
  tool set (Decision 14), so stop and ask rather than adding it.

One surface, two front ends. The tools are thin wrappers over C2's functions, so an agent and a terminal
see the same answers, and anything the CLI learns later the server gets for free.

1. `mcp_server.py`: build the server with the official `mcp` Python SDK, stdio transport, server name
   `md-simulations`. Import the SDK inside `serve()`, not at module scope.
2. One handler per tool, each calling the C2 function and returning its dict unchanged. `check_job` takes
   an absolute or workspace-relative script path and an optional `require_gpu` (Decision 18) and returns
   `{"ok", "checks"}` from W3.
3. The workspace is resolved once at startup from `$MD_WORKSPACE` or the cwd, and reported by every tool's
   result as `"workspace"`, so an agent can see which tree it is talking about.
4. `md-sim mcp [--workspace P]` in `src/md_simulations/cli/main.py`, importing `mcp_server` lazily.
5. `docs/lab-repo-mcp.json`: both servers — the existing `beamtime` entry with its `--root` pointed at the
   lab repo, and a new `md-simulations` entry running `md-sim mcp` from the lab venv.

## P. Packaging and dependency layering

### P1. Take openmm and beamtime out of the eager import path (sonnet + review) — code — **done**

- **Touches:** `src/md_simulations/torch_potentials/build.py` (`from md_simulations.engines import
  BuiltSystem, OpenMMEngine, build_engine` 41, moved into the functions that use it),
  `src/md_simulations/cli.py` (`from md_simulations.engines import build_engine` 8, moved into `_cmd_run`
  11–42), `src/md_simulations/engines/__init__.py` (a `ModuleNotFoundError` hint),
  `src/md_simulations/core/__init__.py` (the same), new `tests/test_packaging.py`.
- **Read:** `src/md_simulations/torch_potentials/__init__.py` (the lazy-resolution docstring and `_LAZY`
  60–110) — this ticket extends that pattern one level down; this plan's Finding 9 and Decision 9.
- **Check:** `.venv/bin/python -m pytest -q tests/test_packaging.py tests/torch_potentials`; then the full
  suite `.venv/bin/python -m ruff check . && .venv/bin/python -m pytest -q`.
- **Done:** after `import md_simulations`, `sys.modules` contains none of `openmm`, `beamtime`, `tqdm`,
  `mdtraj`, `torch`, `ase` (asserted in `tests/test_packaging.py`); `import
  md_simulations.torch_potentials` is likewise clean; `build_potential_from_file` on an AMBER config still
  works exactly as before; `import md_simulations.engines` without openmm raises `ModuleNotFoundError`
  whose message names the `simulate` extra.
- **Risks:** `BuiltSystem`/`OpenMMEngine` used in annotations evaluated at runtime → keep them under
  `TYPE_CHECKING` with `from __future__ import annotations` already present at `build.py:29`
  (`test_annotations_do_not_import_engines`); an `isinstance(engine, OpenMMEngine)` check inside a function
  needing the real class → import it inside that function, not at module scope
  (`test_openmm_engine_dispatch_still_works`); a circular import once `engines` is imported lazily from
  `build` while `engines` imports `config` (`test_import_order_both_ways`); the eager set regressing later
  through a new top-level import (`test_eager_imports` is the guard itself).
- **Stop if:** some module genuinely cannot defer its engine import without changing behaviour — say which
  and stop, rather than making the import conditional at runtime.

The lazy-torch design in `src/md_simulations/torch_potentials/__init__.py` is already right; it is simply defeated one level
down. Until this ticket lands, no arrangement of extras can produce a slim install, so P2 depends on it.

1. `src/md_simulations/torch_potentials/build.py`: move line 41 into the bodies of `build_calculator`, `build_potential`,
   `build_potential_terms` and any other user; keep the names under the existing `if TYPE_CHECKING:` block
   at 43–47 for annotations.
2. `src/md_simulations/cli.py`: move the `build_engine` import into `_cmd_run`.
3. `src/md_simulations/engines/__init__.py` and `src/md_simulations/core/__init__.py`: wrap the openmm/beamtime imports so a missing module
   raises `ModuleNotFoundError("md_simulations.engines needs the 'simulate' extra: pip install
   'md-simulations[simulate]'")`, mirroring `torch_potentials.build.EXTRA_HINT`. Export a module-level
   `EXTRA_HINT` so the CLI can reuse the wording.
4. `tests/test_packaging.py`: `test_eager_imports` runs `sys.executable -c "import md_simulations; import
   sys; print(json.dumps(sorted(sys.modules)))"` in a subprocess (a clean interpreter, not the test
   session's polluted one) and asserts the forbidden set is absent.

### P2. Re-layer the dependencies (sonnet + review) — code

- **Touches:** `pyproject.toml` (`dependencies` 16–28, `[project.optional-dependencies]` 30–120,
  `[tool.uv]` 146–180), new `docs/lab-repo-pyproject.toml`, `README.md` (install section), `CLAUDE.md`,
  `src/md_simulations/core/__init__.py` (the lazy resolution of Decision 19), `tests/test_packaging.py`.
- **Read:** this plan's Findings 9–11 and Decisions 9–12 and 19; `src/md_simulations/torch_potentials/__init__.py`
  (`_LAZY` and `__getattr__`, the pattern step 10 copies); `pyproject.toml`'s existing comments on the
  `cgschnet`/`cu130` separation — that reasoning is unchanged and must survive the edit.
- **Check:** `.venv/bin/python -c "import tomllib,pathlib;
  tomllib.loads(pathlib.Path('pyproject.toml').read_text())"`; then **the user runs `uv lock` outside
  Claude Code** (the sandbox cannot write the uv cache) and pastes the result; then
  `.venv/bin/python -m pytest -q`.
- **Done:** `[project.dependencies]` is exactly `numpy`, `pydantic`, `pyyaml`; `parmed` and `ipykernel`
  appear nowhere; `simulate`, `torch-potentials`, `openmm-potentials`, `mcp`, `analysis` and `prep` carry what
  their modules actually import, including the previously undeclared `networkx`, `matplotlib` and
  `scienceplots`; `all` still resolves; `uv lock` succeeds and `uv.lock` is committed in the same commit;
  `docs/lab-repo-pyproject.toml` is a complete, ready-to-copy project file for the `simulations` repo;
  and, per Decision 19, `import md_simulations.core.forces` succeeds with openmm unimportable while
  `from md_simulations.core import setup_logger` still raises `EXTRA_HINT` (both asserted in
  `tests/test_packaging.py`, extending `test_engines_without_openmm_names_the_simulate_extra`).
- **Risks:** an extra silently losing a dependency it used to get from core → audit every subpackage's
  third-party imports against its extra before committing; `mdtraj`, imported only inside
  `src/md_simulations/torch_potentials/cgschnet.py:275`, belongs in `torch-potentials` too, not only in `simulate`;
  `cgschnet-potentials` pulling openmm back through a chain of extras → read the resolved graph and assert
  it does not; `uv lock` moving torch or mlcg away from what the checkpoints were pickled against → the
  `mlcg` rev pin and the override/build-constraint entries stay byte-identical.
- **Stop if:** `uv lock` cannot resolve the new graph, or the lock moves `mlcg`, `torch` or `e3nn` — stop
  and report the diff rather than accepting it.

This is the ticket the user asked for: each of the package's two uses pays only for what it imports.

1. `dependencies = ["numpy>=1.26", "pydantic>=2.6", "pyyaml>=6.0"]`. Delete `parmed` (unused) and
   `ipykernel` (moves to the lab repo); `openmm`, `mdtraj`, `beamtime` and `tqdm` move to `simulate`.
2. New `simulate = ["openmm>=8.4.0.post2", "mdtraj>=1.10.3", "beamtime>=0.3.0", "tqdm>=4.67.1"]`, with the
   existing comment about why `beamtime` is pinned at a floor carried over verbatim.
3. `torch-potentials = ["torch>=2.0", "ase>=3.22", "mdtraj>=1.10.3"]` — deliberately no openmm; keep the
   existing comment about not pinning a torch index.
4. New `openmm-potentials = ["md-simulations[torch-potentials]", "md-simulations[simulate]"]`, documented
   as what a consumer needs to call `build_potential_from_file` on an AMBER/AWSEM config.
5. New `mcp = ["mcp>=1.0"]`, installed by the `simulations` repo and by no torch consumer (C3).
6. `analysis` gains `mdtraj`, `matplotlib>=3.8` and `scienceplots>=2.2.0` (its modules import them
   eagerly); new `prep = ["mdtraj>=1.10.3", "networkx>=3.0"]`; `cgschnet-dataset` keeps `h5py`.
7. `all` is rebuilt from the new names and keeps its `cu130` caveat comment.
8. `docs/lab-repo-pyproject.toml`: project `simulations`, depending on
   `md-simulations[simulate,analysis,viz,mcp,awsem,martini,cgschnet-potentials]` plus `ipykernel` and
   `jupyterlab`, with `[tool.uv.sources] md-simulations = { path = "../md-simulations", editable = true }`
   (the git URL in a comment for use elsewhere), and a copy of the `constraint-dependencies` and cu130
   index blocks per Decision 12.
9. README install section: four lines — the wrapper (`md-simulations[torch-potentials]`), the wrapper on
   OpenMM configs (`[openmm-potentials]`), running simulations (`[simulate]`) and the agent server (`[mcp]`).
10. `src/md_simulations/core/__init__.py`: replace P1's `try`/`except` around the eager imports with the
   `_LAZY` + `__getattr__` pattern of `torch_potentials/__init__.py`, keeping `EXTRA_HINT` and its
   wording. Every re-exported name is resolved on first access; nothing is imported at module scope, so
   `md_simulations.core.forces`, `core.freezing` and the other numpy-only submodules import under the
   `prep` and `cgschnet-dataset` extras. Audit the remaining `from md_simulations.core...` importers
   (`prep/migrate_forces.py`, `prep/cgschnet_dataset.py`, `sampling/umbrella.py`) against the extra each
   one belongs to; `umbrella.py` is `simulate`-side and may keep needing openmm.

## D. Documentation and handover

### D1. Split the docs along the new seam (sonnet) — code

- **Touches:** `README.md` (448 lines: `## Experiment log` 27, `## Starting a new system` 161,
  `## Layout` 437), `CLAUDE.md` (77 lines), new `docs/agents.md`, new `docs/lab-repo-README.md` (a draft for
  the other repo), new `docs/workspace-md-draft.md` (text for `../claude-instructions/workspace.md`).
- **Read:** `../claude-instructions/workspace.md`; `../claude-instructions/data.md`.
- **Check:** `rg -n 'gen-slurm|SlurmConfig|partition: a30' README.md CLAUDE.md` returns nothing;
  every relative link in `README.md` resolves (`rg -o '\]\(([^)h][^)]*)\)' -r '$1' README.md` then test each
  path exists).
- **Done:** `README.md` describes a package — install, engines, torch potentials, the CLI, `check-job` — and
  no longer documents the experiment log, the campaign or `configs/`; `CLAUDE.md` drops the `configs/*.yaml`
  and data-root bullets from "Project shape" and gains the workspace/site layer and the CLI contract;
  `docs/agents.md` states the agent contract in one page (enumerate with `md-sim --help`, author against
  `md-sim config schema`, verify with `config validate` and `check-job`, never write into the data root,
  never submit — hand the script to the user); the two draft files are ready to paste into the other repos.
- **Risks:** a README section that is really package documentation being moved out (engines, CGSchNet,
  differentiable potentials, analysis) → only `## Experiment log`, `## Starting a new system` and the
  campaign parts of `## Layout` move; deleting the replica-exchange and bond-repair docs, which are package
  behaviour → they stay.
- **Stop if:** a section is genuinely both — then it stays in the package README and the lab README links to it.

1. Move `## Experiment log` and `## Starting a new system` verbatim into `docs/lab-repo-README.md`, rewriting
   paths to the lab repo's layout and `gen-slurm` references to `check-job`.
2. `## Layout` in the package README loses `configs/`, `slurm/`, `data/`, `beamtime/`.
3. `CLAUDE.md`: in "Project shape", replace the `configs/*.yaml` and data-root bullets with the workspace
   descriptor, `sites/`, and a pointer to the `simulations` repo. Keep everything about engines, core/ and
   torch_potentials unchanged.
4. `docs/workspace-md-draft.md`: two sentences for `../claude-instructions/workspace.md` recording that
   `simulations` was split out of `md-simulations` on 2026-11-26 (full clone, then each half deleted its
   other side), that `md-simulations` is the package other projects import, and that `simulations` holds the
   configs, job scripts and the beamtime store.

### S3. The split script, for the user to run (sonnet + review) — code

- **Touches:** new `scripts/split-lab-repo.sh`, new `docs/split-runbook.md`.
- **Read:** `../claude-instructions/{workspace,data,git}.md`; this plan's Decisions 2 and 3 and Finding 7;
  the "Not verified" section.
- **Check:** `bash -n scripts/split-lab-repo.sh`; `shellcheck scripts/split-lab-repo.sh` if available;
  a dry run that creates the clone under `$TMPDIR` instead of `/remote/gpu04/hummerich/projects`
  (`SPLIT_PARENT=$TMPDIR/split-test bash scripts/split-lab-repo.sh --dry-run`) and prints every command it
  would run.
- **Done:** the script is idempotent-safe (refuses to run if the target exists), takes `--dry-run`, and in
  order: clones `md-simulations` to `<parent>/simulations`, removes `src/ tests/ examples/ docs/ uv.lock` there,
  replaces `pyproject.toml` with `docs/lab-repo-pyproject.toml` from P2 and commits, writes a new
  `md-workspace.yaml` and a new `sites/hd-gpu-cluster.yaml` there,
  moves the untracked `beamtime/` store across, replaces `data/` with a symlink to
  `/remote/gpu04/hummerich/data`, installs `docs/lab-repo-mcp.json` (C3) as `.mcp.json`, copies the three trivial
  slurm scripts that are worth keeping as new template examples and deletes the other 14 trivial ones, and
  leaves the 13 orchestration scripts in place. It never touches the `md-simulations` checkout.
- **Risks:** the clone picking up the *working tree* rather than `main` → clone from the path with an explicit
  `--branch main` and warn on uncommitted changes; the untracked `beamtime/` store being copied instead of
  moved and then diverging → move it, and fail loudly if the destination exists; `data/` being a real
  directory with bytes in it (Finding 8) → the script must refuse to replace a non-empty `data/` with a
  symlink and tell the user to move the contents first; the new repo having no remote → the runbook says the
  user creates the GitHub repo and pushes, the script only prints the commands.
- **Stop if:** the script would need to write anything inside `md-simulations` — that is S4's job, after the
  user confirms the clone exists.

**The sandbox cannot write outside this repo, so this ticket writes the script and stops. The user runs it.**
`docs/split-runbook.md` is the handover: preconditions, the command, what to check afterwards, and the exact
`gh repo create` / `git push` lines.

1. Shell, `set -euo pipefail`, `SPLIT_PARENT` defaulting to `/remote/gpu04/hummerich/projects`.
2. Preconditions: `md-simulations` has no uncommitted changes outside the ignored set; `<parent>/simulations`
   does not exist; `/remote/gpu04/hummerich/data` exists.
3. Each destructive step is echoed before it runs and skipped under `--dry-run`.
4. Copy `docs/lab-repo-pyproject.toml` (P2) in as the lab repo's `pyproject.toml`; the runbook tells the
   user to run `uv lock && uv sync` there afterwards, outside Claude Code.
5. `md-workspace.yaml` content exactly as in the Brief; the new site profile carries `partitions:
   [a30]` (extend by hand), `venv_activate: .venv/bin/activate`, `slurm_output: ./slurm/output`, `modules: []`.
6. Templates: keep `ala2-amber14-implicit-330K.sh` (plain single run), `cln-cgschnet-330K.sh` (cgschnet) and
   `trpcage-amber14-implicit-seeded-330K.sh` (a run with extra setup) as new `*.sh.example` template files in the lab repo.

### S4. Delete the lab half from the package (sonnet + review) — code

- **Touches:** `configs/` (82 files removed), `slurm/` (30 scripts removed), `.mcp.json` (removed),
  `data/README.md` (removed), `.gitignore` (the `data/*`, `beamtime/`, `slurm/output/`, `configs/generated/`
  rules removed), `README.md`, `CLAUDE.md`.
- **Read:** `docs/split-runbook.md` (from S3); this plan's "Breaking changes".
- **Check:** `.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check . &&
  env -u MD_DATA_ROOT .venv/bin/python -m pytest -q`; then in a scratch clone:
  `git clone . $TMPDIR/pkgcheck && cd $TMPDIR/pkgcheck && ls` shows no `configs/`, `slurm/`, `data/`,
  `beamtime/`; and the consumer invariant:
  `.venv/bin/python -c "import md_simulations, md_simulations.torch_potentials as tp; print(tp.__file__)"`.
- **Done:** the package repo contains only `src/ tests/ examples/ docs/ scripts/ pyproject.toml uv.lock
  README.md CLAUDE.md LICENSE .gitignore`; the full suite passes with `MD_DATA_ROOT` unset (cgschnet
  regression tests skipping); `grep -n 'md-simulations' ../split-flows/pyproject.toml` shows the git URL
  unchanged and `import md_simulations.torch_potentials` still resolves to the same module path.
- **Risks:** deleting before the user has actually run S3 → the ticket's first step verifies
  `/remote/gpu04/hummerich/projects/simulations/configs/ala2-amber14-implicit-330K.yaml` exists and stops if
  not; `examples/multiscale_ala2/` depending on `configs/` or `data/` → it ships its own
  `ala2-amber14-implicit-330K.yaml` and `data/`, confirm with
  `rg -n '\.\./\.\./(configs|data)' examples/`; a test still reading `REPO/configs` → S1 removed both, re-grep;
  `.gitignore` losing a rule something else needs (`*.dcd`, `core.[0-9]*`) → only the four lab-specific rules
  go.
- **Stop if:** the sibling `simulations` repo does not exist or its `configs/` is incomplete — stop and tell
  the user, deleting nothing.

This is the irreversible half, and it runs only after the user confirms the clone. Everything deleted here
exists in `simulations` with its full history.

1. Verify the sibling repo first, as the Risks line requires.
2. `git rm -r configs slurm .mcp.json data`.
3. `.gitignore`: remove the `data/*`, `!data/README.md`, `output/`, `slurm/output/`, `beamtime/` and
   `configs/generated/` rules. Keep the trajectory extensions (examples and tests still produce them), the
   Python and tooling caches, `core.[0-9]*` and `=*`.
4. README and CLAUDE.md: point at the `simulations` repo for configs, job scripts and the run log.

### D2. Agent instructions for the lab repo (sonnet) — code

- **Touches:** new `docs/lab-repo-CLAUDE.md`, `docs/lab-repo-README.md` (from D1),
  new `docs/lab-repo-job-skill.md`.
- **Read:** `docs/agents.md` (from D1); `/etc/claude-code/CLAUDE.md` (the standalone-`sbatch` rule);
  the memory that gpu04 has no Slurm client.
- **Check:** `rg -n 'md-sim ' docs/lab-repo-*.md` — every command named exists in `md-sim --help`.
- **Done:** the three drafts are complete and self-contained enough to be copied into `simulations/` by the
  user; the job-writing instructions state the whole loop (read the site profile, copy the nearest template example, write the script, run `md-sim check-job`, hand it to the user to submit —
  never call `sbatch`, which does not exist on gpu04); the CLAUDE.md draft states that the data root is
  read-only to agents and that every dataset needs a `datacard.yaml`.
- **Risks:** instructions that tell an agent to submit jobs → they must say the user submits; instructions
  duplicating what `md-sim --help` already says → link to the command, do not restate its flags.
- **Stop if:** the lab repo's own conventions turn out to need decisions you cannot infer — ask the user.

Drafts only; these files land in the other repo by hand, like `docs/workspace-md-draft.md`.

1. `docs/lab-repo-CLAUDE.md`: workspace layout, the site profile, data-root rules, the no-`sbatch` rule,
   where the beamtime store is, what belongs here versus in the package, and that `.mcp.json` registers
   both the beamtime server and `md-sim mcp` (C3) — with the reminder that the MCP tools author and check,
   while the user submits.
2. `docs/lab-repo-job-skill.md`: the write-a-job-script loop above, as a skill the user can drop into the lab repo's `.claude/skills` directory.

## Breaking changes

| What | Before | After | Migration |
|---|---|---|---|
| `md-sim gen-slurm` | Rendered an sbatch script from a config's `slurm:` block | Removed | Copy a `*.sh.example` template in the `simulations` repo, edit, run `md-sim check-job` |
| `SlurmConfig` / the `slurm:` config key | Exported from `md_simulations.config`; present in 74 configs | Removed; an unknown `slurm:` key is a validation error | W2 strips it from every config in the repo; any config elsewhere must drop the block |
| `md_simulations.slurm.render_slurm` | Public function | Removed | No replacement; `md-sim check-job` checks scripts instead of writing them |
| `configs/`, `slurm/`, `data/`, `beamtime/`, `.mcp.json` | In `md-simulations` | In `/remote/gpu04/hummerich/projects/simulations` | Clone the new repo; the package is now installed from, not run inside |
| `pip install md-simulations` | Brought openmm, mdtraj, beamtime, parmed, tqdm, ipykernel | Brings numpy, pydantic, pyyaml | Add an extra: `[simulate]` to run MD, `[torch-potentials]` for the wrapper, `[openmm-potentials]` for the wrapper on OpenMM configs |
| `md-simulations[torch-potentials]` | Included openmm and beamtime through core | torch + ase + mdtraj only | A consumer that calls `build_potential_from_file` on an AMBER/AWSEM config switches the pin to `[openmm-potentials]` |
| `parmed`, `ipykernel` | Runtime dependencies | Removed (parmed unused; ipykernel moves to the `simulations` repo) | Declare them yourself if a notebook needs them |
| `import md_simulations` | Imported openmm, beamtime and tqdm as a side effect | Imports none of them | Code relying on that side effect imports the module it actually needs |
| Data-root discovery | `MD_DATA_ROOT` / `--data-root` / config | Also `md-workspace.yaml`, between the env var and the config | None; existing precedence is unchanged |

## Not verified

- That `TypeAdapter(SimulationConfig).json_schema()` produces a usable `oneOf` for the discriminated union —
  C2 checks it first and stops if not.
- That all 74 `slurm:` blocks are trailing and strippable textually without touching a neighbouring key —
  W2's first step asserts the key sets before and after.
- That nothing outside this repo imports `md_simulations.slurm` or reads a config's `slurm:` key. Checked for
  split-flows in S4; other consumers are the user's to confirm.
- That beamtime's run store survives being moved to another directory — the store holds absolute `output_dir`
  paths in each run's config (`core/tracking.py:run_config`), and `beamtime/_moves.jsonl` suggests moves are
  tracked. S3 moves it; if beamtime objects, the store stays where it is and `run_store` in the descriptor
  points back at it.
- Which extras the consumers actually need: `../split-flows` is not readable from this sandbox, so the
  mapping from today's bare `md-simulations` pin to `[torch-potentials]` or `[openmm-potentials]` is the
  user's to confirm. P2 writes the migration line; S4 checks the git URL only.
- That `uv lock` resolves the re-layered extra graph without moving `mlcg`, `torch` or `e3nn` — P2 stops
  and reports if it does, and the lock is run by the user.
- That no module outside `engines/` and `core/` imports openmm or beamtime at top level — P1's
  `tests/test_packaging.py` is what establishes it, and it is written before the imports are moved.
- That `examples/multiscale_ala2/` is fully self-contained — S4 greps for escapes to `../../configs`.
