# md-simulations

Shared conventions come from `../claude-instructions` (loaded via `../CLAUDE.md`). Additions and
overrides for this project:

- **Python**: 3.10–3.12 for the core package and all extras except the `cgschnet`
  family; `cgschnet`, `cgschnet-potentials` and `cgschnet-dataset` all pull `mlcg`,
  which needs 3.12 (marker-gated, so it is unavailable on 3.10/3.11). The dev env
  (`.python-version`) is 3.12 so all three are available.
- The `cgschnet` extras depend on **plain `mlcg`** and leave the CUDA line to the
  top-level project; `mlcg[cu130]` (torch 2.11 + the CUDA-13 wheel indexes) is the
  separate, orthogonal `cu130` extra. Locally that means
  `uv sync --extra cgschnet --extra cu130` (the extra pins the CUDA-13 wheel indexes,
  so no `--torch-backend` flag is needed -- and `uv sync` does not accept one); a consumer pinning
  its own torch (e.g. split-flows at torch 2.9.1/cu128) takes `cgschnet` alone. Never
  put `mlcg[cu130]` back into a `cgschnet*` extra — that coupling is what forced
  split-flows into a second venv.
- **`platform: CUDA` needs the `openmm-cuda` extra.** The plain `openmm` wheel on PyPI
  ships only Reference/CPU/OpenCL, so `Platform.getPlatformByName("CUDA")` raises and a
  GPU job silently has no device. `openmm-cuda` adds `openmm[cuda13]` (the separate
  OpenMM-CUDA-13 wheel) and is deliberately *not* folded into `cu130` alone, since the
  AMBER/AWSEM engines want the CUDA platform without any torch:
  `uv sync --extra openmm-cuda`. Anything running MD on a GPU — every `--gres=gpu:1`
  slurm wrapper — needs it; `cu130` includes it for the full CUDA-13 stack.
  `[tool.uv] constraint-dependencies` holds nvrtc/nvcc below 13.2 to match the cluster
  driver's CUDA 13.1; without that the newest toolkit emits PTX the driver cannot JIT
  and `Context` creation dies with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)`.
- Typical third-party imports are `numpy`, `openmm`, `mdtraj`; internal ones are
  `from md_simulations.core import ...`, `from md_simulations.config import ...`:

  ```python
  from pathlib import Path
  import logging

  import numpy as np
  import openmm
  import openmm.app as app
  import openmm.unit as unit
  ```

- Keep smoke tests tiny (a handful of MD steps) so the suite stays fast.

### Torch potentials

Applies to `src/md_simulations/torch_potentials/` (and any torch code elsewhere):

- Keep the public torch API in **nm / kJ·mol⁻¹ / kJ·mol⁻¹·nm⁻¹**. Every conversion
  factor is *defined* in `bridge.py`, never per-calculator. ASE-backed engines convert
  inside `bridge.EnergyFn`; native-torch engines apply the factors as autograd ops at
  their own module boundary, so their force conversion is derived rather than
  hand-written
- Torch imports stay out of the eager import path (see `torch_potentials/__init__.py`)
- So do openmm and beamtime: `import md_simulations` (and `md_simulations.torch_potentials`,
  `config`, `workspace`, `preflight`, `cli`) imports none of `openmm`, `beamtime`, `tqdm`, `mdtraj`,
  `torch`, `ase`. `md_simulations.engines` and the openmm/beamtime-backed names of
  `md_simulations.core` are the `simulate`-extra half and raise `ModuleNotFoundError` naming that
  extra (`EXTRA_HINT`); import them inside the function that needs them, never at module scope in
  the core layer. `core/__init__` resolves its names lazily (`_LAZY` + `__getattr__`, like
  `torch_potentials`), so the numpy-only submodules (`core.forces`, `core.replica_exchange`,
  `core.logging`) import without openmm: `prep/migrate_forces.py` and `prep/cgschnet_dataset.py`
  depend on that. A new `core` name goes into both `__all__` and `_LAZY`. `core.freezing`,
  `integrators`, `equilibration`, `reporters`, `runner`, `summary`, `remd` and `tracking` need openmm
  (or beamtime/tqdm). `tests/test_packaging.py` pins this.
- **Extras**: `[project.dependencies]` is `numpy`, `pydantic`, `pyyaml` and nothing else. `simulate`
  (openmm, mdtraj, beamtime, tqdm) is the MD half; `torch-potentials` (torch, ase, mdtraj) has no
  openmm and no beamtime; `openmm-potentials` is both; `prep`, `analysis`, `viz` and `mcp` carry what
  their modules import. A new third-party import goes into the extra of the module that makes it,
  never back into `dependencies`. Tests need `simulate`: `uv sync --extra simulate`.

## Project shape

- `src/md_simulations/core/` — engine-agnostic building blocks (logging, integrators,
  equilibration, reporters, run loop, Beamtime tracking). Engines depend on core, never the
  reverse.
- Every engine's `run()` opens its Beamtime run(s) (`core/tracking.py`) and hands them to
  `simulate()`. Hierarchy: system group → simulation run, or replica-set group → `rNN` runs.
  Batched CGSchNet replicas use beamtime's internal `new_run`, since `beamtime.init` allows
  one live run per process; that is why `beamtime` is pinned exactly.
- `src/md_simulations/engines/` — one module per force field / engine, each an `Engine`
  subclass building `(topology, system, positions)`.
- `src/md_simulations/config/` — typed pydantic config models loaded from YAML.
- `src/md_simulations/workspace.py` — `md-workspace.yaml` (`Workspace`) and its site profile
  `sites/<site>.yaml` (`Site`): where configs, data and the run store live and which cluster this is.
  Nothing else in `src/` may name a partition, a venv path or a data directory. Exported from
  `md_simulations.workspace`, not from `config`.
- `src/md_simulations/cli/` — the `md-sim` noun/verb dispatcher. `main.py` parses `run` and `check-job`
  itself and routes the `config` group to `config_cmds.py` (`schema|validate|describe|ls|new`, a
  `verb -> (summary, handler)` table, core-only, no alias); `config describe` lists input paths by
  the field names in its `_INPUT_FIELDS`, so a new path-valued config field is added there; `prep`, `analysis` and `sampling` pass everything after the verb to
  the standalone script via `_dispatch.py`, whose table holds `module:function` *strings* (a verb's
  module is imported only when chosen, so help and listings load no torch/openmm) and sets
  `sys.argv[0]` to `md-sim-<verb>` so usage lines match the alias. `_output.py` is the shared
  `--json` envelope (`{"ok": true, ...}` / `{"ok": false, "error": {"type", "message"}}`; exit
  0 ok, 1 failure, 2 usage). The 14 `md-sim-*` entry points in `pyproject.toml` stay as aliases of
  the same mains, and `tests/test_cli.py` pins them against the dispatch table. A new script gets
  an alias, a `GROUPS` entry and a `SUMMARIES` line.
- `src/md_simulations/preflight.py` — `md-sim check-job`: `parse_script` reads `#SBATCH` and
  `md-sim run` out of a shell script, `check_job` checks them against the workspace, site and configs.
  Core-only imports. An undeterminable fact is `skipped`, never a pass.
- `src/md_simulations/mcp_server.py` — `md-sim mcp`: six read-and-author tools (`list_configs`,
  `describe_config`, `validate_config`, `config_schema`, `new_config`, `check_job`) that run the
  matching `md-sim ... --json` command in-process (`run_cli`, stdout captured under `_INVOKE_LOCK`,
  since the SDK runs a sync tool in a worker thread) and return its envelope
  plus `"workspace"`, so the CLI stays the one surface. Core-only imports; the `mcp` SDK (the `mcp`
  extra) is imported inside `build_server`, and its absence is `McpUnavailable` (exit 1). A new tool
  is a `make_tools` entry plus a `TOOL_DESCRIPTIONS` line; never one that runs or submits a job.
  `new_config` refuses any output outside `config_dir`. The lab repo installs its own git-ignored
  `.mcp.json` that starts this server beside the beamtime one.
- `src/md_simulations/torch_potentials/` — differentiable torch potentials built from
  a config (`torch-potentials` extra). `bridge`/`potential`/`batch` need torch;
  `build`/`interop`/`calculators` do not, and the package `__init__` resolves the
  torch-backed names lazily so `import md_simulations` still works on a core
  install. New engines get an ASE calculator under `calculators/` — never a change
  to `Potential` or `bridge`. The exception is an engine whose native representation
  is *already* a differentiable, batched torch module: `cgschnet` therefore ships a
  `PotentialLike` class in `cgschnet.py`, dispatched from `build_potential`, and still
  touches neither `Potential` nor `bridge`. It imports torch only — mlcg arrives via
  `torch.load` at build time, so the module stays importable without the extra.
- This repo is the package only: `src/`, `tests/`, `examples/`, `docs/`. The per-system configs,
  the orchestration slurm scripts, the Beamtime store, `md-workspace.yaml` and `sites/<site>.yaml`
  live in the sibling `simulations` repo (`../simulations`), which installs this package and keeps the
  simulation data in its own gitignored `data/` directory (the shared root `/remote/gpu04/hummerich/data`
  holds only derived datasets). Nothing here may default a data root
  (precedence is `--data-root` > `MD_DATA_ROOT` > workspace > config) or name a partition, a venv
  or a data directory. Never commit trajectories.
- **CLI contract** (`docs/agents.md`): `md-sim` is one discoverable noun/verb command, every
  command takes `--json` and answers with the `_output.py` envelope, and an agent authors against
  `md-sim config schema`, verifies with `config validate` and `check-job`, never writes into the
  data root and never submits a job. A new command keeps that contract and is added to the dispatch
  table, so `md-sim --help` and `md-sim --json` list it.
