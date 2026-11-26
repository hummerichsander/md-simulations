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
  `uv sync --extra cgschnet --extra cu130 --torch-backend=cu130`; a consumer pinning
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
  `torch`, `ase`. `md_simulations.engines` and `md_simulations.core` are the `simulate`-extra half and
  raise `ModuleNotFoundError` naming that extra (`EXTRA_HINT`); import them inside the function that
  needs them, never at module scope in the core layer. `tests/test_packaging.py` pins this.

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
- `src/md_simulations/preflight.py` — `md-sim check-job`: `parse_script` reads `#SBATCH` and
  `md-sim run` out of a shell script, `check_job` checks them against the workspace, site and configs.
  Core-only imports. An undeterminable fact is `skipped`, never a pass.
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
- `configs/*.yaml` — one versioned config per simulated system.
- Simulation *data* lives outside the repo under a configurable data root
  (`MD_DATA_ROOT` env var or `--data-root`); never commit trajectories.
