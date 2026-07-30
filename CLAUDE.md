# Python Style & Conventions

### Imports

Group imports in this order, separated by blank lines:
1. Standard library (`from typing import ...`, `from abc import ...`, `logging`, `os`)
2. Third-party (`numpy`, `openmm`, `mdtraj`, …)
3. Internal (`from md_simulations.core import ...`, `from md_simulations.config import ...`)

Always import frequently-used names explicitly rather than accessing them through the module:

```python
from pathlib import Path
import logging

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
```

### Type Annotations

- Use Python 3.10+ union syntax: `str | None`, `float | None` — never `Optional[str]`
- Use lowercase built-in generics: `dict[str, float]`, `list[int]`, `tuple[app.Topology, openmm.System]` — never `Dict`, `List`, `Tuple`
- Annotate every public function signature (parameters and return type)
- Use `Literal[...]` for string-valued enum parameters

### Docstrings

Every public class, method, and function gets a docstring. Format:

```python
def setup_system(pdb_file: str, forcefield_files: list[str]) -> tuple[app.Modeller, openmm.System]:
    """Short one-line description.

    :param pdb_file: Description of pdb_file.
    :param forcefield_files: Description of forcefield_files.
    :return: Description of the return value."""
```

- One-line summary first, then blank line, then `:param`/`:return:` lines
- Always end sentences with a period
- Class docstrings: one short line is enough

### Naming

- `snake_case` for all variables, functions, and modules
- Single uppercase letters for matrices: `M`, `R`, `Q`

### Modern Python Idioms

- Walrus operator for conditional assignments
- `match`/`case` for dispatching on string literals (e.g. engine selection)
- f-strings for all string formatting

### PyTorch Patterns

Applies to `src/md_simulations/torch_potentials/` (and any torch code elsewhere):

- Prefer `torch.cat` / `torch.einsum` / `torch.stack` over manual loops
- Keep the public torch API in **nm / kJ·mol⁻¹ / kJ·mol⁻¹·nm⁻¹**; convert at the
  single boundary in `bridge.py`, never per-calculator
- Torch imports stay out of the eager import path (see `torch_potentials/__init__.py`)

### Comments

Write comments only when the *why* is non-obvious: a hidden constraint, a numerical
workaround, or a subtle invariant. One short line is enough — no multi-line blocks.
Use a blank line, not a comment banner, to separate logical blocks.

### Testing

- Use `pytest` with fixtures in `conftest.py`
- Keep smoke tests tiny (a handful of MD steps) so the suite stays fast

## Tooling

- **Linter/formatter**: `ruff` (configured in `pyproject.toml`; excludes `tests/`, `.venv/`)
- **Package manager**: `uv`
- **Python**: 3.10–3.12 for the core package and all extras except `cgschnet`;
  the `cgschnet`/`mlcg` extra needs 3.12 (marker-gated, so it is unavailable on
  3.10/3.11). The dev env (`.python-version`) is 3.12 so cgschnet is available.

### Visualization
For visualization use matplotlib with the scienceplots theme:

```python
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "nature"])
```

The figure size should be taken from sciencplots: 3.3 x 2.5 inches for single-column figures, 6.9 x 2.5 inches for double-column figures:

```python
plt.figure(figsize=(3.3, 2.5))  # single-column figure
plt.figure(figsize=(6.9, 2.5))  # double-column figure
```

When saving figures, use pdf format and `bbox_inches="tight"` to avoid clipping labels:

```python
plt.savefig("figure.pdf", bbox_inches="tight")
```

In case some plots need rasterization, use `dpi=400`:

```python
plt.savefig("figure.pdf", dpi=400, bbox_inches="tight")
```

## Project shape

- `src/md_simulations/core/` — engine-agnostic building blocks (logging, integrators,
  equilibration, reporters, run loop). Engines depend on core, never the reverse.
- `src/md_simulations/engines/` — one module per force field / engine, each an `Engine`
  subclass building `(topology, system, positions)`.
- `src/md_simulations/config/` — typed pydantic config models loaded from YAML.
- `src/md_simulations/torch_potentials/` — differentiable torch potentials built from
  a config (`torch-potentials` extra). `bridge`/`potential`/`batch` need torch;
  `build`/`interop`/`calculators` do not, and the package `__init__` resolves the
  torch-backed names lazily so `import md_simulations` still works on a core
  install. New engines get an ASE calculator under `calculators/` — never a change
  to `Potential` or `bridge`.
- `configs/*.yaml` — one versioned config per simulated system.
- Simulation *data* lives outside the repo under a configurable data root
  (`MD_DATA_ROOT` env var or `--data-root`); never commit trajectories.
