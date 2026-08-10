# md-simulations

A standalone, config-driven molecular dynamics simulation toolkit built around
OpenMM (with native GROMACS support), designed to be reused across research
projects and run on a SLURM cluster.

## Features

- **Multiple engines**, one shared core (integrators, reporters, run loop):
  - `amber` — all-atom AMBER, explicit solvent (PME) or implicit GB
  - `pure_liquid` — NPT simulation of a pre-built periodic box (e.g. CHARMM36)
  - `martini` — MARTINI3 coarse-grained from GROMACS `.gro`/`.top`
  - `gromacs_input` — OpenMM run directly from GROMACS-format inputs (e.g. FreeSolv)
  - `awsem` — OpenAWSEM coarse-grained (optional `[awsem]` extra)
  - `gromacs_native` — native `gmx grompp`/`gmx mdrun` (optional `[gromacs]` extra)
- **One YAML config per system** instead of hardcoded CLI-arg lists.
- **Configurable data root** — trajectories live outside the repo.
- **Generated SLURM jobs** from each config.

## Install

```bash
uv sync                            # core (OpenMM) engines
uv sync --extra awsem              # + OpenAWSEM
uv sync --extra martini            # + vermouth/insane CG prep
uv sync --extra cgschnet           # + CGSchNet/mlcg ML CG (GPU; see below)
uv sync --extra torch-potentials   # + differentiable torch potentials (see below)
```

Everything installs into the single project `.venv` — there is no separate conda
env. The project targets **Python 3.12** (required by `mlcg`).

## Project layout

```
configs/        one YAML per simulated system (+ configs/mdp/ templates)
slurm/          generated SLURM batch scripts, one per config
slurm/output/   job stdout/stderr logs (git-ignored)
data/           default data root: inputs + trajectory outputs (git-ignored)
examples/       runnable end-to-end examples
src/md_simulations/   the installable package
```

The project hosts both the **SLURM scripts** (in `slurm/`) and the **simulation
data** (under `data/`). Trajectories and logs are git-ignored; the config YAMLs
and generated `.sh` scripts are tracked.

**Starting a new system:** copy the matching template from `configs/templates/`
(one per engine, every field commented) to `configs/<name>.yaml` and edit it.

## Usage

```bash
# Run a simulation described by a config (data root defaults to ./data)
md-sim run configs/cln-amber14.yaml

# Submit the pre-generated SLURM job (run from the project root)
sbatch slurm/cln-amber14.sh
```

Generated jobs `cd` to `$SLURM_SUBMIT_DIR`, activate `.venv`, and default
`MD_DATA_ROOT` to `./data` — so submit from the project root. Override the data
root anytime with `--data-root /path` or `export MD_DATA_ROOT=/path` (see
`data/README.md`). Outputs land in `<data_root>/<output_subdir>/` (e.g.
`cln/AMBER14_340K/`):

- `trajectory.dcd` — production trajectory
- `energies.csv`, `forces.txt` — per-frame potential energy and forces
- `initial_structure.pdb` — starting frame (written up front; use as the
  topology reference for visualising the trajectory)
- `final_structure.pdb` — last frame
- `simulation_summary.txt` — concise, publication-oriented parameter summary
  (force field, ensemble, thermostat/barostat, timestep, simulated time, …)

(Native-GROMACS runs write GROMACS `md.*` files plus `initial_structure.pdb`
and `simulation_summary.txt`.)

### Regenerating SLURM scripts

After editing a config's `slurm:` block, regenerate its job script:

```bash
md-sim gen-slurm configs/cln-amber14.yaml -o slurm/cln-amber14.sh
```

Each file in `configs/` is a versioned translation of one of the original
per-system SLURM scripts (ala2/ala3/ala6, chignolin, Trp-cage, Fs-peptide,
hexadecane, acetone/FreeSolv, …).

## Engines

Every engine shares the same run/output flow; a YAML picks the engine via
`engine:` and supplies its inputs. Start from a commented template
(`configs/templates/<engine>.yaml`) and run with `md-sim run configs/<name>.yaml`.

### `amber` — all-atom AMBER

Explicit solvent (PME, water added automatically) or implicit GB
(`implicit_solvent: true` + a GB force-field XML). Input: an all-atom PDB.

```bash
cp configs/templates/amber.yaml configs/cln.yaml   # set input_pdb, forcefield, temperature
md-sim run configs/cln.yaml
```

### `pure_liquid` — NPT of a pre-built box

For a fully periodic PDB (no solvation/hydrogens added), sampled at constant
pressure via a Monte Carlo barostat. Input: a periodic-box PDB.

```bash
cp configs/templates/pure_liquid.yaml configs/hexadecane.yaml   # set input_pdb, forcefield, pressure
md-sim run configs/hexadecane.yaml
```

### `martini` — MARTINI3 coarse-grained (OpenMM)

Runs a CG system from GROMACS `.gro` + `.top` (with a MARTINI ITP include dir),
NVT or NPT. Input: `.gro`, `.top`, `include_dir`.

```bash
cp configs/templates/martini.yaml configs/fsp-cg.yaml   # set input_gro, top, include_dir
md-sim run configs/fsp-cg.yaml
```

### `gromacs_input` — OpenMM from GROMACS inputs

Builds the OpenMM system directly from a deposited `.gro` + `.top` pair (e.g.
FreeSolv); no ForceField XML. Input: `.gro`, `.top`.

```bash
cp configs/templates/gromacs_input.yaml configs/acetone.yaml   # set input_gro, input_top
md-sim run configs/acetone.yaml
```

### `awsem` — OpenAWSEM coarse-grained

3-bead CG with implicit solvent (optionally associative-memory biased). Needs the
extra (`uv sync --extra awsem`). Input: an all-atom PDB (`prepare: true`
auto-converts to CG).

```bash
cp configs/templates/awsem.yaml configs/cln-awsem.yaml   # set input_pdb, chains, memories
md-sim run configs/cln-awsem.yaml
```

### `gromacs_native` — native `gmx grompp` + `gmx mdrun`

Drives GROMACS directly; run parameters (nsteps/dt/ensemble) come from the
`.mdp`. Needs a `gmx` binary on PATH (loaded via the job's `modules:` block).
Input: `.gro`, `.top`, `.mdp`.

```bash
cp configs/templates/gromacs_native.yaml configs/fsp-gmx.yaml   # set input_gro, top, mdp
md-sim gen-slurm configs/fsp-gmx.yaml -o slurm/fsp-gmx.sh
sbatch slurm/fsp-gmx.sh                                          # module-loads gromacs
```

### `cgschnet` — ML coarse-grained (CGSchNet / mlcg)

Runs the Clementi-group transferable CG model (Charron et al., *Nat. Chem.* 2025).
The model consumes `mlcg.AtomicData` and is **not** TorchScriptable, so it cannot
run through OpenMM-Torch's `TorchForce`; dynamics are driven by mlcg's own Langevin
integrator (`mlcg.simulation.LangevinSimulation`), and the numpy output is converted
into the same `trajectory.dcd` / `energies.csv` / `forces.txt` / PDB / summary files
as every other engine (model Å·kcal → output nm·kJ).

**Environment (uv-only — no conda):** the whole stack resolves as the `cgschnet`
extra, which mirrors mlcg's own CUDA-13 wheel setup (pinned `torch==2.11.0+cu130`,
PyG `torch-cluster`, `openmm-torch`). `mlcg` pins Python 3.12, so the project
targets 3.12.

```bash
uv sync --extra cgschnet          # into the project .venv (Python 3.12)
```

cgschnet SLURM jobs activate the same `.venv` as every other engine — no special
activation. (On a CUDA-12 host, switch the `cu130`/`cuda13` markers in
`pyproject.toml`'s `[tool.uv]` and `cgschnet` extra to `cu128`/`cuda12`.)

**Workflow — re-export the checkpoint, point the config at it, run:**

```bash
# 1. Re-export the distributed checkpoint for the current torch-geometric
#    (the released .pt is pickled against a pre-2.5 PyG and won't load as-is):
md-sim-cgschnet-reexport data/models/cgschnet/transferable_model_and_prior.pt \
    -o data/models/cgschnet/transferable_model_and_prior.reexport.pt
# 2. Point the config at the re-exported model + an mlcg configurations .pt
#    (a List[AtomicData] with initial positions, atom types, masses and prior
#    neighbor lists — the distributed per-protein artifact), then run:
md-sim run configs/trpcage-cgschnet-300K.yaml
```

The config needs `model_file` (re-exported), `configurations_file` (the mlcg
`List[AtomicData]`), and `input_pdb` (a CG PDB used only as the output topology; a
generic bead topology is synthesized if it does not match the bead count). `replica`
selects which configuration in the list starts the (first) trajectory.

**Throughput — set `n_replicas`.** A single CG molecule (~100 beads) badly
underuses the GPU, so per-step overhead dominates and wall-clock throughput is
poor. mlcg batches many independent replicas into one forward per step at nearly
the same per-step cost, so set `n_replicas` (e.g. to the number of starting
structures in `configurations.pt`) to multiply sampling per GPU-hour. Each replica
writes its own `trajectory_NN.dcd` / `energies_NN.csv` / `forces_NN.txt` /
`initial_structure_NN.pdb` / `final_structure_NN.pdb` (a single replica keeps the
plain names). Starting structures are tiled from the configurations list if
`n_replicas` exceeds its length (independent random velocities still decorrelate
the replicas).

## Differentiable potentials

`md_simulations.torch_potentials` turns an OpenMM-family or `cgschnet` config into a
**torch-differentiable** potential energy, so ML code can score configurations
against the same force field that defines a simulation — without re-specifying the
system. Needs the `torch-potentials` extra (`cgschnet-potentials` for the CG model).

```python
import mdtraj as md
import torch

from md_simulations.torch_potentials import build_potential_from_file

traj = md.load("data/pdbs/1UAO-cleaned.pdb")
pot = build_potential_from_file(
    traj.topology, "configs/cln-awsem-330K.yaml", data_root="data"
)

x = torch.tensor(traj.xyz[0], requires_grad=True)   # (N, 3) nm; or (F, N, 3)
energy = pot(x)                                     # scalar kJ/mol; or (F,)
energy.backward()
forces = -x.grad                                    # kJ/mol/nm
```

**Unit convention** (enforced across the whole torch API): positions in **nm**,
energy in **kJ·mol⁻¹**, forces in **kJ·mol⁻¹·nm⁻¹**. ASE calculators are eV/Å by
contract, so the conversion happens once, in `bridge.py`, at the single torch
boundary.

- **Batching** — pass `(F, N, 3)` to treat the frame axis as a torch batch axis
  and get `(F,)` back. For several *different* systems use
  `batched_potential_energy([(pot1, x1), (pot2, x2), ...])`.
- **Periodic systems** — pass `unitcell_lengths` (nm) and optionally
  `unitcell_angles` (degrees) alongside the positions; a single `(3,)` broadcasts
  across a batch.
- **Force groups** — `groups=` (a bitmask or set of indices) restricts the
  evaluation to a subset of OpenMM force groups, so you can score individual
  energy terms.
- **Gradients are first-order only** on the ASE path. The engine is opaque to
  autograd (`backward` reconstructs the gradient from the engine's forces), so there
  are no Hessians and no double-backward. The cgschnet potential below is the
  exception.
- **Adding an engine** — write an ASE `Calculator` under
  `torch_potentials/calculators/` and pass it to `Potential`; the bridge is
  engine-agnostic and neither it nor `Potential` needs changing.

Lower-level entry points: `build_calculator` / `build_calculator_from_file` return
the ASE calculator alone (no torch needed), and `Potential(topology, calculator)`
binds one yourself. See `examples/torch_potentials/`.

### Coarse-grained ML potentials (CGSchNet / mlcg)

`cgschnet` takes the other route. Its mlcg model is neither TorchScriptable nor an ASE
backend, but it *is* already a differentiable torch module, so `build_potential` returns
a native `CGSchNetPotential` instead of wrapping an ASE calculator. `build_calculator`
raises for this engine. Both types satisfy the `PotentialLike` protocol, so callers see
one interface.

```bash
uv sync --extra cgschnet-potentials     # needs Python 3.12: mlcg pins ==3.12.*
```

```python
pot = build_potential_from_file(
    cg_topology, "configs/trpcage-cgschnet-300K.yaml", data_root="data"
)
with torch.no_grad():
    energy = pot(R)                     # (B, 97, 3) nm -> (B,) kJ/mol
```

What differs from the ASE path:

- **Second derivatives work.** The distributed checkpoint wraps every term in an
  `mlcg.nn.gradients.GradientsOut` force head that calls `torch.autograd.grad` and then
  detaches `data.pos` — which raises under `torch.no_grad()` and disconnects every term
  after the first from the caller's positions. `unwrap_energy_models` strips those
  shells, leaving a plainly differentiable composition (and ~50× less work when only the
  energy is wanted).
- **Frames batch into one forward**, on-device, with no numpy round trip. Chunked at
  `max_batch_frames=64`; the 15 Å radius graph is rebuilt from the coordinates each
  forward, so cost grows with the frame count.
- **Bead order is checked, not assumed.** The prior interaction lists index bead
  *positions*, so a CG topology ordered differently from the model's own (e.g. a
  projected `N, CA, C, O, CB` against the model's `N, CA, CB, C, O`) would give silently
  wrong energies. The potential derives the permutation from `input_pdb` by matching
  `(residue index, atom name)` and applies it as an autograd op, so forces come back in
  the caller's order. An unresolvable topology raises rather than falling back.
- **Zero-interaction priors are pruned.** Terms for residue types absent from the
  molecule have an empty `index_mapping`, and mlcg's priors `scatter` without
  `dim_size`, so an empty term collapses the energy sum to length zero (18 of 51 terms
  for trp-cage). mlcg handles this with `specialize_priors`, which is unusable here
  because it bakes a fixed frame count into its static buffers.
- **Non-periodic.** Every prior neighbour list has `rcut=None`, so passing
  `unitcell_lengths` raises instead of being silently ignored.
- **Construction is expensive** (a ~390 MB unpickle). Build once and reuse.

`configurations_file` is required, not optional: it is the only source of the bead
embedding indices and the prior neighbour lists. Accuracy against mlcg's own simulation
output is ~2e-3 kJ/mol on energies and ~1e-4 relative on forces at float32
(`tests/torch_potentials/test_cgschnet_regression.py`).

## Other tools

- **Umbrella sampling** (specialised grid workflow, own CLI):
  ```bash
  md-sim-umbrella input.pdb --num-grid-points 36 --output-dir out
  ```
- **Prep tools** in `md_simulations.prep`: `martini_cg_map` (CG mapping matrix,
  needs the `martini` extra), `itp_bonds_to_pdb` (CONECT records from an ITP),
  and the vendored `olives` script.

## Analysis

Trajectory analysis lives in `md_simulations.analysis` (needs the `analysis`
extra: `uv sync --extra analysis`).

- **TICA** — featurise a trajectory (Cα distances and/or backbone torsions) and
  fit a time-lagged independent component model:
  ```bash
  md-sim-tica --pdb data/cln/AWSEM_330K/final_structure.pdb \
      --frames data/cln/AWSEM_330K/trajectory.dcd \
      --features ca_distances --lags 1000 --dim 2 \
      -o out --prefix cln --plot
  ```
  Fitting writes `<prefix>_model.pkl` and the `<prefix>_projection.npy` TIC
  coordinates; `--plot` saves the free-energy surface as `<prefix>_fes.pdf`. To
  project another trajectory onto that same space (e.g. a CG run against a
  reference), reuse the pickled model with `--tica-model out/cln_model.pkl`.

  Pass several `--lags` to compare free-energy surfaces across lag times in one
  multi-panel PDF (one model fitted per lag):
  ```bash
  md-sim-tica --pdb … --frames … --features ca_distances \
      --lags 100 500 1000 5000 --dim 2 -o out --prefix cln --plot
  ```

- **Collective variables** — compute structural CVs (RMSD to native, fraction
  of native contacts `native_contacts`, radius of gyration `rg`, end-to-end
  distance `end_to_end`) and plot them as 2-D free-energy surfaces:
  ```bash
  md-sim-cvs --pdb data/cln/AWSEM_330K/final_structure.pdb \
      --frames data/cln/AWSEM_330K/trajectory.dcd \
      --cvs native_contacts rmsd rg \
      -o out --prefix cln --plot
  ```
  CVs are saved to `<prefix>_cvs.npz` (one named array each) and `--plot` writes
  `<prefix>_fes.pdf`. With two CVs this is a single surface; with more, **every
  pairwise projection** becomes its own panel in the figure. `rmsd` and
  `native_contacts` need a reference: `--native` (defaults to `--pdb`). Tune the
  native-contact definition with `--q-cutoff` (nm) and `--q-min-seq-sep`.

See `CLAUDE.md` for code conventions.
