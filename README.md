# md-simulations

Run molecular dynamics from a YAML file. One config describes a system — force field,
thermostat, how long — and the same command runs it locally or as a SLURM job, whether the
engine underneath is OpenMM, GROMACS, or a machine-learned coarse-grained model. The point
is that switching engines doesn't mean rewriting your setup.

```bash
uv sync
md-sim run configs/cln-amber14-implicit-330K.yaml     # locally
sbatch slurm/cln-amber14-implicit-330K.sh             # or on the cluster
```

Every run writes the same files to `<data_root>/<output_subdir>/`, regardless of engine:

| file | what it is |
| --- | --- |
| `trajectory.dcd` | the production trajectory |
| `energies.csv`, `forces.txt` | per-frame potential energy and forces |
| `initial_structure.pdb` | first frame, written up front — use it as the topology for viewing the DCD |
| `final_structure.pdb` | last frame, and the restart point for a follow-up run |
| `simulation_summary.txt` | force field, ensemble, thermostat, timestep, simulated time |

Data lives outside the repo. `MD_DATA_ROOT` (or `--data-root`) sets the root and defaults
to `./data`; configs and generated job scripts are tracked, trajectories are not.

## Starting a new system

Copy the template for your engine — every field is commented — then edit and run:

```bash
cp configs/templates/amber.yaml configs/mysystem.yaml
md-sim run configs/mysystem.yaml
```

For a cluster job, generate the batch script from the config's `slurm:` block. Rerun this
after changing that block, since the `.sh` is generated, not live:

```bash
md-sim gen-slurm configs/mysystem.yaml -o slurm/mysystem.sh
```

Generated jobs `cd` to `$SLURM_SUBMIT_DIR` and activate `.venv`, so submit from the project
root.

## Engines

Pick one with `engine:` in the config. All of them share the same integrators, reporters
and output files, so the table is mostly about what inputs each needs.

| engine | runs | inputs | needs |
| --- | --- | --- | --- |
| `amber` | all-atom AMBER, explicit (PME) or implicit GB | all-atom PDB | — |
| `pure_liquid` | NPT of a pre-built periodic box (e.g. CHARMM36) | periodic PDB | — |
| `martini` | MARTINI3 coarse-grained under OpenMM | `.gro`, `.top`, ITP dir | `--extra martini` |
| `gromacs_input` | OpenMM built straight from GROMACS inputs (e.g. FreeSolv) | `.gro`, `.top` | — |
| `awsem` | OpenAWSEM 3-bead CG, optionally memory-biased | all-atom PDB | `--extra awsem` |
| `gromacs_native` | real `gmx grompp` + `gmx mdrun` | `.gro`, `.top`, `.mdp` | `gmx` on PATH |
| `cgschnet` | ML coarse-grained (CGSchNet / mlcg) | see below | `--extra cgschnet` |

Two behave a little differently from the rest. `awsem` will convert an all-atom PDB to CG
for you with `prepare: true`. `gromacs_native` takes its run parameters from the `.mdp`
rather than the YAML, so `num_steps` and friends are ignored there — and it needs `gmx`,
which usually means adding a `modules:` block so the job loads it.

## The CGSchNet engine

This one is worth its own section because it is genuinely different: it runs the
Clementi-group transferable CG model (Charron et al., *Nat. Chem.* 2025), and that model is
not TorchScriptable, so it can't go through OpenMM-Torch. Dynamics come from mlcg's own
Langevin integrator instead, with the output converted into the usual files (the model
works in Å and kcal/mol; you still get nm and kJ/mol).

mlcg pins Python 3.12, which is why the whole project targets 3.12. The extra mirrors
mlcg's CUDA-13 wheel setup:

```bash
uv sync --extra cgschnet
```

On a CUDA-12 host, switch the `cu130`/`cuda13` markers in `pyproject.toml` to
`cu128`/`cuda12`. Jobs activate the same `.venv` as every other engine.

The released checkpoint was pickled against a pre-2.5 torch-geometric and won't load as-is,
so re-export it once:

```bash
md-sim-cgschnet-reexport data/models/cgschnet/transferable_model_and_prior.pt \
    -o data/models/cgschnet/transferable_model_and_prior.reexport.pt
md-sim run configs/trpcage-cgschnet-300K.yaml
```

The config wants three things: `model_file` (re-exported), `configurations_file`, and
`input_pdb`. `configurations_file` is not optional — that mlcg `List[AtomicData]` is the
only place the bead embedding indices and prior neighbour lists come from. `input_pdb`, by
contrast, only labels the output trajectory.

**Set `n_replicas`.** One CG molecule is ~100 beads, which leaves the GPU almost idle and
lets per-step overhead dominate. mlcg batches independent replicas into a single forward at
nearly the same cost per step, so raising `n_replicas` multiplies your sampling per
GPU-hour essentially for free. A good default is the number of starting structures in
`configurations.pt`. Each replica writes its own suffixed files; starting structures are
reused if you ask for more replicas than there are configurations, with random velocities
to decorrelate them.

### Fine-tuning on your own trajectory

The transferable model is an average over many proteins. If you have an all-atom trajectory
of *your* system, you can teach the model its physics instead. That means force matching,
which needs coarse-grained coordinates paired with the forces that belong to them.
Projecting coordinates is easy; projecting forces is where people lose an afternoon:

```bash
uv sync --extra cgschnet-dataset
md-sim-cgschnet-dataset --run data/trp-cage/AMBER14_implicit_300K \
    --bead-pdb data/pdbs/2JOF_5B-CG.pdb --out data/trp-cage/fm_300K \
    --cgschnet-config configs/trpcage-cgschnet-300K.yaml --name trpcage
```

Out comes an `.h5` that mlcg's training script reads directly, plus the same arrays as
`.npy` if you want to poke at them. Forces stream from disk in blocks, so a 341 MB
`forces.txt` never has to fit in memory.

Three things to know:

*Read the constraint count it prints.* It reports how many rigid bonds it found and checks
that against what OpenMM actually applied — on trp-cage both say 136. That agreement is the
cheapest possible confirmation the projection is sound. If you see something like 602, you
handed it nanometres: aggforce thinks in Ångström and decides "constrained" by how little a
distance fluctuates, so nanometre input makes every stiff bond look frozen. One silent
factor of ten and every force in the dataset is wrong.

*Don't agonise over `--force-map`.* Any map obeying the consistency condition gives the same
answer on average, so a "wrong" pick can't bias what the model learns — only make the
training signal noisier. The default `slice_optimize` is about a third quieter than
`slice_aggregate` on trp-cage, which helps when frames are scarce, but both converge to the
same force field.

*The network only learns the leftovers.* Of the 52 terms in the checkpoint, only SchNet has
parameters; the other 51 are fixed analytic priors for bonds, angles and torsions. So the
useful target isn't the CG force but whatever the priors get wrong. With
`--cgschnet-config` the tool evaluates the priors, subtracts them, and stores the remainder
as `cg_delta_forces`. On trp-cage the priors already cover 82 % of the force (RMS 22.4 →
9.5 kcal/mol/Å) — which doubles as a sanity check, since delta forces nearly as large as
the total mean something upstream is off.

For training, start from mlcg's worked example in `examples/h5_pl/single_molecule/`
([mlcg](https://github.com/ClementiGroup/mlcg)). Conveniently it's trp-cage as well, at the
same 97 beads, and the `.h5` here uses the same layout, so its `training.yaml` needs almost
no editing. Upstream also maintains [mlcg-tk](https://github.com/ClementiGroup/mlcg-tk/)
for dataset prep — worth comparing before a large campaign.

## Differentiable potentials

Sometimes you want to *score* configurations with the same force field that defines a
simulation, from inside ML code, without respecifying the system.
`md_simulations.torch_potentials` turns any config into a torch-differentiable energy:

```python
import mdtraj as md
import torch

from md_simulations.torch_potentials import build_potential_from_file

traj = md.load("data/pdbs/1UAO-cleaned.pdb")
pot = build_potential_from_file(traj.topology, "configs/cln-awsem-330K.yaml", data_root="data")

x = torch.tensor(traj.xyz[0], requires_grad=True)   # (N, 3) nm, or (F, N, 3)
energy = pot(x)                                     # kJ/mol, or (F,)
energy.backward()
forces = -x.grad                                    # kJ/mol/nm
```

Positions are always **nm**, energies **kJ/mol**, forces **kJ/mol/nm**, everywhere in this
API. ASE calculators speak eV/Å, so that conversion happens once, at the single torch
boundary in `bridge.py`.

Hand it a whole trajectory instead of one frame and the frame axis just becomes a batch
axis, `(F, N, 3)` in and `(F,)` out. For several *different* systems at once there's
`batched_potential_energy([(pot1, x1), (pot2, x2), ...])`. Periodic boxes take
`unitcell_lengths` in nm alongside the positions, and `groups=` narrows the evaluation to a
subset of OpenMM force groups when you want to see one energy term at a time.

The one real limitation: on the ASE path you get first derivatives and nothing more. The
engine is opaque to autograd, so `backward` reconstructs the gradient from the forces the
engine reports, which means no Hessians and no double-backward. CGSchNet, below, is the
exception.

Adding an engine is a small job — write an ASE `Calculator` under
`torch_potentials/calculators/` and pass it to `Potential`; nothing else changes. If you
want the calculator without torch in the picture at all, `build_calculator` gives you that.
Working examples live in `examples/torch_potentials/`.

### The CGSchNet potential

CGSchNet takes the other route: its model is neither TorchScriptable nor an ASE backend,
but it *is* already a differentiable torch module, so you get a native `CGSchNetPotential`
rather than an ASE wrapper (`build_calculator` raises here). Both satisfy the same
`PotentialLike` protocol, so callers can't tell.

```bash
uv sync --extra cgschnet-potentials
```

```python
pot = build_potential_from_file(
    cg_topology, "configs/trpcage-cgschnet-300K.yaml", data_root="data"
)
with torch.no_grad():
    energy = pot(R)                     # (B, 97, 3) nm -> (B,) kJ/mol
```

Unlike the ASE path, this one gives you second derivatives — Hessian-vector products and
all. Getting there took some surgery: the checkpoint wraps every term in a `GradientsOut`
force head that detaches `data.pos`, which blows up under `no_grad()` and quietly
disconnects every term after the first from your input. `unwrap_energy_models` strips those
shells off, which leaves a plainly differentiable composition and, as a bonus, about 50×
less work when all you wanted was the energy.

The trap here is bead order. The model's prior interaction lists index bead *positions*, not
names, so handing it a topology ordered `N, CA, C, O, CB` when the model expects
`N, CA, CB, C, O` produces energies that are wrong without looking wrong. Rather than trust
you to get it right, the potential works out the permutation itself from `input_pdb` by
matching residue and atom name, and applies it as an autograd op so forces come back in your
ordering. A topology it can't resolve raises instead of guessing.

Two smaller things. Priors for residue types your molecule doesn't contain end up with an
empty interaction list, and because mlcg's priors `scatter` without `dim_size` a single
empty term collapses the entire energy sum — so they get pruned (18 of 51 terms for
trp-cage). And the model is non-periodic, so passing `unitcell_lengths` raises rather than
being silently dropped.

Build the potential once and hold onto it: construction unpickles roughly 390 MB. After
that, frames batch into one on-device forward with no numpy round trip, chunked at
`max_batch_frames=64`, though the 15 Å radius graph is rebuilt every forward so cost does
grow with frame count. Against mlcg's own simulation output it agrees to ~2e-3 kJ/mol on
energies and ~1e-4 relative on forces at float32
(`tests/torch_potentials/test_cgschnet_regression.py`).

## Analysis

Needs `uv sync --extra analysis`.

**TICA** — featurise a trajectory (Cα distances, backbone torsions) and fit a time-lagged
independent component model:

```bash
md-sim-tica --pdb data/cln/AWSEM_330K/final_structure.pdb \
    --frames data/cln/AWSEM_330K/trajectory.dcd \
    --features ca_distances --lags 1000 --dim 2 -o out --prefix cln --plot
```

You get `<prefix>_model.pkl`, the `<prefix>_projection.npy` TIC coordinates, and with
`--plot` a free-energy surface. Reuse the pickled model via `--tica-model` to project a
second trajectory into the same space — that's how you compare a CG run against its
all-atom reference. Passing several `--lags` fits one model per lag and puts the surfaces
side by side in one PDF.

**Collective variables** — RMSD to native, fraction of native contacts, radius of gyration,
end-to-end distance:

```bash
md-sim-cvs --pdb data/cln/AWSEM_330K/final_structure.pdb \
    --frames data/cln/AWSEM_330K/trajectory.dcd \
    --cvs native_contacts rmsd rg -o out --prefix cln --plot
```

CVs land in `<prefix>_cvs.npz`. Two CVs give one free-energy surface; more gives every
pairwise projection as its own panel. `rmsd` and `native_contacts` need a reference
(`--native`, defaulting to `--pdb`), and `--q-cutoff` / `--q-min-seq-sep` tune the contact
definition.

## Other tools

- **Umbrella sampling**, a specialised grid workflow with its own CLI:
  `md-sim-umbrella input.pdb --num-grid-points 36 --output-dir out`
- **Prep helpers** in `md_simulations.prep`: `martini_cg_map` (CG mapping matrix, needs the
  `martini` extra), `itp_bonds_to_pdb` (CONECT records from an ITP), and the vendored
  `olives` script. The CGSchNet ones — `cgschnet_map`, `cgschnet_embeddings`,
  `cgschnet_reexport`, `cgschnet_dataset` — are covered above.

## Layout

```
configs/              one YAML per system, plus templates/ and mdp/
slurm/                generated batch scripts; output/ holds job logs
data/                 default data root (git-ignored)
examples/             runnable end-to-end examples
src/md_simulations/   the package
```

See `CLAUDE.md` for code conventions.
