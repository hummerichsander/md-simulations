# md-simulations

Run molecular dynamics from a YAML file. One config describes a system — force field,
thermostat, how long — and the same command runs it locally or inside a job script, whether the
engine underneath is OpenMM, GROMACS, or a machine-learned coarse-grained model. The point
is that switching engines doesn't mean rewriting your setup.

This is the package. It holds the engines, the typed config models, the differentiable torch
potentials and the `md-sim` command; it contains no configs for particular systems, no job
scripts and no knowledge of any cluster. A research repo that runs campaigns with it keeps those
itself (see [Workspaces](#workspaces)).

```bash
uv sync
md-sim run my-system.yaml --data-root /path/to/data
```

Every run writes the same files to `<data_root>/<output_subdir>/`, regardless of engine:

| file | what it is |
| --- | --- |
| `trajectory.dcd` | the production trajectory |
| `energies.csv`, `forces.npy` | per-frame potential energy and forces |
| `initial_structure.pdb` | first frame, written up front — use it as the topology for viewing the DCD |
| `final_structure.pdb` | last frame, and the restart point for a follow-up run |
| `simulation_summary.txt` | force field, ensemble, thermostat, timestep, simulated time |

Data lives outside the package and outside any repo that uses it. The data root is set by
`--data-root`, `MD_DATA_ROOT`, the workspace descriptor or the config's `data_root`, in that order
of precedence; nothing defaults it, and a run with none of them exits with an error naming all four.
Trajectories are never committed.

Every `md-sim run` is also logged with [Beamtime](https://hummerichsander.github.io/beamtime/): the
config, the output directory, energies, temperature and speed every `report_interval` steps, the
console output and references to the output files. The optional `beamtime:` block of a config
(`group`, `name`) sets where the run lands; `md-sim config schema` lists it. Where the store lives
and how the groups are laid out is up to the repo running the campaign.

## Workspaces

A repo that runs many systems can state its layout once in an `md-workspace.yaml` (`config_dir`,
`data_root`, `run_store`, `site`), found by walking up from the cwd to the git root or named by
`$MD_WORKSPACE`; relative paths resolve against the descriptor's directory and `${VAR}` /
`${VAR:-default}` expand from the environment (`data_root: ${MD_DATA_ROOT:-./data}`). Cluster facts
(partitions, venv activation, modules) live in `sites/<site>.yaml` beside it, never in the package.
`md-sim config ls`, `md-sim check-job` and `md-sim mcp` read it; see `md_simulations.workspace`.
Hand-written configs and job scripts belong in that repo, not here.

## Replica exchange

Any OpenMM engine (`amber`, `pure_liquid`, `martini`, `gromacs_input`, `awsem`) can run
temperature replica exchange by adding a `replica_exchange:` block — `md-sim config schema --engine amber` lists its fields. One replica runs per temperature of a ladder (explicit, or
geometric from `temperature` to `t_max`), and every `exchange_interval` steps neighbouring
temperatures attempt to swap. Swaps move the temperature, not the coordinates: the replicas'
velocities are rescaled by `sqrt(T_new / T_old)`. The system must be NVT.

`temperature` is the target (lowest) rung. Its files are written into the output directory exactly
as for a plain run (`trajectory.dcd`, `energies.csv`, `forces.npy`, the two PDBs and
`simulation_summary.txt`, which gains a replica-exchange section), so downstream code does not
change. The other temperatures write into `T01_360K/`, ... as `other_output` says, and
`exchanges.csv` has one row per attempt: who sits at each temperature and which pairs swapped.
Beamtime gets one run per temperature below `<group>/<name>`, each logging its swap acceptance
with the next temperature up, and the global round-trip count.

`checkpoint_interval` writes `checkpoint/`. A rerun that finds one stops rather than overwrite it;
continue with `md-sim run x.yaml --resume`. OpenMM checkpoints belong to the platform and
OpenMM version that wrote them, so resume on the same kind of GPU node. The Beamtime runs are
continued as a new attempt, not duplicated. Replicas currently run one after another in one
process, so a job costs about `n_replicas` plain runs of GPU time.

## Conditional sampling (frozen atoms)

`frozen_atoms:` holds a subset of the atoms immobile, so the run samples the Boltzmann
distribution of the others *conditioned* on the frozen coordinates — sidechains at a fixed
backbone, say. Select by atom name (matched against the built topology, so hydrogens added at
build time are covered) or by index, and set `expect_count` to fail fast when the selection
resolves against a topology it was not written for:

```yaml
constraints: none
timestep: 0.0005
frozen_atoms:
  names: [N, CA, CB, C, O]
  expect_count: 47
```

Freezing is by zero mass, which drops those atoms from the equations of motion exactly; a stiff
positional restraint would instead leave a thermal smear of width `sqrt(kT/k)` on the very
coordinates the conditional is defined by, and would add its energy to `forces.npy`.

The catch is constraints: OpenMM refuses, at context creation, any constraint involving a massless
particle, because SHAKE/CCMA split the correction between two mobile partners in proportion to
inverse mass and a frozen partner can absorb none of it. Any frozen heavy atom bearing a hydrogen
therefore forces `constraints: none`, hence explicit X–H stretches (~11 fs period) and a ~0.5 fs
timestep. The build-time check names the offending bond rather than letting the job fail on the
GPU. The dof count, the reported temperature and `simulation_summary.txt` all account for the
frozen atoms, and the centre-of-mass motion remover is dropped along with them.

Comparing such a run against an ensemble produced with `constraints: hbonds` means crossing a
convention: flexible dynamics and the resampler's independent-harmonic draw are *not* the same
density (the chignolin fiber pilot measured a 3 % narrower X–H width under flexible dynamics).
Put the reference in the unconstrained convention first — see the next section.

## Repairing constrained bonds

A trajectory run with `constraints: hbonds` has every X–H bond pinned at exactly its equilibrium
length, so those coordinates carry no thermal variance — measured std 0.40 pm, against an XTC
quantization floor of 0.41. Anything that reweights a generative model against the all-atom
potential then pays `d / 2` nats² of log-weight variance (8.1 nats for trp-cage's 136 bonds) for
coordinates the model generates and the constrained Hamiltonian does not score at all. Two tools
repair that, both taking the bond set from the *constrained twin* of the config they are given —
so neither assumes "the hydrogens", and `--constrained-as allbonds` works the same way.

| | `md-sim-resample-hbonds` | `md-sim-equilibrate-hbonds` |
| --- | --- | --- |
| what it does | draws each length from `N(r0, sqrt(kT/k))` and slides the H along its own bond axis | integrates the free ends in the unconstrained potential, every other atom frozen at zero mass |
| distribution | the bond treated as an isolated harmonic oscillator | the true conditional `p(x_H \| x_heavy)`, angles and nonbonded terms included |
| cost | seconds for a whole trajectory, one numpy call | hours; a sharded array job |
| leaves alone | every other coordinate, exactly | every other coordinate, exactly (zero mass, so bit-identical) |

The draw is exact to the extent that the length decouples from everything else, which it nearly
does — the residual coupling measures 650 against the bond's own 285000 kJ/mol/nm² — but not
quite: the two differ by a few percent in width, and only the second is what the potential
actually scores. Use the resampler when speed matters or the ensemble is being put into a
reference's convention; use the equilibration when the output is itself the reference.

```bash
md-sim sampling equilibrate-hbonds x-unconstrained.yaml --trajectory traj.trr --topology top.pdb \
    --out out/ --time 5.0 --friction 20.0       # one shard; see --help for --start/--stop/--stride
```

`--time` (ps of dynamics per frame) and `--friction` are the two knobs below. A whole trajectory is
an array job over shards, which a job script in the repo running the campaign provides.

The config must set `constraints: none` — it is both the only build carrying the bond terms and
the only one OpenMM will freeze atoms in. Two numbers have to be measured rather than assumed, and
they trade off: convergence is a **plateau in `--time`**, not agreement with `sqrt(kT/k)`
(that analytic width is the thing being corrected), and raising `--friction` reaches the plateau in
fewer steps while biasing nothing, since the stationary distribution of Langevin dynamics does not
depend on the friction.

The array shards by **contiguous block**, not by stride, because the shards are concatenated back
into one trajectory and only contiguous blocks concatenate in shard order into the *input's* frame
order — which is what every per-frame array derived from the same trajectory (energies, forces,
scores, held-out frame indices) is indexed by. Velocity draws are keyed to the absolute frame
index, so rerunning one shard reproduces exactly the frames it owned.

## The `md-sim` command

`md-sim` is one entry point: `md-sim run`, `md-sim check-job`, `md-sim mcp`, and the groups
`config`, `prep`, `analysis` and `sampling`. `md-sim <group>` lists a group's verbs and `md-sim --json` lists the whole
surface, so nothing has to be known in advance. `md-sim <group> <verb> ...` runs the standalone
script of the same name with its arguments untouched, so `md-sim prep cgschnet-priors --help` prints
exactly what `md-sim-cgschnet-priors --help` does; those `md-sim-*` commands stay installed.

`md-sim config` needs only the core install (no openmm) and lets you author and check a config
without reading its source or running anything:

| Command | What it does |
|---|---|
| `config schema [--engine E]` | JSON Schema of the config union (one `oneOf` branch per engine), or of one engine; under `--json` it is the envelope's `payload` |
| `config validate <cfg>` | exit 0, or exit 1 with every error as `{"loc", "msg", "type"}` |
| `config describe <cfg> [--data-root P]` | engine, system, resolved `output_dir`, each input path with an `exists` flag, Beamtime group and name, `total_time_ns`, `frame_spacing_ps`, `n_frames`; with no data root it still exits 0 and reports relative paths as not existing |
| `config ls` | the workspace's configs with engine, system and temperature; a config that fails to load is listed with its error |
| `config new --from <cfg> --set k=v -o <out> [--force]` | a validated copy with overrides; `k` may be dotted (`replica_exchange.n_replicas`), `v` is a YAML scalar, an unknown field is an error, and comments in the source are not kept |

`--json` is on every `md-sim` command (a script behind a group keeps its own options). It prints one
JSON object on stdout: `{"ok": true, ...}` on success and `{"ok": false, "error": {"type",
"message"}}` with a non-zero exit on failure (1 expected failure, 2 usage). Under
`md-sim run <config> --json` the run's own logging goes to stderr and the object is
`{"ok", "config", "data_root", "output_dir", "engine", "system", "steps"}`.

### `md-sim check-job`: verify a job script before submitting

The package does not generate job scripts, and configs carry no `slurm:` block: write the script
by hand (or have an agent write it) and check it here before it is submitted. A job script is a
property of a cluster and a campaign, so it lives in the repo that runs one.

`md-sim check-job <script.sh> [--data-root P] [--json]` reads the `#SBATCH` header and the
`md-sim run <config>` commands of a script and checks them against the workspace, the site profile
and the configs, one line per check: `directives` (a `#SBATCH` after the first command is ignored by
slurm), `config` (the path exists and validates), `inputs` (`input_pdb`/`input_gro`/`top`/`model_file`
exist under the data root), `partition` (in the site's `partitions`), `gres` (a GPU is requested; `--require-gpu` fails the check if none is),
`output` (the `--output`/`--error` directory exists; slurm does not create it) and `venv` (the site's
`venv_activate` exists). It exits 0 unless a check failed, 1 if one did and 2 if the script or the
workspace cannot be read. A fact the script does not state statically (a config path in a shell
variable, a data root exported from `$PWD`) is reported `skipped`, never passed; a workspace without a
site, or a site with no `partitions`, skips the site checks. Relative paths resolve against the
workspace root, the submit directory. `--json` prints `{"ok", "checks": [{"name", "ok", "status",
"detail"}]}`, or the failure envelope above (exit 2) when the script or workspace cannot be read.

### `md-sim mcp`: the same surface for an agent

`md-sim mcp [--workspace P]` serves the config surface over MCP on stdio, so an agent in a
workspace can use it without a shell. It is listed by `md-sim --json` like any other command. It needs the `mcp` extra (`pip install 'md-simulations[mcp]'`),
which only the lab repo installs; without it `md-sim mcp` exits 1 and names the extra, and
`import md_simulations` never imports `mcp`. The workspace is found once at startup (`--workspace`,
else `$MD_WORKSPACE`, else upward from the cwd; none is exit 2) and every result carries it as
`"workspace"`.

| Tool | Same as | Arguments |
|---|---|---|
| `list_configs` | `config ls` | none |
| `describe_config` | `config describe` | `path`, optional `data_root` |
| `validate_config` | `config validate` | `path` |
| `config_schema` | `config schema` | optional `engine` |
| `new_config` | `config new` | `source`, `output`, optional `overrides` (`KEY=VALUE` list), `force` |
| `check_job` | `check-job` | `script`, optional `require_gpu` |

A result is the envelope the `--json` command prints plus `"workspace"`; a failure is
`{"ok": false, "error": {...}}` as a result, not a transport error. Relative config paths resolve
against `config_dir`, a relative script against the workspace root. `new_config` writes only a
`.yaml` file inside `config_dir` and never the workspace descriptor or `sites/`. The tool set is
read-and-author only: nothing starts a simulation or calls `sbatch`, and run state stays with the
beamtime server. `docs/lab-repo-mcp.json` is a draft `.mcp.json` for the `simulations` repo.

## Engines

Pick one with `engine:` in the config. All of them share the same integrators, reporters
and output files, so the table is mostly about what inputs each needs. `md-sim config schema
--engine <name>` prints the fields of one engine, and `md-sim config describe <cfg>` shows what a
config resolves to.

| engine | runs | inputs | needs |
| --- | --- | --- | --- |
| `amber` | all-atom AMBER, explicit (PME) or implicit GB | all-atom PDB | — |
| `pure_liquid` | NPT of a pre-built periodic box (e.g. CHARMM36) | periodic PDB | — |
| `martini` | MARTINI3 coarse-grained under OpenMM | `.gro`, `.top`, ITP dir | `--extra martini` |
| `gromacs_input` | OpenMM built straight from GROMACS inputs (e.g. FreeSolv) | `.gro`, `.top` | — |
| `awsem` | OpenAWSEM 3-bead CG, optionally memory-biased | all-atom PDB | `--extra awsem` |
| `gromacs_native` | real `gmx grompp` + `gmx mdrun` | `.gro`, `.top`, `.mdp` | `gmx` on PATH |
| `cgschnet` | ML coarse-grained (CGSchNet / mlcg) | see below | `--extra cgschnet` (+ `--extra cu130`) |

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

mlcg pins Python 3.12, which is why the whole project targets 3.12. The `cgschnet` extra
is plain `mlcg` and leaves the CUDA line alone; the CUDA-13 wheel set that matches the
cluster is the separate `cu130` extra:

```bash
uv sync --extra cgschnet --extra cu130 --torch-backend=cu130
```

On a CUDA-12 host, drop `--extra cu130` and install torch with `--torch-backend=cu128`.
A project that consumes md-simulations and pins its own torch takes `--extra cgschnet`
alone. Jobs activate the same `.venv` as every other engine.

The released checkpoint was pickled against a pre-2.5 torch-geometric and won't load as-is,
so re-export it once:

```bash
md-sim-cgschnet-reexport data/models/cgschnet/transferable_model_and_prior.pt \
    -o data/models/cgschnet/transferable_model_and_prior.reexport.pt
md-sim run trpcage-cgschnet-300K.yaml
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
uv sync --extra cgschnet-dataset --extra cu130 --torch-backend=cu130
md-sim-cgschnet-dataset --run data/trp-cage/AMBER14_implicit_300K \
    --bead-pdb data/pdbs/2JOF_5B-CG.pdb --out data/trp-cage/fm_300K \
    --cgschnet-config trpcage-cgschnet-300K.yaml --name trpcage
```

Out comes an `.h5` that mlcg's training script reads directly, plus the same arrays as
`.npy` if you want to poke at them. Forces are memory-mapped and read in blocks, so a
multi-GB `forces.npy` never has to fit in memory.

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
pot = build_potential_from_file(traj.topology, "cln-awsem-330K.yaml", data_root="data")

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
subset of OpenMM force groups. That is only a filter, though: `forcefield.createSystem`
leaves every force in group 0, so on an AMBER system `groups={0}` returns the total.

To see one energy term at a time, use `build_potential_terms` /
`build_potential_terms_from_file`. They return `{force name: Potential}` and give each force
its own group before the `Context` is built, so the parts sum to `build_potential`'s total:

```python
terms = build_potential_terms_from_file(top, "trpcage-amber14-implicit-300K.yaml")
{name: pot(x).item() for name, pot in terms.items()}
# {'HarmonicBondForce': 407.0, 'NonbondedForce': -1164.0, 'CustomGBForce': -2553.9, ...}
```

Each term keeps its own calculator on a shared `Context`, because ASE caches on geometry
alone — mutating `groups` on one calculator would return the previous term.

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
uv sync --extra cgschnet-potentials --extra cu130 --torch-backend=cu130
```

(`cu130` only when md-simulations is the top-level project; a consumer with its own torch
pin takes `--extra cgschnet-potentials` alone.)

```python
pot = build_potential_from_file(
    cg_topology, "trpcage-cgschnet-300K.yaml", data_root="data"
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
- **Prep helpers** in `md_simulations.prep`: `extract_frames` (frames of a trajectory out as
  PDBs, i.e. starting structures for new runs: `md-sim-extract-frames --trajectory t.trr
  --topology top.pdb --frames 0 1 2 -o data/pdbs/start`), `martini_cg_map` (CG mapping matrix,
  needs the `martini` extra), `itp_bonds_to_pdb` (CONECT records from an ITP), and the vendored
  `olives` script. The CGSchNet ones — `cgschnet_map`, `cgschnet_embeddings`,
  `cgschnet_reexport`, `cgschnet_dataset` — are covered above.

## Layout

```
src/md_simulations/   the package
tests/                the suite; needs no data root and no cluster
examples/             runnable end-to-end examples
docs/                 agent contract (agents.md), drafts for the lab repo, plans
```

See `CLAUDE.md` for code conventions.
