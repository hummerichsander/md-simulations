# simulations

Draft README for the lab repo `/remote/gpu04/hummerich/projects/simulations`, to be pasted there
by the user (S3 of `docs/plans/split-package-and-lab.md` creates the repo; D2 writes its agent
instructions). Delete this line when pasting.

The research repo around the [`md-simulations`](https://github.com/hummerichsander/md-simulations) package: one versioned config per simulated
system, the hand-written job scripts that run them on the cluster, and the Beamtime experiment
log. The engines, the config models, the torch potentials and the `md-sim` command live in the
package; this repo holds only what is particular to this lab and this cluster.

```bash
uv sync                                   # md-simulations[simulate,...] plus the notebook tooling
md-sim config ls                          # the configs in this workspace
md-sim run configs/cln-amber14-implicit-330K.yaml     # locally
md-sim check-job slurm/ala2-amber14-implicit-330K.sh  # verify a script ...
sbatch slurm/ala2-amber14-implicit-330K.sh            # ... then submit it
```

## Layout

```
md-workspace.yaml     the workspace descriptor: config_dir, data_root, run_store, site
sites/                one profile per cluster: partitions, venv activation, slurm output, modules
configs/              one YAML per system, plus templates/ and mdp/
slurm/                hand-written batch scripts; output/ holds job logs
data/                 symlink to the shared data root /remote/gpu04/hummerich/data (git-ignored)
beamtime/             experiment log (git-ignored; see beamtime.yaml)
```

`md-workspace.yaml` is how the package finds all of this (walking up from the cwd, or `$MD_WORKSPACE`);
the package documents the file in its README under "Workspaces". Data never lives in this repo.

## Experiment log

Every `md-sim run` is also logged with [Beamtime](https://hummerichsander.github.io/beamtime/)
into `beamtime/` at the workspace root (set in `beamtime.yaml`; the logs are not committed). A
run records the config, the absolute output directory, energies, temperature and speed every
`report_interval` steps (potential energy and RMS force per saved frame for `cgschnet`), the
console output, and references to the output files, which stay on the data root. Logs are
grouped as systems, simulations and replica sets:

```
beamtime/
  cln/                              one group per system
    AMBER14_implicit_stabilized_315K  a simulation is a run, named after its output folder
    AMBER14_implicit_330K/          a replica set is a group of runs
      r00  r01  ...
```

A config sets its place with an optional `beamtime:` block. `group` defaults to the system and
`name` to the last part of `output_subdir`. Configs that each run one replica of a set name the
set as their group:

```yaml
output_subdir: cln/AMBER14_implicit_330K_r01
beamtime:
  group: cln/AMBER14_implicit_330K
  name: r01
```

A `cgschnet` config with `n_replicas > 1` makes this group by itself:
`<group>/<name>/r00, r01, ...`, one run per batched replica. Those runs do not capture the
console output, and a job killed by slurm leaves them `lost` rather than `killed`.
`beamtime serve --open` shows the log in the browser.

## Starting a new system

Copy the template for your engine — every field is commented — then edit and run:

```bash
cp configs/templates/amber.yaml configs/mysystem.yaml
md-sim run configs/mysystem.yaml
```

For a cluster job, write the batch script by hand (or have an agent write it) in `slurm/`, starting
from one of the template examples there, and verify it with `md-sim check-job` before submitting.
Nothing generates job scripts, and configs carry no `slurm:` block: the partition, walltime and
job name are the script's own. `md-sim check-job <script.sh> [--require-gpu] [--json]` is
documented in the [package README](https://github.com/hummerichsander/md-simulations#md-sim-check-job-verify-a-job-script-before-submitting);
it checks the `#SBATCH` header, the config, its input files, the partition, the GPU request, the
log directory and the venv against `md-workspace.yaml` and `sites/<site>.yaml`.

### Worked example: conditional sampling

`configs/cln-fiber-amber14-implicit-330K-rep*.yaml` + `slurm/cln-fiber-amber14-implicit-330K.sh`
are a worked example of `frozen_atoms:` (documented in the package README): six chignolin runs, each
freezing the 47-bead slow subspace of one conformational mode.

### Bond repair at scale

`slurm/equilibrate-h-bonds.sh <config> <trajectory> <topology>` runs `md-sim-equilibrate-hbonds` as
a sharded array job; `EQUIL_TIME` and `FRICTION` in the environment set its `--time` and
`--friction` (`EQUIL_TIME=5.0 FRICTION=20.0 sbatch slurm/equilibrate-h-bonds.sh ...`, knobs
listed in the script header). The method is documented in the package README under "Repairing
constrained bonds".

## For agents

Read the package's `docs/agents.md` first: enumerate the CLI with `md-sim --help`, author against
`md-sim config schema`, verify with `md-sim config validate` and `md-sim check-job`, never write
into the data root, and never submit. D2 adds this repo's own `CLAUDE.md`.
