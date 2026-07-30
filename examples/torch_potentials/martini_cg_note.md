# Coarse-grained (MARTINI) systems — where they plug in

Coarse-grained models plug in exactly like all-atom ones: build the OpenMM
`System` for the CG force field, wrap its `Context` as an `OpenMMCalculator`, and
hand it to `Potential`. Only the engine setup changes; the mdtraj + torch
frontend (`Potential(traj.topology, calc)`, `pot(x)`, backprop) is identical.

## Building the CG system

For an all-atom AMBER system the OpenMM `System` comes from the built-in force
field:

```python
from openmm import app
ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
system = ff.createSystem(pdb.topology, nonbondedMethod=app.PME, ...)
```

For a MARTINI system, you build the `System` through the MARTINI setup path
instead (e.g. [`martini_openmm`](https://github.com/maccallumlab/martini_openmm),
which reads a GROMACS `.top` + `.itp` topology):

```python
import martini_openmm as martini
from openmm import app, unit

conf = app.GromacsGroFile("system.gro")
top = martini.MartiniTopFile(
    "system.top",
    periodicBoxVectors=conf.getPeriodicBoxVectors(),
)
system = top.create_system(nonbonded_cutoff=1.1 * unit.nanometer)
```

Everything downstream is the ordinary frontend:

```python
import mdtraj as md, torch
from openmm import Context, Platform, VerletIntegrator, unit
from md_simulations.torch_potentials import Potential
from md_simulations.torch_potentials.calculators.openmm import OpenMMCalculator

context = Context(system, VerletIntegrator(1 * unit.femtosecond),
                  Platform.getPlatformByName("CPU"))

# An mdtraj Trajectory carries the (CG) topology. If you already have the OpenMM
# topology, md.Topology.from_openmm(omm_topology) converts it directly.
traj = md.load("system.gro")                 # or build from the OpenMM topology
pot  = Potential(traj.topology, OpenMMCalculator(context))

x = torch.tensor(traj.xyz[0], dtype=torch.float64, requires_grad=True)   # nm
energy = pot(x)                              # differentiable kJ/mol
energy.backward()                            # forces = -x.grad
```

## Virtual sites

MARTINI (and 4-site waters like TIP4P) use **virtual sites**. Only the *real*
particles are torch degrees of freedom: `x` (and the trajectory) must contain
exactly the real particles, in the engine's atom order. Never expose virtual
sites as torch coordinates. OpenMM recomputes virtual-site positions inside
`context.setPositions` via `context.computeVirtualSites()`; the forces returned
for real particles already include the virtual-site contributions projected back.

## Neighbor lists / cutoffs / PBC

These are entirely the engine's responsibility. The frontend never touches them —
set them up when you build the `System` (`nonbonded_cutoff`, `periodicBoxVectors`,
etc.). At call time you pass the box to `pot` as `unitcell_lengths` (nm), e.g.
`pot(x, unitcell_lengths=torch.tensor(traj.unitcell_lengths[0]))`.
