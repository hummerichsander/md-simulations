"""End-to-end all-atom OpenMM example via the mdtraj + torch frontend.

Build an OpenMM ``System`` from a force field, wrap its ``Context`` as an ASE
calculator, and drive it through
:class:`~md_simulations.torch_potentials.Potential` with torch tensors. The
topology travels OpenMM -> mdtraj -> (internally) ASE; the coordinates you
differentiate and the forces you get back are torch, in nm / kJ·mol⁻¹.

Run with::

    python examples/openmm_all_atom.py"""

import mdtraj as md
import numpy as np
import torch
from openmm import Context, Platform, VerletIntegrator, app, unit

from md_simulations.torch_potentials import Potential
from md_simulations.torch_potentials.bridge import FORCE_EV_ANG_TO_KJMOL_NM
from md_simulations.torch_potentials.calculators.openmm import OpenMMCalculator


def build_water_cluster() -> tuple[md.Trajectory, Context]:
    """Build a 3-molecule TIP3P water cluster as an mdtraj Trajectory + OpenMM context.

    :return: ``(traj, context)`` -- the trajectory carries the topology and initial
        coordinates (nm); the context is ready for the OpenMM calculator."""
    base = np.array([[0.0, 0.0, 0.0], [0.09572, 0.0, 0.0], [-0.0239987, 0.0926627, 0.0]])  # nm
    offsets = np.array([[0.0, 0.0, 0.0], [0.32, 0.0, 0.0], [0.0, 0.31, 0.05]])

    top = app.Topology()
    chain = top.addChain()
    positions_nm = []
    for off in offsets:
        res = top.addResidue("HOH", chain)
        o = top.addAtom("O", app.element.oxygen, res)
        h1 = top.addAtom("H1", app.element.hydrogen, res)
        h2 = top.addAtom("H2", app.element.hydrogen, res)
        top.addBond(o, h1)
        top.addBond(o, h2)
        positions_nm.extend(base + off)

    ff = app.ForceField("amber14/tip3p.xml")
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff, rigidWater=True)

    integrator = VerletIntegrator(1.0 * unit.femtosecond)  # required, unused here
    context = Context(system, integrator, Platform.getPlatformByName("Reference"))

    # OpenMM topology -> mdtraj Trajectory (first-class interop; both work in nm).
    traj = md.Trajectory(np.array(positions_nm)[None], md.Topology.from_openmm(top))
    return traj, context


def main() -> None:
    """Compute a differentiable energy and recover forces via backprop.

    :return: None."""
    traj, context = build_water_cluster()
    pot = Potential(traj.topology, OpenMMCalculator(context))

    # Positions (nm) are the only differentiable DOF; taken straight from mdtraj.
    x = torch.tensor(traj.xyz[0], dtype=torch.float64, requires_grad=True)
    energy = pot(x)
    energy.backward()

    print(f"energy   = {energy.item():.6f} kJ/mol")
    print(f"max|force| (from -x.grad) = {(-x.grad).abs().max().item():.6f} kJ/mol/nm")

    # The gradient equals the engine's analytic forces (converted to kJ/mol/nm).
    forces = pot.atoms.get_forces() * FORCE_EV_ANG_TO_KJMOL_NM
    np.testing.assert_allclose(-x.grad.numpy(), forces, atol=1e-6)
    print("OK: -x.grad matches the engine forces")


if __name__ == "__main__":
    main()
