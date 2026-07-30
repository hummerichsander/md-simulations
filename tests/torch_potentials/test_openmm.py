"""OpenMM reference-backend tests through the ``Potential`` frontend.

Proves the torch + mdtraj frontend drives a real OpenMM engine: a tight
``gradcheck`` on a smooth programmatic system, force-group subsetting, and an
all-atom AMBER (TIP3P) smoke test."""

import numpy as np
import openmm
import pytest
import torch
from openmm import (
    Context,
    HarmonicBondForce,
    NonbondedForce,
    Platform,
    System,
    VerletIntegrator,
    unit,
)

from md_simulations.torch_potentials import Potential
from md_simulations.torch_potentials.bridge import FORCE_EV_ANG_TO_KJMOL_NM
from md_simulations.torch_potentials.calculators.openmm import OpenMMCalculator

from traj_util import make_traj

DIATOMIC_POS_NM = np.array([[0.0, 0.0, 0.0], [0.17, 0.01, 0.0]])


def _reference_context(system: System) -> Context:
    """Build a double-precision Reference-platform context for ``system``.

    :param system: The OpenMM ``System`` to evaluate.
    :return: An OpenMM ``Context`` on the Reference platform."""
    integrator = VerletIntegrator(1.0 * unit.femtosecond)  # unused for single-point
    platform = Platform.getPlatformByName("Reference")
    return Context(system, integrator, platform)


def _diatomic_system(group_bond: int = 0, group_nb: int = 0) -> System:
    """A two-particle system: stiff harmonic bond + Lennard-Jones nonbonded.

    :param group_bond: OpenMM force group for the bond term.
    :param group_nb: OpenMM force group for the nonbonded term.
    :return: The configured OpenMM ``System``."""
    system = System()
    system.addParticle(16.0)
    system.addParticle(16.0)

    bond = HarmonicBondForce()
    bond.addBond(0, 1, 0.15 * unit.nanometer, 4.0e5)  # r0 = 0.15 nm, k stiff but smooth
    bond.setForceGroup(group_bond)
    system.addForce(bond)

    nb = NonbondedForce()
    nb.addParticle(0.0, 0.3, 0.5)  # charge e, sigma nm, epsilon kJ/mol
    nb.addParticle(0.0, 0.3, 0.5)
    nb.setForceGroup(group_nb)
    system.addForce(nb)
    return system


@pytest.fixture
def diatomic_pot() -> Potential:
    """A ``Potential`` over the diatomic OpenMM system (all force groups).

    :return: A ``Potential`` with an ``OpenMMCalculator`` attached."""
    context = _reference_context(_diatomic_system())
    traj = make_traj(["O", "O"], DIATOMIC_POS_NM)
    return Potential(traj.topology, OpenMMCalculator(context))


def test_openmm_gradcheck(diatomic_pot: Potential) -> None:
    """``gradcheck`` through the real OpenMM context validates units and sign.

    :param diatomic_pot: Diatomic OpenMM potential fixture."""
    x = torch.tensor(DIATOMIC_POS_NM, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(diatomic_pot, (x,), atol=1e-5)


def test_openmm_grad_matches_forces(diatomic_pot: Potential) -> None:
    """``-x.grad`` matches the OpenMM forces (sign + unit convention).

    :param diatomic_pot: Diatomic OpenMM potential fixture."""
    x = torch.tensor(DIATOMIC_POS_NM, dtype=torch.float64, requires_grad=True)
    energy = diatomic_pot(x)
    assert np.isfinite(energy.item())

    energy.backward()
    forces = diatomic_pot.atoms.get_forces() * FORCE_EV_ANG_TO_KJMOL_NM
    np.testing.assert_allclose(-x.grad.numpy(), forces, atol=1e-6)


def test_from_system_matches_manual_context() -> None:
    """``OpenMMCalculator.from_system`` matches a hand-wired context.

    :return: None."""
    system = _diatomic_system()
    traj = make_traj(["O", "O"], DIATOMIC_POS_NM)
    x = torch.tensor(DIATOMIC_POS_NM, dtype=torch.float64)

    manual = Potential(traj.topology, OpenMMCalculator(_reference_context(system)))(x).item()
    from_sys = Potential(
        traj.topology,
        OpenMMCalculator.from_system(
            _diatomic_system(), DIATOMIC_POS_NM * unit.nanometer, platform="Reference"
        ),
    )(x).item()

    assert np.isclose(manual, from_sys, atol=1e-6)


def test_force_group_subset() -> None:
    """Force groups select a subset of terms; the parts sum to the whole.

    :return: None."""
    context = _reference_context(_diatomic_system(group_bond=0, group_nb=1))
    traj = make_traj(["O", "O"], DIATOMIC_POS_NM)
    x = torch.tensor(DIATOMIC_POS_NM, dtype=torch.float64)

    e_all = Potential(traj.topology, OpenMMCalculator(context, groups=-1))(x).item()
    e_bond = Potential(traj.topology, OpenMMCalculator(context, groups={0}))(x).item()
    e_nb = Potential(traj.topology, OpenMMCalculator(context, groups={1}))(x).item()

    assert not np.isclose(e_bond, e_all)
    assert np.isclose(e_all, e_bond + e_nb, atol=1e-6)


def _build_water() -> tuple[Context, list[str], np.ndarray]:
    """Build a 3-water TIP3P (AMBER-family) OpenMM context.

    :return: ``(context, symbols, positions_nm)`` in the engine's atom order."""
    from openmm import app

    base = np.array(
        [[0.0, 0.0, 0.0], [0.09572, 0.0, 0.0], [-0.0239987, 0.0926627, 0.0]]
    )  # nm: O, H1, H2
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
    context = _reference_context(system)
    return context, ["O", "H", "H"] * 3, np.array(positions_nm)


def test_amber_water_smoke() -> None:
    """All-atom AMBER (TIP3P) smoke test: finite energy and force matching.

    :return: None."""
    try:
        context, symbols, positions_nm = _build_water()
    except Exception as exc:  # pragma: no cover - depends on shipped ff files
        pytest.skip(f"TIP3P force field unavailable: {exc}")

    pot = Potential(make_traj(symbols, positions_nm).topology, OpenMMCalculator(context))
    x = torch.tensor(positions_nm, dtype=torch.float64, requires_grad=True)

    energy = pot(x)
    assert np.isfinite(energy.item())

    energy.backward()
    forces = pot.atoms.get_forces() * FORCE_EV_ANG_TO_KJMOL_NM
    np.testing.assert_allclose(-x.grad.numpy(), forces, atol=1e-6)
