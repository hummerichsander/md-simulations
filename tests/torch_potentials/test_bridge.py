"""Core bridge tests: gradcheck, force matching, and backend agnosticism.

The torch API is in nm / kJ/mol / (kJ/mol/nm); ASE calculators report eV / A,
so engine comparisons apply the conversion factors from :mod:`md_simulations.torch_potentials.bridge`."""

import numpy as np
import torch
from ase import Atoms
from ase.calculators.emt import EMT

from md_simulations.torch_potentials.bridge import (
    EV_TO_KJMOL,
    FORCE_EV_ANG_TO_KJMOL_NM,
    EnergyFn,
    positions_to_nm,
    potential_energy,
)


def test_gradcheck_positions(emt_cluster: Atoms) -> None:
    """``gradcheck`` validates sign, units and chain rule of the whole bridge.

    It finite-differences the engine's energy and compares against the engine's
    analytic forces routed through ``backward`` (in the nm / kJ.mol convention).

    :param emt_cluster: Small non-periodic Cu cluster fixture."""
    x = torch.tensor(positions_to_nm(emt_cluster), requires_grad=True)
    assert torch.autograd.gradcheck(EnergyFn.apply, (x, emt_cluster), atol=1e-5)


def test_grad_matches_forces(emt_cluster: Atoms) -> None:
    """``-x.grad`` must equal the engine's forces converted to kJ/mol/nm.

    :param emt_cluster: Small non-periodic Cu cluster fixture."""
    x = torch.tensor(positions_to_nm(emt_cluster), requires_grad=True)
    energy = potential_energy(x, emt_cluster)
    energy.backward()

    forces_kjmol_nm = emt_cluster.get_forces() * FORCE_EV_ANG_TO_KJMOL_NM
    np.testing.assert_allclose(-x.grad.numpy(), forces_kjmol_nm, atol=1e-6)


def test_energy_value_matches_engine(emt_cluster: Atoms) -> None:
    """The differentiable energy equals the engine's energy in kJ/mol.

    :param emt_cluster: Small non-periodic Cu cluster fixture."""
    x = torch.tensor(positions_to_nm(emt_cluster), requires_grad=True)
    energy = potential_energy(x, emt_cluster)
    assert np.isclose(energy.item(), emt_cluster.get_potential_energy() * EV_TO_KJMOL)


def test_gradient_flows_through_upstream(emt_cluster: Atoms) -> None:
    """Gradients flow into an upstream torch computation that produced coords.

    Simulates a network output: ``x = raw @ R``; the loss on the energy must
    produce a nonzero gradient on ``raw``.

    :param emt_cluster: Small non-periodic Cu cluster fixture."""
    raw = torch.tensor(positions_to_nm(emt_cluster), requires_grad=True)
    R = torch.eye(3) + 1e-3 * torch.randn(3, 3)
    x = raw @ R

    energy = potential_energy(x, emt_cluster)
    energy.backward()

    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0


def test_backend_swap_emt() -> None:
    """The same ``EnergyFn`` works against ASE's EMT with no code change.

    Proves the core references only ``get_potential_energy``/``get_forces`` --
    it is engine-agnostic.

    :return: None."""
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms.calc = EMT()

    x = torch.tensor(positions_to_nm(atoms), requires_grad=True)
    energy = potential_energy(x, atoms)
    energy.backward()

    forces_kjmol_nm = atoms.get_forces() * FORCE_EV_ANG_TO_KJMOL_NM
    np.testing.assert_allclose(-x.grad.numpy(), forces_kjmol_nm, atol=1e-6)


def test_float32_opt_in(emt_cluster: Atoms) -> None:
    """float32 positions yield a float32 energy and finite gradients.

    :param emt_cluster: Small non-periodic Cu cluster fixture."""
    x = torch.tensor(positions_to_nm(emt_cluster), dtype=torch.float32, requires_grad=True)
    energy = potential_energy(x, emt_cluster)
    assert energy.dtype == torch.float32
    energy.backward()
    assert torch.isfinite(x.grad).all()
