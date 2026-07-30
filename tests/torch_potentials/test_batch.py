"""Tests for batching over heterogeneous systems (distinct ``Potential``s)."""

import numpy as np
import torch
from ase.calculators.emt import EMT

from md_simulations.torch_potentials import Potential, batched_potential_energy
from md_simulations.torch_potentials.bridge import FORCE_EV_ANG_TO_KJMOL_NM

from traj_util import make_traj


def _make_item(sep: float, n: int = 2) -> tuple[Potential, torch.Tensor]:
    """Build a linear Cu system as a ``(Potential, positions)`` batch item.

    :param sep: Nearest-neighbour separation along x in nm.
    :param n: Number of atoms in a line.
    :return: ``(potential, positions)`` with ``requires_grad`` positions in nm."""
    xyz = np.array([[[sep * i, 0.0, 0.0] for i in range(n)]])  # (1, n, 3) nm
    traj = make_traj(["Cu"] * n, xyz)
    pot = Potential(traj.topology, EMT())
    x = torch.tensor(traj.xyz[0], dtype=torch.float64, requires_grad=True)
    return pot, x


def test_batch_stacks_energies() -> None:
    """Batched energies match per-item single evaluations.

    :return: None."""
    items = [_make_item(0.24), _make_item(0.26, n=3), _make_item(0.28)]
    batched = batched_potential_energy(items)

    expected = torch.stack([pot(x) for pot, x in items])
    np.testing.assert_allclose(batched.detach().numpy(), expected.detach().numpy())
    assert batched.shape == (3,)


def test_batch_accepts_generator() -> None:
    """An iterable input works whether it is a list or a generator.

    :return: None."""
    items = [_make_item(0.24), _make_item(0.26)]
    from_list = batched_potential_energy(items)
    from_gen = batched_potential_energy(item for item in items)
    np.testing.assert_allclose(from_list.detach().numpy(), from_gen.detach().numpy())


def test_batch_backward_routes_forces() -> None:
    """A single backward routes the correct force into each item's positions.

    :return: None."""
    items = [_make_item(0.24), _make_item(0.26, n=3)]
    batched_potential_energy(items).sum().backward()

    for pot, x in items:
        forces = pot.atoms.get_forces() * FORCE_EV_ANG_TO_KJMOL_NM
        np.testing.assert_allclose(-x.grad.numpy(), forces, atol=1e-6)


def test_batch_accepts_unitcell_lengths() -> None:
    """A 3-tuple item passes ``unitcell_lengths`` through to the potential.

    :return: None."""
    pot, x = _make_item(0.25)
    lengths = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)  # nm

    batched = batched_potential_energy([(pot, x, lengths)])
    direct = pot(x, unitcell_lengths=lengths)
    np.testing.assert_allclose(batched.detach().numpy(), direct.detach().numpy())


def test_batch_threaded_matches_sequential() -> None:
    """Threaded evaluation (distinct potentials) matches sequential.

    :return: None."""
    items = [_make_item(0.24 + 0.01 * i) for i in range(4)]
    seq = batched_potential_energy(items, num_workers=1)
    par = batched_potential_energy(items, num_workers=4)
    np.testing.assert_allclose(seq.detach().numpy(), par.detach().numpy())
