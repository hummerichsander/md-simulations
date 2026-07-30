"""Public ``Potential`` frontend tests against the EMT backend.

EMT is a fast, dependency-light ASE calculator, ideal for validating the
torch + mdtraj frontend (single-frame gradients, frame batching, periodic box,
engine swap) without needing OpenMM."""

import mdtraj as md
import numpy as np
import torch
from ase.build import bulk
from ase.calculators.emt import EMT

from md_simulations.torch_potentials import Potential
from md_simulations.torch_potentials.bridge import EV_TO_KJMOL, FORCE_EV_ANG_TO_KJMOL_NM

from traj_util import make_traj


def test_gradcheck_single_frame(cu_traj: md.Trajectory) -> None:
    """``gradcheck`` validates the whole frontend (sign, units, chain rule).

    :param cu_traj: Non-periodic Cu cluster trajectory fixture."""
    pot = Potential(cu_traj.topology, EMT())
    x = torch.tensor(cu_traj.xyz[0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pot, (x,), atol=1e-5)


def test_grad_matches_forces(cu_traj: md.Trajectory) -> None:
    """``-x.grad`` equals the engine's forces converted to kJ/mol/nm.

    :param cu_traj: Non-periodic Cu cluster trajectory fixture."""
    pot = Potential(cu_traj.topology, EMT())
    x = torch.tensor(cu_traj.xyz[0], dtype=torch.float64, requires_grad=True)
    pot(x).backward()

    forces = pot.atoms.get_forces() * FORCE_EV_ANG_TO_KJMOL_NM
    np.testing.assert_allclose(-x.grad.numpy(), forces, atol=1e-6)


def test_batched_matches_per_frame(cu_traj: md.Trajectory) -> None:
    """A ``(F, N, 3)`` batch returns ``(F,)`` equal to per-frame evaluations.

    :param cu_traj: Non-periodic Cu cluster trajectory fixture."""
    pot = Potential(cu_traj.topology, EMT())
    X = torch.tensor(cu_traj.xyz, dtype=torch.float64, requires_grad=True)
    E = pot(X)

    assert E.shape == (cu_traj.n_frames,)
    per_frame = np.array(
        [pot(torch.tensor(cu_traj.xyz[f], dtype=torch.float64)).item() for f in range(cu_traj.n_frames)]
    )
    np.testing.assert_allclose(E.detach().numpy(), per_frame, atol=1e-8)


def test_batched_backward_routes_per_frame_forces(cu_traj: md.Trajectory) -> None:
    """One ``.backward()`` on the batch routes each frame's force into ``X.grad``.

    :param cu_traj: Non-periodic Cu cluster trajectory fixture."""
    pot = Potential(cu_traj.topology, EMT())
    X = torch.tensor(cu_traj.xyz, dtype=torch.float64, requires_grad=True)
    pot(X).sum().backward()

    assert X.grad.shape == X.shape
    for f in range(cu_traj.n_frames):
        xf = torch.tensor(cu_traj.xyz[f], dtype=torch.float64, requires_grad=True)
        pot(xf).backward()
        np.testing.assert_allclose(X.grad[f].numpy(), xf.grad.numpy(), atol=1e-8)


def test_unitcell_lengths_match_periodic_reference() -> None:
    """A periodic energy via ``unitcell_lengths`` matches a periodic ASE reference.

    :return: None."""
    ref = bulk("Cu", "fcc", a=3.6, cubic=True)
    ref.rattle(stdev=0.05, seed=1)
    ref.calc = EMT()
    e_ref = ref.get_potential_energy() * EV_TO_KJMOL

    xyz_nm = ref.get_positions() / 10.0  # A -> nm, exact float64
    traj = make_traj(["Cu"] * len(ref), xyz_nm)
    pot = Potential(traj.topology, EMT())

    x = torch.tensor(xyz_nm, dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor(np.diag(np.asarray(ref.get_cell())) / 10.0)  # cubic -> (3,) nm
    energy = pot(x, unitcell_lengths=lengths)

    assert np.isclose(energy.item(), e_ref, atol=1e-5)


def test_unitcell_changes_energy(cu_traj: md.Trajectory) -> None:
    """Supplying a box (periodic) changes the energy versus non-periodic.

    :param cu_traj: Non-periodic Cu cluster trajectory fixture."""
    pot = Potential(cu_traj.topology, EMT())
    x = torch.tensor(cu_traj.xyz[0], dtype=torch.float64)

    free = pot(x).item()
    periodic = pot(x, unitcell_lengths=torch.tensor([0.4, 0.4, 0.4])).item()
    assert not np.isclose(free, periodic)


def test_backend_swap() -> None:
    """The same ``Potential`` code works with a calculator on a different system.

    :return: None."""
    traj = make_traj(["Pt", "Pt"], np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]))
    pot = Potential(traj.topology, EMT())

    x = torch.tensor(traj.xyz[0], dtype=torch.float64, requires_grad=True)
    pot(x).backward()

    forces = pot.atoms.get_forces() * FORCE_EV_ANG_TO_KJMOL_NM
    np.testing.assert_allclose(-x.grad.numpy(), forces, atol=1e-6)


def test_float32_opt_in(cu_traj: md.Trajectory) -> None:
    """float32 positions give a float32 energy and finite gradients.

    :param cu_traj: Non-periodic Cu cluster trajectory fixture."""
    pot = Potential(cu_traj.topology, EMT())
    x = torch.tensor(cu_traj.xyz[0], dtype=torch.float32, requires_grad=True)

    energy = pot(x)
    assert energy.dtype == torch.float32
    energy.backward()
    assert torch.isfinite(x.grad).all()
