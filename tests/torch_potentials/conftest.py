"""Shared pytest fixtures for the md_simulations.torch_potentials test suite.

The whole directory is skipped without the ``torch-potentials`` extra, so the
fixtures below may import torch and ase freely.

Note the autouse ``_float64_default`` fixture: it is deliberately scoped to this
directory rather than the top-level ``tests/conftest.py``, so it cannot perturb the
default dtype for the rest of the suite (``test_cgschnet`` in particular)."""

import mdtraj as md
import numpy as np
import pytest

pytest.importorskip("torch", reason="requires the 'torch-potentials' extra")

import torch  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402

from traj_util import make_traj  # noqa: E402


@pytest.fixture(autouse=True)
def _float64_default():
    """Run every test in float64 so gradient checks are numerically tight.

    :return: Yields control to the test, then restores the default dtype."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture
def emt_cluster() -> Atoms:
    """A small non-periodic Cu cluster with an attached EMT calculator.

    Small and smooth -- ideal for ``gradcheck`` on positions.

    :return: An ASE ``Atoms`` object with EMT attached."""
    atoms = Atoms(
        "Cu4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.2, 0.1],
            [0.1, 1.9, 0.0],
            [1.7, 1.6, 1.5],
        ],
    )
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def cu_traj() -> md.Trajectory:
    """A 3-frame, non-periodic 4-atom Cu cluster trajectory (nm).

    The frame axis exercises the batched ``Potential`` path; the geometry is the
    non-periodic ``emt_cluster`` (in nm) with small per-frame perturbations.

    :return: An mdtraj ``Trajectory`` of shape ``(3, 4, 3)``."""
    base = np.array(
        [[0.0, 0.0, 0.0], [0.18, 0.02, 0.01], [0.01, 0.19, 0.0], [0.17, 0.16, 0.15]]
    )  # nm
    xyz = np.stack([base, base + 0.01, base - 0.008])
    return make_traj(["Cu"] * 4, xyz)
