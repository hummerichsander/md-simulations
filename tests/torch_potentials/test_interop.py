"""Tests for the mdtraj -> ASE topology converter."""

import numpy as np

from md_simulations.torch_potentials.interop import atoms_from_topology

from traj_util import make_traj


def test_symbols_and_count_preserved() -> None:
    """The converter carries the topology's element symbols and atom count.

    :return: None."""
    traj = make_traj(["O", "H", "H"], np.zeros((3, 3)))
    atoms = atoms_from_topology(traj.topology)

    assert len(atoms) == 3
    assert list(atoms.get_chemical_symbols()) == ["O", "H", "H"]
