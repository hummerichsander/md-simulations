"""Helpers for building small mdtraj trajectories in tests.

Deliberately *not* folded into ``conftest.py``: a nested ``conftest`` shadows the
top-level one for ``from conftest import ...``, which would break
``tests/test_config.py``. A uniquely-named helper module sidesteps that."""

import mdtraj as md
import numpy as np


def make_traj(symbols: list[str], xyz_nm) -> md.Trajectory:
    """Build a single-residue mdtraj Trajectory from element symbols and coords.

    :param symbols: Per-atom chemical symbols, e.g. ``["Cu", "Cu"]``.
    :param xyz_nm: Coordinates in nm, shape ``(N, 3)`` or ``(F, N, 3)``.
    :return: An mdtraj ``Trajectory`` with one chain/residue and no periodic box."""
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("SYS", chain)
    for s in symbols:
        top.add_atom(s, md.element.Element.getBySymbol(s), res)

    xyz = np.asarray(xyz_nm, dtype=np.float32).reshape(-1, len(symbols), 3)
    return md.Trajectory(xyz, top)
