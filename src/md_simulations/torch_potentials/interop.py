"""Build an ASE ``Atoms`` from an mdtraj ``Topology``.

The frontend is torch + mdtraj, but the engine layer speaks ASE. This converter
is the bridge: it carries only the *topology* (chemical identity per atom) into
an ``ase.Atoms``. Positions and the periodic box are torch tensors supplied at
call time (:class:`md_simulations.torch_potentials.potential.Potential`), so they are placeholders
here -- the differentiable bridge overwrites the positions on every evaluation."""

import numpy as np
from ase import Atoms


def atoms_from_topology(topology: "mdtraj.Topology") -> Atoms:  # noqa: F821
    """Create an ``ase.Atoms`` carrying the topology's per-atom chemical symbols.

    Positions are zero-filled placeholders; the caller sets the real (torch)
    positions and box per evaluation. Only the topology's element symbols matter.

    :param topology: An mdtraj ``Topology`` defining the atoms.
    :return: An ``ase.Atoms`` with ``n_atoms`` atoms and matching chemical symbols."""
    symbols = [atom.element.symbol for atom in topology.atoms]
    return Atoms(symbols=symbols, positions=np.zeros((len(symbols), 3)))
