"""Torch + mdtraj frontend: a differentiable potential bound to a system.

:class:`Potential` is the public face of the package. You build it once from an
mdtraj ``Topology`` (the chemical identity of the system) and an ASE ``Calculator``
(the pluggable engine), then call it with torch tensors:

- ``positions`` -- ``(N, 3)`` for one frame, or ``(F, N, 3)`` to treat the frame
  axis as a torch batch axis (returns ``(F,)``).
- ``unitcell_lengths`` / ``unitcell_angles`` -- optional torch tensors describing
  the periodic box (nm / degrees), passed alongside the positions.

Everything numeric is a torch tensor; ASE lives entirely inside. Positions are
the only differentiable input (forces come from ``.backward()``); the box is
static metadata."""

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from torch import Tensor

from md_simulations.torch_potentials.bridge import NM_TO_ANG, potential_energy
from md_simulations.torch_potentials.interop import atoms_from_topology


class Potential:
    """Differentiable potential energy of a fixed system under a swappable engine.

    The mdtraj ``Topology`` fixes the atoms; the ASE ``Calculator`` is the
    engine (EMT, LAMMPS, OpenMM, ...). Positions and box are supplied per call as
    torch tensors."""

    def __init__(self, topology: "mdtraj.Topology", calculator: Calculator) -> None:  # noqa: F821
        """Bind a topology to an ASE calculator.

        :param topology: mdtraj ``Topology`` defining the atoms.
        :param calculator: A ready ASE ``Calculator`` instance (the engine)."""
        self._atoms = atoms_from_topology(topology)
        self._atoms.calc = calculator
        self._topology = topology

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the bound system."""
        return len(self._atoms)

    @property
    def topology(self) -> "mdtraj.Topology":  # noqa: F821
        """The bound mdtraj ``Topology``."""
        return self._topology

    @property
    def atoms(self) -> Atoms:
        """The internal ASE ``Atoms`` handle (read-only; positions are transient)."""
        return self._atoms

    def __call__(
        self,
        positions: Tensor,
        unitcell_lengths: Tensor | None = None,
        unitcell_angles: Tensor | None = None,
    ) -> Tensor:
        """Return the differentiable potential energy at ``positions``.

        :param positions: ``(N, 3)`` (one frame) or ``(F, N, 3)`` (a batch of
            frames) atomic positions in nanometre; set ``requires_grad=True`` to
            recover forces as ``-positions.grad``.
        :param unitcell_lengths: Box edge lengths in nm, ``(3,)`` or ``(F, 3)``;
            ``None`` for a non-periodic system. A single ``(3,)`` is broadcast
            across a batch.
        :param unitcell_angles: Box angles in degrees, ``(3,)`` or ``(F, 3)``;
            ``None`` defaults to 90 degrees (orthorhombic).
        :return: Scalar energy in kJ/mol for ``(N, 3)`` input, else a ``(F,)``
            tensor of energies, differentiable w.r.t. ``positions``."""
        lengths = None if unitcell_lengths is None else \
            unitcell_lengths.detach().cpu().numpy() * NM_TO_ANG  # nm -> A
        angles = None if unitcell_angles is None else unitcell_angles.detach().cpu().numpy()

        if positions.dim() == 3:
            energies = []
            for f in range(positions.shape[0]):
                self._apply_box(_row(lengths, f), _row(angles, f))
                energies.append(potential_energy(positions[f], self._atoms))
            return torch.stack(energies)

        if positions.dim() == 2:
            self._apply_box(lengths, angles)
            return potential_energy(positions, self._atoms)

        raise ValueError(f"positions must be (N, 3) or (F, N, 3), got {tuple(positions.shape)}")

    def _apply_box(self, lengths: np.ndarray | None, angles: np.ndarray | None) -> None:
        """Set the internal cell/PBC from edge lengths (A) and angles (deg).

        :param lengths: ``(3,)`` edge lengths in angstrom, or ``None`` for no box.
        :param angles: ``(3,)`` angles in degrees, or ``None`` for orthorhombic."""
        if lengths is None:
            self._atoms.set_pbc(False)
            self._atoms.set_cell([0.0, 0.0, 0.0])
            return
        a, b, c = (float(v) for v in lengths)
        alpha, beta, gamma = (90.0, 90.0, 90.0) if angles is None else (float(v) for v in angles)
        self._atoms.set_cell([a, b, c, alpha, beta, gamma])
        self._atoms.set_pbc(True)


def _row(arr: np.ndarray | None, frame: int) -> np.ndarray | None:
    """Select a per-frame row, broadcasting a single ``(3,)`` across the batch.

    :param arr: ``None``, a ``(3,)`` array (shared), or a ``(F, 3)`` array.
    :param frame: Frame index into a ``(F, 3)`` array.
    :return: The ``(3,)`` row for ``frame``, or ``None``."""
    if arr is None:
        return None
    return arr[frame] if arr.ndim == 2 else arr
