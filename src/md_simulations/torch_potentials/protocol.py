"""The structural contract every torch potential in this package satisfies.

:class:`Potential` (ASE-backed) and :class:`~md_simulations.torch_potentials.cgschnet.CGSchNetPotential`
(native torch) share no base class -- they wrap fundamentally different engines --
but they are interchangeable to a caller: bind a system once, then call with torch
positions in nm and get energies in kJ/mol back. :class:`PotentialLike` names that
contract so :func:`~md_simulations.torch_potentials.build.build_potential` has one
honest return type.

This module deliberately imports nothing but ``typing`` (``Tensor`` only under
``TYPE_CHECKING``), so ``build.py`` and the package ``__init__`` can import it
eagerly on a core install without dragging in torch."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor


@runtime_checkable
class PotentialLike(Protocol):
    """A differentiable potential energy bound to a fixed system (nm in, kJ/mol out)."""

    @property
    def n_atoms(self) -> int:
        """Number of atoms (or coarse-grained beads) in the bound system."""
        ...

    @property
    def topology(self) -> "mdtraj.Topology":  # noqa: F821
        """The bound mdtraj ``Topology``."""
        ...

    def __call__(
        self,
        positions: "Tensor",
        unitcell_lengths: "Tensor | None" = None,
        unitcell_angles: "Tensor | None" = None,
    ) -> "Tensor":
        """Return the differentiable potential energy at ``positions``.

        :param positions: ``(N, 3)`` (one frame) or ``(F, N, 3)`` (a batch of frames)
            positions in nanometre.
        :param unitcell_lengths: Box edge lengths in nm, or ``None`` for a
            non-periodic system.
        :param unitcell_angles: Box angles in degrees, or ``None`` for orthorhombic.
        :return: Scalar energy in kJ/mol for ``(N, 3)`` input, else ``(F,)``."""
        ...
