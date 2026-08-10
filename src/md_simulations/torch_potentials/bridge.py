"""Engine-agnostic differentiable bridge from an ASE ``Calculator`` to PyTorch.

This module is the reusable core of the package. It must never reference a
specific engine: it only ever calls ``atoms.get_potential_energy()`` and
``atoms.get_forces()`` through ASE's standardized ``Calculator`` contract.

Convention (enforced everywhere in the public torch API):
- positions in nanometre (nm)
- energy in kilojoule per mole (kJ/mol)
- forces in kJ/mol/nm

ASE calculators are eV/A by contract (that is how *every* backend -- OpenMM,
LAMMPS, EMT -- reports), so the eV/A <-> kJ/mol/nm conversion happens here, at
the single torch boundary. Doing it in the bridge (rather than in one
calculator) keeps the nm/kJ.mol convention identical across every ASE backend."""

from typing import Any

import ase.units as u
import numpy as np
import torch
from ase import Atoms
from torch import Tensor

# 1 nm expressed in A (== 10.0). ASE positions are in A, the torch API in nm.
NM_TO_ANG: float = u.nm
# 1 eV expressed in kJ/mol (~= 96.485). u.kJ / u.mol is 1 kJ/mol in ASE's eV.
EV_TO_KJMOL: float = 1.0 / (u.kJ / u.mol)
# eV/A force -> kJ/mol/nm: convert the energy (*EV_TO_KJMOL) and the inverse
# length (per-A -> per-nm multiplies by A-per-nm == NM_TO_ANG).
FORCE_EV_ANG_TO_KJMOL_NM: float = EV_TO_KJMOL * NM_TO_ANG
# 1 kcal/mol in kJ/mol. Not an ASE unit: mlcg-family models are native kcal/mol + A
# rather than eV + A. Kept here so every conversion factor in the package has one
# home, even though the native-torch potentials apply it at their own boundary (as an
# autograd op, so their force conversion is derived rather than hand-written).
KCALMOL_TO_KJMOL: float = 4.184


class EnergyFn(torch.autograd.Function):
    """Differentiable potential energy of an ASE ``Atoms`` object.

    ``forward`` pushes torch positions (nm) into the ``Atoms`` object (converting
    to ASE's A), triggers the engine to compute energy and forces, and caches the
    forces in the torch convention. ``backward`` builds the position gradient
    from those cached forces via the chain rule.

    Only positions are differentiable; the engine itself is opaque to autograd,
    so this yields first derivatives only (no Hessians, no double-backward)."""

    @staticmethod
    def forward(ctx: Any, positions: Tensor, atoms: Atoms) -> Tensor:
        """Compute the potential energy for ``positions``.

        :param ctx: Autograd context used to cache forces for ``backward``.
        :param positions: ``(N, 3)`` atomic positions in nanometre.
        :param atoms: ASE ``Atoms`` object carrying an attached calculator.
        :return: Scalar potential energy in kJ/mol, differentiable w.r.t. positions."""
        device, dtype = positions.device, positions.dtype

        # nm -> A at the boundary; set_positions triggers a recompute.
        atoms.set_positions(positions.detach().cpu().numpy() * NM_TO_ANG)
        energy = atoms.get_potential_energy() * EV_TO_KJMOL  # eV -> kJ/mol
        forces = atoms.get_forces() * FORCE_EV_ANG_TO_KJMOL_NM  # eV/A -> kJ/mol/nm

        ctx.save_for_backward(torch.as_tensor(forces, device=device, dtype=dtype))
        return torch.as_tensor(energy, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, None]:
        """Propagate the upstream gradient into the position gradient.

        Since ``F = -dE/dx``, an upstream loss ``L`` gives
        ``dL/dx = dL/dE * dE/dx = -grad_output * F`` (all in the kJ/mol, nm
        convention, so units are consistent end to end).

        :param ctx: Autograd context holding the cached forces (kJ/mol/nm).
        :param grad_output: Upstream gradient ``dL/dE`` (scalar).
        :return: Gradient w.r.t. each ``forward`` input, in order; ``None`` for
            the non-differentiable ``atoms`` argument."""
        (forces,) = ctx.saved_tensors
        grad_positions = -grad_output * forces  # dL/dx = dL/dE * (-F)
        return grad_positions, None  # None: `atoms` is not differentiable


def potential_energy(positions: Tensor, atoms: Atoms) -> Tensor:
    """Return the differentiable potential energy of ``atoms`` at ``positions``.

    Thin convenience wrapper around :meth:`EnergyFn.apply`.

    :param positions: ``(N, 3)`` atomic positions in nanometre. Set
        ``requires_grad=True`` to obtain forces via ``energy.backward()``.
    :param atoms: ASE ``Atoms`` object with an attached calculator.
    :return: Scalar potential energy in kJ/mol, differentiable w.r.t. ``positions``.

    Example::

        x = torch.tensor(atoms.get_positions() * 0.1, dtype=torch.float64,
                         requires_grad=True)          # A -> nm
        energy = potential_energy(x, atoms)
        energy.backward()
        forces = -x.grad  # kJ/mol/nm"""
    return EnergyFn.apply(positions, atoms)


def positions_to_nm(atoms: Atoms) -> np.ndarray:
    """Return an ``Atoms`` object's positions in nanometre (ASE stores them in A).

    Convenience for building the torch input tensor in the package convention.

    :param atoms: ASE ``Atoms`` object.
    :return: ``(N, 3)`` positions in nanometre."""
    return atoms.get_positions() / NM_TO_ANG
