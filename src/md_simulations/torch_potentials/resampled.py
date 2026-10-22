"""The exact density of a trajectory whose constrained bonds were redrawn by ``resample_h_bonds``.

``sampling/resample_h_bonds.py`` takes frames of a constrained simulation and slides each
constrained bond's free atom along its axis to a length ``d ~ N(r0, sigma^2)``, ``sigma =
sqrt(kT / k)``. The result is not the unconstrained force field's Boltzmann distribution -- that
would shift each length by the other forces along the bond (~0.7 sigma for polar hydrogens) and by
the ``d^2`` Jacobian -- but it is an exactly known density. With ``pi`` sliding every free atom back
to ``r0``, which recovers the constrained frame,

    -log p(x) = beta U(pi(x)) + sum_bonds [ (d - r0)^2 / (2 sigma^2) + 2 log d ] + const,

since the constrained ensemble puts the free atom's direction on the sphere of radius ``r0`` and
the draw spreads it radially with density ``N(d) / d^2`` per unit volume. ``U`` is the
unconstrained potential: at ``d = r0`` its constrained bond terms vanish, so it equals the
constrained one there. Integrating out the lengths returns the constrained ensemble exactly.

:class:`ResampledBondPotential` returns ``kT`` times that, in kJ/mol, so it drops in wherever a
potential is scored (reweighting, energy distributions, forces). The mass-metric (Fixman) factor
of the constrained dynamics, which depends only on the angles between constraints sharing an atom,
is left out."""

from collections.abc import Callable
from pathlib import Path

import mdtraj as md
import numpy as np
import torch
from torch import Tensor

from md_simulations.config import load_config, resolve_data_root
from md_simulations.torch_potentials.build import build_potential

BOLTZMANN = 8.314462618e-3  # kJ/mol/K


class ResampledBondPotential:
    """A potential, scored at constrained bond lengths, plus the resampler's bond-length density."""

    def __init__(
        self,
        potential: Callable[..., Tensor],
        movers: np.ndarray,
        anchors: np.ndarray,
        r0: np.ndarray,
        sigma: np.ndarray,
        temperature: float,
    ) -> None:
        """Wrap a potential with the resampled bonds' projection and density.

        :param potential: The unconstrained potential of the system, nm in, kJ/mol out.
        :param movers: Index of the atom each resampled bond moves.
        :param anchors: Index of the atom its length is measured from.
        :param r0: Equilibrium length per bond, in nm.
        :param sigma: Gaussian width per bond, in nm.
        :param temperature: Temperature the widths were drawn at, in K."""
        self.potential = potential
        self.movers = torch.as_tensor(movers, dtype=torch.long)
        self.anchors = torch.as_tensor(anchors, dtype=torch.long)
        self.r0 = torch.as_tensor(r0)
        self.sigma = torch.as_tensor(sigma)
        self.kT = BOLTZMANN * temperature

    def __call__(
        self,
        positions: Tensor,
        unitcell_lengths: Tensor | None = None,
        unitcell_angles: Tensor | None = None,
    ) -> Tensor:
        """Return ``-kT log p`` of the resampled ensemble, up to a constant.

        :param positions: ``(N, 3)`` or ``(F, N, 3)`` positions in nm.
        :param unitcell_lengths: Box edge lengths in nm, passed to the wrapped potential.
        :param unitcell_angles: Box angles in degrees, passed to the wrapped potential.
        :return: Energy in kJ/mol, a scalar or ``(F,)``, differentiable w.r.t. ``positions``."""
        r0 = self.r0.to(positions)
        sigma = self.sigma.to(positions)

        # molecules are whole, as in the resampler, so the bond axis needs no minimum image.
        axis = positions[..., self.movers, :] - positions[..., self.anchors, :]
        d = axis.norm(dim=-1)
        projected = positions.clone()
        projected[..., self.movers, :] = (
            positions[..., self.anchors, :] + (r0 / d)[..., None] * axis
        )

        bonds = (((d - r0) / sigma) ** 2 / 2 + 2 * torch.log(d)).sum(-1)
        u = self.potential(projected, unitcell_lengths, unitcell_angles)

        return u + self.kT * bonds


def build_resampled_bond_potential_from_file(
    topology: md.Topology,
    path: str | Path,
    data_root: str | Path | None = None,
    *,
    constrained_as: str = "hbonds",
    platform: str | None = None,
    quiet: bool = True,
) -> ResampledBondPotential:
    """Build the resampled ensemble's potential from the unconstrained config the draw used.

    The bond set, ``r0`` and ``sigma`` come from
    :func:`~md_simulations.sampling.resample_h_bonds.constrained_bond_parameters`, the same call
    the resampler makes, so the density matches the draw exactly.

    :param topology: mdtraj ``Topology`` whose atoms match the config's system.
    :param path: The ``constraints: none`` config the trajectory was resampled with.
    :param data_root: Root the config's paths resolve against.
    :param constrained_as: The constraint scheme the reference trajectory was produced with.
    :param platform: OpenMM platform name, or ``None`` for the default.
    :param quiet: Suppress the verbose build output.
    :return: The potential."""
    from md_simulations.sampling.resample_h_bonds import constrained_bond_parameters

    config = load_config(path)
    root = resolve_data_root(config, None if data_root is None else str(data_root))
    movers, anchors, r0, sigma = constrained_bond_parameters(config, root, constrained_as)
    potential = build_potential(topology, config, root, platform=platform, quiet=quiet)

    return ResampledBondPotential(potential, movers, anchors, r0, sigma, config.temperature)
