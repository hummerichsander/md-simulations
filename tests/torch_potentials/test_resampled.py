"""Tests for the density of a trajectory whose constrained bonds were redrawn."""

import logging
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest
import torch
from openmm.unit import nanometer

from md_simulations.config.base import AmberConfig
from md_simulations.engines import build_engine
from md_simulations.sampling.resample_h_bonds import resample_bond_lengths
from md_simulations.torch_potentials.resampled import BOLTZMANN, ResampledBondPotential


def _directional(positions: torch.Tensor, unitcell_lengths=None, unitcell_angles=None):
    """A stand-in potential that sees every coordinate, in kJ/mol.

    :param positions: ``(F, N, 3)`` positions in nm.
    :return: ``(F,)`` energies."""
    return (positions**2).sum((-1, -2)) + positions[..., 0].sum(-1)


def _two_bonds() -> dict:
    """Two bonds sharing nothing: atoms 1 and 3 move, measured from 0 and 2.

    :return: The bond parameters, at 300 K."""
    return {
        "movers": np.array([1, 3]),
        "anchors": np.array([0, 2]),
        "r0": np.array([0.10, 0.11]),
        "sigma": np.array([0.003, 0.004]),
        "temperature": 300.0,
    }


class TestResampledBondPotential:
    """Tests for the projection and the bond-length density."""

    def test_scores_the_projected_frame_plus_the_bond_density(self):
        """The energy is the potential at ``r0`` plus ``kT`` times the Gaussian and Jacobian terms."""
        bonds = _two_bonds()
        pot = ResampledBondPotential(_directional, **bonds)
        x = torch.randn(5, 4, 3) * 0.1

        axis = x[:, bonds["movers"]] - x[:, bonds["anchors"]]
        d = axis.norm(dim=-1)
        r0, sigma = torch.tensor(bonds["r0"]), torch.tensor(bonds["sigma"])
        projected = x.clone()
        projected[:, bonds["movers"]] = x[:, bonds["anchors"]] + r0[:, None] * axis / d[..., None]
        expected = _directional(projected) + BOLTZMANN * 300.0 * (
            ((d - r0) / sigma) ** 2 / 2 + 2 * torch.log(d)
        ).sum(-1)

        assert torch.allclose(pot(x), expected)

    def test_a_radial_move_changes_only_the_bond_density(self):
        """Sliding a free atom along its bond leaves the projected frame unchanged."""
        bonds = _two_bonds()
        pot = ResampledBondPotential(_directional, **bonds)
        x = torch.randn(1, 4, 3) * 0.1
        y = x.clone()
        y[:, 1] = x[:, 0] + 1.3 * (x[:, 1] - x[:, 0])

        d_x, d_y = ((z[:, 1] - z[:, 0]).norm(dim=-1) for z in (x, y))
        kT, r0, sigma = BOLTZMANN * 300.0, 0.10, 0.003

        def term(d):
            return kT * (((d - r0) / sigma) ** 2 / 2 + 2 * torch.log(d))

        assert torch.allclose(pot(y) - pot(x), term(d_y) - term(d_x))

    def test_stein_identity_holds_on_resampled_frames(self):
        """On frames the resampler drew, ``E[s_i . (x_i - x_j)] = -3`` for every bonded pair.

        With a flat potential the constrained ensemble puts each free atom uniformly on the sphere
        of radius ``r0``; the resampler then spreads it radially, so the frames are exact samples of
        the potential's density and its score must satisfy Stein's identity."""
        rng = np.random.default_rng(0)
        bonds = _two_bonds()
        F = 200_000

        positions = np.zeros((F, 4, 3))
        positions[:, 2] = rng.normal(size=(F, 3))
        for mover, anchor, r0 in zip(bonds["movers"], bonds["anchors"], bonds["r0"]):
            u = rng.normal(size=(F, 3))
            positions[:, mover] = (
                positions[:, anchor] + r0 * u / np.linalg.norm(u, axis=-1)[:, None]
            )
        positions = resample_bond_lengths(
            positions, bonds["movers"], bonds["anchors"], bonds["r0"], bonds["sigma"], rng
        )

        pot = ResampledBondPotential(lambda x, *_: 0.0 * x.sum((-1, -2)), **bonds)
        x = torch.tensor(positions, requires_grad=True)
        pot(x).sum().backward()
        score = -x.grad / (BOLTZMANN * 300.0)

        for i, j in ((1, 0), (0, 1), (3, 2), (2, 3)):
            stein = (score[:, i] * (x[:, i] - x[:, j])).sum(-1).detach()
            assert abs(stein.mean().item() + 3.0) < 4 * stein.std().item() / F**0.5


class TestBuild:
    """Tests for building the potential from the config the resampler used."""

    def test_matches_the_force_field_at_equilibrium_lengths(self, tmp_path: Path, write_yaml):
        """At ``d = r0`` the energy is the unconstrained force field's plus a constant.

        :param tmp_path: Pytest temporary directory (used as the data root).
        :param write_yaml: Fixture writing a config dict to a YAML file."""
        from test_engines import _ALA3_PDB

        from md_simulations.torch_potentials import (
            build_potential_from_file,
            build_resampled_bond_potential_from_file,
        )

        (tmp_path / "ala3.pdb").write_text(_ALA3_PDB)
        config = AmberConfig(
            system="ala3",
            output_subdir="ala3",
            input_pdb="ala3.pdb",
            forcefield=["amber14-all.xml", "implicit/gbn2.xml"],
            implicit_solvent=True,
            nonbonded_cutoff=2.0,
            constraints="none",
        )
        path = write_yaml(config.model_dump(mode="json"))

        try:
            # the engine adds the hydrogens, so the system's own topology and positions are used.
            built = build_engine(config, tmp_path, logging.getLogger("test")).build_system()
            top = md.Topology.from_openmm(built.topology)
            pot = build_resampled_bond_potential_from_file(top, path, data_root=tmp_path)
        except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - needs the shipped XMLs
            pytest.skip(f"amber14 force field unavailable: {exc}")
        flexible = build_potential_from_file(top, path, data_root=tmp_path)

        x = torch.tensor(np.array(built.positions.value_in_unit(nanometer)))[None]
        axis = x[:, pot.movers] - x[:, pot.anchors]
        x[:, pot.movers] = x[:, pot.anchors] + pot.r0[:, None] * axis / axis.norm(dim=-1)[..., None]

        jacobian = 2 * pot.kT * torch.log(pot.r0).sum()
        assert torch.allclose(pot(x), flexible(x) + jacobian, atol=1e-4)
