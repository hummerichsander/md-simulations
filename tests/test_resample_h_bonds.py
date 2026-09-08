"""Tests for the analytic resampling of constrained bond lengths."""

import logging
from pathlib import Path

import numpy as np
import pytest

from md_simulations.config.base import AmberConfig
from md_simulations.sampling.resample_h_bonds import (
    constrained_bond_parameters,
    resample_bond_lengths,
)

from test_engines import _ALA3_PDB

_LOG = logging.getLogger("test")


def _two_bond_system(n_frames: int = 4000, seed: int = 0):
    """Build a synthetic frame stack with two bonds sharing an anchor, plus a spectator atom.

    Atom 0 is the anchor, 1 and 2 the moving ends along orthogonal axes, 3 a spectator that no
    bond names.

    :param n_frames: Number of frames to generate.
    :param seed: Seed for the jitter applied to the whole assembly.
    :return: Positions of shape (n_frames, 4, 3) and the bond index arrays."""
    rng = np.random.default_rng(seed)
    base = np.array([[0.0, 0.0, 0.0], [0.11, 0.0, 0.0], [0.0, 0.1, 0.0], [0.5, 0.5, 0.5]])
    positions = np.broadcast_to(base, (n_frames, 4, 3)).copy()
    # rigid-body jitter, so the bond directions differ between frames
    positions += rng.normal(0.0, 0.01, size=(n_frames, 1, 3))

    return positions, np.array([1, 2]), np.array([0, 0])


class TestResampleBondLengths:
    """Tests for the resample_bond_lengths function."""

    def test_lengths_match_the_requested_distribution(self):
        """Test that the redrawn lengths have the given mean and width.

        :return: None."""
        positions, movers, anchors = _two_bond_system()
        r0 = np.array([0.109, 0.101])
        sigma = np.array([0.0029, 0.0024])

        out = resample_bond_lengths(
            positions, movers, anchors, r0, sigma, np.random.default_rng(1)
        )
        d = np.linalg.norm(out[:, movers] - out[:, anchors], axis=-1)

        assert np.allclose(d.mean(axis=0), r0, atol=2e-4)
        assert np.allclose(d.std(axis=0), sigma, rtol=0.05)

    def test_angles_are_preserved_exactly(self):
        """Test that a radial move leaves the angle at the shared anchor untouched.

        This is the whole reason for moving along the existing bond axis rather than redrawing a
        position: the angular degrees of freedom were never broken and carry real thermal spread.

        :return: None."""
        positions, movers, anchors = _two_bond_system(n_frames=64)

        def angle(x):
            u = x[:, 1] - x[:, 0]
            v = x[:, 2] - x[:, 0]
            u = u / np.linalg.norm(u, axis=-1, keepdims=True)
            v = v / np.linalg.norm(v, axis=-1, keepdims=True)
            return np.arccos(np.clip((u * v).sum(-1), -1.0, 1.0))

        out = resample_bond_lengths(
            positions, movers, anchors, np.array([0.15, 0.05]), np.array([0.003, 0.003]),
            np.random.default_rng(2),
        )

        assert np.allclose(angle(out), angle(positions), atol=1e-12)

    def test_only_the_moving_atoms_change(self):
        """Test that anchors and atoms no bond names are left bit-identical.

        :return: None."""
        positions, movers, anchors = _two_bond_system(n_frames=32)

        out = resample_bond_lengths(
            positions, movers, anchors, np.array([0.109, 0.101]), np.array([0.003, 0.003]),
            np.random.default_rng(3),
        )

        assert np.array_equal(out[:, 0], positions[:, 0])  # anchor
        assert np.array_equal(out[:, 3], positions[:, 3])  # spectator
        assert not np.allclose(out[:, 1], positions[:, 1])

    def test_the_seed_decides_the_draw(self):
        """Test that the output is reproducible under a seed and varies without one.

        :return: None."""
        positions, movers, anchors = _two_bond_system(n_frames=16)
        args = (positions, movers, anchors, np.array([0.109, 0.101]), np.array([0.003, 0.003]))

        a = resample_bond_lengths(*args, np.random.default_rng(7))
        b = resample_bond_lengths(*args, np.random.default_rng(7))
        c = resample_bond_lengths(*args, np.random.default_rng(8))

        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_input_is_not_modified(self):
        """Test that the input array is left alone rather than resampled in place.

        :return: None."""
        positions, movers, anchors = _two_bond_system(n_frames=8)
        before = positions.copy()

        resample_bond_lengths(
            positions, movers, anchors, np.array([0.109, 0.101]), np.array([0.003, 0.003]),
            np.random.default_rng(4),
        )

        assert np.array_equal(positions, before)


class TestConstrainedBondParameters:
    """Tests for the derivation of the constrained bond set from a config."""

    def test_derives_every_constrained_bond_of_a_peptide(self, tmp_path: Path) -> None:
        """Test that the hbonds set is found, with one hydrogen mover per bond.

        :param tmp_path: Pytest temporary directory (used as the data root).
        :return: None."""
        # written outside the guard below, so a filesystem failure cannot pass for a missing XML
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

        try:
            movers, anchors, r0, sigma = constrained_bond_parameters(config, tmp_path, "hbonds")
        except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - needs the shipped XMLs
            pytest.skip(f"amber14 force field unavailable: {exc}")

        assert len(movers) == 17  # the tripeptide's bonds to hydrogen
        assert len(np.unique(movers)) == len(movers)  # one free end each
        assert not set(movers) & set(anchors)  # a mover is never someone's anchor
        assert np.all((r0 > 0.09) & (r0 < 0.12))  # X-H equilibrium lengths, nm
        assert np.all((sigma > 0.002) & (sigma < 0.004))  # sqrt(kT/k) at 300 K

    def test_a_constrained_config_is_refused(self, tmp_path: Path) -> None:
        """Test that a config which still constrains its bonds is rejected, not silently used.

        Under `hbonds` OpenMM deletes the very HarmonicBondForce terms the draw reads, so there
        would be nothing to draw from.

        :param tmp_path: Pytest temporary directory.
        :return: None."""
        config = AmberConfig(
            system="t", output_subdir="t", input_pdb="a.pdb", constraints="hbonds"
        )

        with pytest.raises(ValueError, match="constraints"):
            constrained_bond_parameters(config, tmp_path, "hbonds")
