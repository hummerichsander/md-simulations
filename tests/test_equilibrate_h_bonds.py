"""Tests for the conditional equilibration of constrained bonds."""

import logging
from pathlib import Path

import numpy as np
import openmm.unit as unit
import pytest

from md_simulations.config.base import AmberConfig
from md_simulations.sampling.equilibrate_h_bonds import (
    bond_groups,
    build_frozen_system,
    build_simulation,
    count_frames,
    equilibrate_frames,
    equilibrate_trajectory,
    shard_bounds,
)
from md_simulations.sampling.resample_h_bonds import constrained_bond_parameters

from test_engines import _ALA3_PDB

_LOG = logging.getLogger("test")


def _config(tmp_path: Path, constraints: str = "none") -> AmberConfig:
    """Write the test tripeptide and return a config for it.

    :param tmp_path: Directory used as the data root; the input PDB is written into it.
    :param constraints: The ``constraints`` scheme the config asks for.
    :return: The configuration."""
    (tmp_path / "ala3.pdb").write_text(_ALA3_PDB)
    return AmberConfig(
        system="ala3",
        output_subdir="ala3",
        input_pdb="ala3.pdb",
        forcefield=["amber14-all.xml", "implicit/gbn2.xml"],
        implicit_solvent=True,
        nonbonded_cutoff=2.0,
        constraints=constraints,  # type: ignore[arg-type]
        temperature=330.0,
    )


@pytest.fixture
def built(tmp_path: Path):
    """Build the frozen ala3 system and the bond selection it conditions on.

    The engine adds hydrogens at build time, so the coordinates come from the *build*, not from
    the input PDB -- which has 16 atoms where the built system has 42.

    :param tmp_path: Pytest temporary directory, used as the data root.
    :return: ``(config, system, topology, movers, anchors, r0, positions)``."""
    config = _config(tmp_path)
    try:
        movers, anchors, r0, _sigma = constrained_bond_parameters(config, tmp_path, "hbonds", _LOG)
        built = build_frozen_system(config, tmp_path, movers, _LOG)
    except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - needs the shipped XMLs
        pytest.skip(f"amber14 force field unavailable: {exc}")

    positions = np.array(built.positions.value_in_unit(unit.nanometer))
    return config, built.system, built.topology, movers, anchors, r0, positions


class TestFrozenSystem:
    """Tests for the build that conditions on everything but the constrained bonds."""

    def test_only_the_constrained_free_ends_are_mobile(self, built) -> None:
        """Every atom that is not a constrained bond's free end has zero mass.

        :return: None."""
        _config_, system, _top, movers, _anchors, _r0, _x = built

        mobile = {
            i
            for i in range(system.getNumParticles())
            if system.getParticleMass(i).value_in_unit(unit.dalton) > 0
        }
        assert mobile == set(int(i) for i in movers)
        assert len(mobile) < system.getNumParticles()

    def test_a_constrained_config_is_refused(self, tmp_path: Path) -> None:
        """A build that still carries constraints cannot be frozen or equilibrated.

        :return: None."""
        config = _config(tmp_path, constraints="hbonds")
        with pytest.raises(ValueError, match="constraints"):
            constrained_bond_parameters(config, tmp_path, "hbonds", _LOG)


class TestEquilibrateFrames:
    """Tests for the per-frame equilibration."""

    def _run(self, built, seed: int = 11, n_frames: int = 2):
        """Equilibrate a couple of jittered copies of the input structure.

        :param built: The fixture's build.
        :param seed: Seed for the velocity draws.
        :param n_frames: Number of frames to equilibrate.
        :return: The input and output coordinate stacks."""
        config, system, topology, movers, _anchors, _r0, x = built
        positions = np.repeat(x[None], n_frames, axis=0)
        simulation = build_simulation(
            system, topology, config.temperature, 1.0, 0.0005, "Reference", seed
        )
        out = equilibrate_frames(
            positions,
            system,
            topology,
            movers,
            temperature=config.temperature,
            timestep=0.0005,
            equilibration_time=0.005,  # 10 steps: enough to move, fast enough for a test
            seed=seed,
            simulation=simulation,
        )
        return positions, out

    def test_frozen_atoms_come_back_bit_identical(self, built) -> None:
        """The conditioning coordinates are preserved exactly, not merely closely.

        :return: None."""
        _c, _s, _t, movers, _a, _r0, _x = built
        positions, out = self._run(built)

        frozen = np.setdiff1d(np.arange(positions.shape[1]), movers)
        assert np.array_equal(out[:, frozen], positions[:, frozen])

    def test_the_free_ends_move(self, built) -> None:
        """The atoms being equilibrated actually leave their input positions.

        :return: None."""
        _c, _s, _t, movers, _a, _r0, _x = built
        positions, out = self._run(built)

        assert not np.allclose(out[:, movers], positions[:, movers])

    def test_the_same_seed_reproduces_the_output(self, built) -> None:
        """Seeding covers the velocity draw as well as the integrator.

        :return: None."""
        _p1, a = self._run(built, seed=11)
        _p2, b = self._run(built, seed=11)
        assert np.array_equal(a, b)

    def test_a_different_seed_gives_a_different_output(self, built) -> None:
        """Two frames of identical input do not share a draw.

        :return: None."""
        _p1, a = self._run(built, seed=11)
        _p2, b = self._run(built, seed=12)
        assert not np.allclose(a, b)

    def test_identical_frames_are_not_equilibrated_identically(self, built) -> None:
        """Frames are seeded per frame, so a repeated input does not give a repeated output.

        :return: None."""
        _positions, out = self._run(built, n_frames=2)
        assert not np.allclose(out[0], out[1])


class TestEquilibrateTrajectory:
    """Tests for the streaming, file-to-file path."""

    def test_round_trip(self, built, tmp_path: Path) -> None:
        """A written trajectory comes back with its frame count and its frozen atoms intact.

        :return: None."""
        import mdtraj as md

        config, system, topology, movers, anchors, r0, x = built
        rng = np.random.default_rng(0)
        positions = np.repeat(x[None], 6, axis=0) + rng.normal(0, 0.005, (6, len(x), 3))

        top = md.Topology.from_openmm(topology)
        top_path = tmp_path / "built.pdb"
        md.Trajectory(x.astype(np.float32), top).save_pdb(str(top_path))
        traj_path = tmp_path / "in.trr"
        md.Trajectory(positions.astype(np.float32), top).save(str(traj_path))

        simulation = build_simulation(
            system, topology, config.temperature, 1.0, 0.0005, "Reference", 3
        )
        out_path = tmp_path / "out" / "equilibrated.trr"
        n, before, after, final = equilibrate_trajectory(
            str(traj_path),
            str(top_path),
            out_path,
            simulation,
            movers,
            anchors,
            r0,
            temperature=config.temperature,
            timestep=0.0005,
            equilibration_time=0.005,
            seed=3,
            chunk=4,  # smaller than the trajectory, so the chunk seam is exercised
            groups=bond_groups(topology, movers, anchors),
        )

        assert n == 6
        assert final["width/out_pm"] > 0
        assert {"energy/in_kJ_mol", "energy/out_kJ_mol", "energy/drop_kJ_mol"} <= set(final)
        assert any(key.startswith("width/N-H") for key in final)
        assert before.shape == after.shape == movers.shape
        out = md.load(str(out_path), top=top)
        assert out.n_frames == 6
        assert count_frames(str(out_path), str(top_path)) == 6

        frozen = np.setdiff1d(np.arange(len(x)), movers)
        np.testing.assert_allclose(
            out.xyz[:, frozen], positions[:, frozen].astype(np.float32), rtol=0, atol=1e-6
        )


class TestShardBounds:
    """Tests for the contiguous split the array job uses."""

    def test_shards_tile_the_selection_in_order(self) -> None:
        """The shards are contiguous, disjoint, in order, and cover everything.

        :return: None."""
        bounds = [shard_bounds(1000, i, 7) for i in range(7)]

        assert bounds[0][0] == 0
        assert bounds[-1][1] == 1000
        for (_, stop), (start, _) in zip(bounds, bounds[1:]):
            assert stop == start
        assert sum(stop - start for start, stop in bounds) == 1000

    def test_a_sub_range_is_split_not_the_whole_file(self) -> None:
        """``--start``/``--stop`` narrow what is sharded.

        :return: None."""
        bounds = [shard_bounds(1000, i, 4, start=100, stop=500) for i in range(4)]
        assert bounds[0][0] == 100
        assert bounds[-1][1] == 500

    def test_a_stop_past_the_end_is_clamped(self) -> None:
        """A selection longer than the trajectory stops at the trajectory.

        :return: None."""
        assert shard_bounds(10, 0, 1, stop=99) == (0, 10)

    def test_an_out_of_range_shard_is_refused(self) -> None:
        """A shard index outside the array is an error, not a silent empty block.

        :return: None."""
        with pytest.raises(ValueError, match="out of range"):
            shard_bounds(100, 3, 3)

    def test_an_empty_selection_is_refused(self) -> None:
        """A selection with no frames in it is an error.

        :return: None."""
        with pytest.raises(ValueError, match="empty selection"):
            shard_bounds(100, 0, 2, start=50, stop=50)
