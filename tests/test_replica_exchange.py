import logging
import math
import struct
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pytest
from beamtime.format.run import read_effective_metrics, read_run_info
from beamtime.format.tree import iter_runs
from openmm import Vec3
from pydantic import TypeAdapter, ValidationError

from md_simulations.config.base import AmberConfig, ReplicaExchangeConfig, SimulationConfig
from md_simulations.core import remd
from md_simulations.core.replica_exchange import (
    ExchangeStats,
    attempt_swaps,
    attempted_pairs,
    geometric_ladder,
    swap_probability,
)
from md_simulations.core.forces import FORCES_FILE, ForceWriter, count_frames
from md_simulations.core.remd import ReplicaExchange, rescale_velocities
from md_simulations.engines.base import BuiltSystem, OpenMMEngine

from conftest import SAMPLE_CONFIGS

TEMPERATURES = [300.0, 400.0, 550.0]
HEADER = '#"Step","Potential Energy (kJ/mole)"'


def _argon_pair() -> BuiltSystem:
    """Build a 2-particle harmonic system without data files.

    :return: Topology, system and positions."""
    topology = app.Topology()
    residue = topology.addResidue("AR2", topology.addChain())
    argon = app.Element.getBySymbol("Ar")
    topology.addAtom("A1", argon, residue)
    topology.addAtom("A2", argon, residue)

    system = openmm.System()
    system.addParticle(39.948)
    system.addParticle(39.948)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.2, 1000.0)
    system.addForce(bond)
    positions = [Vec3(0, 0, 0), Vec3(0.2, 0, 0)] * unit.nanometer
    return BuiltSystem(topology=topology, system=system, positions=positions)


class ArgonEngine(OpenMMEngine):
    """OpenMM engine on the argon pair."""

    def build_system(self) -> BuiltSystem:
        """Build the argon pair.

        :return: The built system."""
        return _argon_pair()


def _config(**rex) -> AmberConfig:
    """Build a tiny T-REMD config: 40 steps, a frame every 4, an exchange every 8.

    :param rex: Fields of the replica-exchange block to override.
    :return: The config."""
    block = dict(temperatures=TEMPERATURES, exchange_interval=8, seed=1, checkpoint_interval=16)
    return AmberConfig(
        system="argon",
        output_subdir="argon/pair_remd",
        input_pdb="a.pdb",
        num_steps=40,
        output_freq=4,
        report_interval=4,
        temperature=300.0,
        replica_exchange=block | rex,
    )


def _engine(tmp_path: Path, config: AmberConfig | None = None, resume: bool = False):
    """Make the argon engine for a config.

    :param tmp_path: Pytest temporary directory.
    :param config: The config; a default T-REMD config if omitted.
    :param resume: Whether to resume from a checkpoint.
    :return: The engine."""
    return ArgonEngine(config or _config(), tmp_path, logging.getLogger("test"), resume)


def _rows(path: Path) -> list[str]:
    """Read the lines of a text file.

    :param path: The file.
    :return: Its lines."""
    return path.read_text().splitlines()


def _dcd_frames(path: Path) -> int:
    """Read the frame count of a DCD and check that its size agrees with it.

    :param path: The DCD file of the argon pair.
    :return: The number of frames."""
    data = path.read_bytes()
    (frames,) = struct.unpack_from("<i", data, 8)
    assert len(data) == 276 + frames * 3 * (8 + 4 * 2)
    return frames


def _outputs(root: Path) -> dict[str, list]:
    """Collect the step columns and row counts of the outputs of a T-REMD run.

    :param root: The output directory.
    :return: Step columns of the energies and exchanges files, and the frame counts."""
    steps = lambda path: [int(row.split(",")[0]) for row in _rows(path)[1:]]  # noqa: E731
    return {
        "energies": steps(root / "energies.csv"),
        "exchanges": steps(root / "exchanges.csv"),
        "forces": count_frames(root / FORCES_FILE),
        "frames": _dcd_frames(root / "trajectory.dcd"),
        "other": steps(root / "T01_400K/energies.csv"),
    }


def test_equal_temperatures_always_accept() -> None:
    """Swapping at one temperature never changes the weight."""
    assert swap_probability(2.0, 2.0, 1.0, 100.0) == 1.0


def test_swap_probability_hand_computed() -> None:
    """A swap that raises the joint weight's denominator is accepted with exp(delta)."""
    # (2 - 1) * (1 - 3) = -2
    assert swap_probability(2.0, 1.0, 1.0, 3.0) == pytest.approx(math.exp(-2.0))


def test_downhill_swaps_always_accept_and_never_overflow() -> None:
    """The colder slot holding the higher energy swaps with certainty, however large the gap."""
    assert swap_probability(2.0, 1.0, 5.0, 3.0) == 1.0
    assert swap_probability(1.0, 0.0, 1e9, 0.0) == 1.0
    assert swap_probability(1.0, 0.0, 0.0, 1e9) == 0.0


def test_parity_alternates_and_replica_at_stays_a_permutation() -> None:
    """Even and odd pairs are attempted in turn; every swap permutes the replicas."""
    betas = [1.0, 0.8, 0.6, 0.4, 0.2]
    assert attempted_pairs(0, 5) == [0, 2]
    assert attempted_pairs(1, 5) == [1, 3]

    # strictly falling energies: every attempted swap is downhill.
    energies = [4.0, 3.0, 2.0, 1.0, 0.0]
    rng = np.random.default_rng(0)
    replica_at, accepted = attempt_swaps(energies, betas, [0, 1, 2, 3, 4], 0, rng)
    assert (replica_at, accepted) == ([1, 0, 3, 2, 4], [0, 2])
    replica_at, accepted = attempt_swaps(energies, betas, replica_at, 1, rng)
    assert (replica_at, accepted) == ([1, 3, 0, 4, 2], [1, 3])

    rng = np.random.default_rng(1)
    replica_at, parity = list(range(5)), 0
    for _ in range(200):
        energies = list(rng.normal(size=5) * 3)
        replica_at, _ = attempt_swaps(energies, betas, replica_at, parity, rng)
        parity = 1 - parity
        assert sorted(replica_at) == list(range(5))


def test_round_trips_on_a_scripted_sequence() -> None:
    """Only lowest → highest → lowest counts; starting on top or in the middle does not."""
    stats = ExchangeStats(3)
    stats.touch_ends([0, 1, 2])
    for arrangement, trips in [
        ([1, 0, 2], 0),
        ([1, 2, 0], 0),  # replica 0 reaches the top coming from the bottom
        ([0, 2, 1], 1),  # ... and is back at the bottom
        ([1, 0, 2], 2),  # replica 1 sat at the bottom, went to the top, and is back
    ]:
        stats.touch_ends(arrangement)
        assert stats.round_trips == trips

    stats = ExchangeStats(3)
    stats.touch_ends([0, 1, 2])
    for arrangement in ([2, 1, 0], [0, 1, 2]):
        stats.touch_ends(arrangement)
    # replica 2 started on top: its trip to the bottom is half a trip. Replica 0 made a full one.
    assert stats.round_trips == 1

    restored = ExchangeStats.from_dict(stats.to_dict())
    assert restored == stats


def test_stats_record_acceptance() -> None:
    """Attempts and acceptances are counted for the pairs of the parity attempted."""
    stats = ExchangeStats(4)
    stats.record(0, [0], [1, 0, 2, 3])
    stats.record(1, [], [1, 0, 2, 3])
    assert stats.attempts == [1, 1, 1]
    assert stats.accepted == [1, 0, 0]
    assert [stats.acceptance(k) for k in range(3)] == [1.0, 0.0, 0.0]


def test_geometric_ladder_endpoints() -> None:
    """The ladder starts and ends exactly at the given temperatures with a constant ratio."""
    ladder = geometric_ladder(300.0, 500.0, 5)
    assert ladder[0] == 300.0 and ladder[-1] == 500.0
    ratios = [b / a for a, b in zip(ladder, ladder[1:])]
    assert ratios == pytest.approx([ratios[0]] * 4)
    with pytest.raises(ValueError):
        geometric_ladder(300.0, 500.0, 1)


def test_velocity_rescale_scales_kinetic_energy() -> None:
    """Scaling the velocities by sqrt(T_new / T_old) scales the kinetic energy by T_new / T_old."""
    built = _argon_pair()
    simulation = app.Simulation(
        built.topology, built.system, openmm.LangevinIntegrator(300, 1.0, 0.002)
    )
    simulation.context.setPositions(built.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 7)

    def kinetic() -> float:
        return simulation.context.getState(getEnergy=True).getKineticEnergy()._value

    before = kinetic()
    rescale_velocities(simulation, math.sqrt(550.0 / 300.0))
    assert kinetic() == pytest.approx(before * 550.0 / 300.0)


def test_exchange_moves_temperatures_not_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted swap sets the integrators' temperatures, rescales velocities, keeps positions.

    :param tmp_path: Pytest temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture."""
    built = _argon_pair()
    driver = ReplicaExchange(
        topology=built.topology,
        system=built.system,
        positions=built.positions,
        box_vectors=None,
        config=_config(),
        output_dir=tmp_path,
        runs=[None] * 3,  # type: ignore[list-item]
        logger=logging.getLogger("test"),
    )
    driver.writers = [None] * 3
    driver._exchanges = open(tmp_path / "exchanges.csv", "w")
    for r, simulation in enumerate(driver.simulations):
        simulation.context.setVelocitiesToTemperature(TEMPERATURES[r] * unit.kelvin, r)
    monkeypatch.setattr(remd, "attempt_swaps", lambda e, b, replica_at, p, rng: ([1, 0, 2], [0]))

    def snapshot(r: int) -> tuple[float, np.ndarray]:
        state = driver.simulations[r].context.getState(getEnergy=True, getPositions=True)
        return state.getKineticEnergy()._value, state.getPositions(asNumpy=True)._value

    (ke0, pos0), (ke1, pos1) = snapshot(0), snapshot(1)
    driver._exchange(driver._sample())

    assert driver.replica_at == [1, 0, 2]
    assert [s.integrator.getTemperature()._value for s in driver.simulations] == [400, 300, 550]
    (new_ke0, new_pos0), (new_ke1, new_pos1) = snapshot(0), snapshot(1)
    assert new_ke0 == pytest.approx(ke0 * 400 / 300)
    assert new_ke1 == pytest.approx(ke1 * 300 / 400)
    assert np.array_equal(new_pos0, pos0) and np.array_equal(new_pos1, pos1)
    driver._exchanges.close()


@pytest.mark.parametrize("other_output", ["energies", "full", "none"])
def test_remd_smoke(tmp_path: Path, beamtime_logdir: Path, other_output: str) -> None:
    """A T-REMD run writes the plain-MD files at the target and the per-temperature extras.

    :param tmp_path: Pytest temporary directory.
    :param beamtime_logdir: The temporary log dir.
    :param other_output: What the non-target temperatures write (parametrised)."""
    config = _config(other_output=other_output, other_output_freq=8)
    root = _engine(tmp_path, config).run()

    for name in (
        "trajectory.dcd",
        "energies.csv",
        FORCES_FILE,
        "initial_structure.pdb",
        "final_structure.pdb",
        "simulation_summary.txt",
        "exchanges.csv",
    ):
        assert (root / name).stat().st_size > 0, name
    energies = _rows(root / "energies.csv")
    assert energies[0] == HEADER
    assert [int(r.split(",")[0]) for r in energies[1:]] == list(range(4, 41, 4))
    assert all(math.isfinite(float(r.split(",")[1])) for r in energies[1:])
    assert _dcd_frames(root / "trajectory.dcd") == 10
    assert np.load(root / FORCES_FILE).shape == (10, 2, 3)

    exchanges = _rows(root / "exchanges.csv")
    assert exchanges[0] == (
        "step,parity,replica_at_T00,replica_at_T01,replica_at_T02,"
        "accepted_T00_T01,accepted_T01_T02"
    )
    assert len(exchanges) == 1 + 5
    for i, row in enumerate(exchanges[1:]):
        step, parity, *rest = row.split(",")
        assert (int(step), int(parity)) == (8 * (i + 1), i % 2)
        assert sorted(int(r) for r in rest[:3]) == [0, 1, 2]
        flags = rest[3:]
        assert flags[1 - int(parity)] == ""
        assert flags[int(parity)] in ("0", "1")

    summary = (root / "simulation_summary.txt").read_text()
    assert "[Replica exchange]" in summary and "Round trips" in summary
    assert "Acceptance 300.0-400.0 K" in summary

    for k, temperature in enumerate(TEMPERATURES[1:], start=1):
        folder = root / f"T{k:02d}_{temperature:.0f}K"
        if other_output == "none":
            assert not folder.exists()
            continue
        assert _rows(folder / "energies.csv")[0] == HEADER
        assert len(_rows(folder / "energies.csv")) == 1 + 5
        assert (folder / "trajectory.dcd").exists() == (other_output == "full")
        assert (folder / FORCES_FILE).exists() == (other_output == "full")
    if other_output == "full":
        assert _dcd_frames(root / "T02_550K/trajectory.dcd") == 5
        assert count_frames(root / "T02_550K" / FORCES_FILE) == 5

    run_dirs = list(iter_runs(beamtime_logdir))
    assert {d.parent for d in run_dirs} == {beamtime_logdir / "argon/pair_remd"}
    assert sorted(read_run_info(d).name for d in run_dirs) == [
        "T00_300K",
        "T01_400K",
        "T02_550K",
    ]
    for d in run_dirs:
        assert read_run_info(d).status == "finished"
        rows = [r for r in read_effective_metrics(d) if "energy/potential" in r]
        assert [r["_step"] for r in rows] == list(range(4, 41, 4))
        assert all(r["temperature"] > 0 for r in rows)
        assert all("exchange/round_trips" in r for r in rows)
        has_upper = read_run_info(d).name != "T02_550K"
        assert all(("exchange/acceptance" in r) == has_upper for r in rows)


def test_target_files_match_plain_md(tmp_path: Path) -> None:
    """The target temperature's files have the layout and formats of a plain-MD run.

    :param tmp_path: Pytest temporary directory."""
    remd_root = _engine(tmp_path / "remd").run()
    plain_config = _config().model_copy(update={"replica_exchange": None})
    plain_root = _engine(tmp_path / "plain", plain_config).run()

    remd_lines, plain_lines = _rows(remd_root / "energies.csv"), _rows(plain_root / "energies.csv")
    assert len(remd_lines) == len(plain_lines)
    assert [len(line.split()) for line in remd_lines[1:]] == [
        len(line.split()) for line in plain_lines[1:]
    ]
    assert np.load(remd_root / FORCES_FILE).shape == np.load(plain_root / FORCES_FILE).shape
    assert _rows(remd_root / "energies.csv")[0] == _rows(plain_root / "energies.csv")[0]
    # the header holds a creation time at bytes 100-260; everything else must agree.
    remd_dcd = (remd_root / "trajectory.dcd").read_bytes()
    plain_dcd = (plain_root / "trajectory.dcd").read_bytes()
    assert (remd_dcd[:100], remd_dcd[260:276]) == (plain_dcd[:100], plain_dcd[260:276])
    assert len(remd_dcd) == len(plain_dcd)


def test_checkpoint_needs_resume_flag(tmp_path: Path) -> None:
    """A checkpoint is never overwritten, and resuming needs one.

    :param tmp_path: Pytest temporary directory."""
    with pytest.raises(FileNotFoundError):
        _engine(tmp_path, resume=True).run()
    _engine(tmp_path).run()
    with pytest.raises(FileExistsError, match="--resume"):
        _engine(tmp_path).run()


def test_resume_matches_a_straight_run(
    tmp_path: Path, beamtime_logdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run killed after its checkpoint and resumed has the outputs of one that was not.

    :param tmp_path: Pytest temporary directory.
    :param beamtime_logdir: The temporary log dir.
    :param monkeypatch: Pytest monkeypatch fixture."""
    config = _config(other_output="energies", other_output_freq=8)
    straight = _engine(tmp_path / "straight", config).run()
    straight_logs = {
        read_run_info(d).name: [r["_step"] for r in read_effective_metrics(d)]
        for d in iter_runs(beamtime_logdir)
    }

    original = ReplicaExchange._advance

    def die_after_step_24(self: ReplicaExchange, n_steps: int) -> None:
        if self.step >= 24:
            raise RuntimeError("killed")
        original(self, n_steps)

    with monkeypatch.context() as patch:
        patch.setattr(ReplicaExchange, "_advance", die_after_step_24)
        with pytest.raises(RuntimeError, match="killed"):
            _engine(tmp_path / "resumed", config).run()
    # step 24 is past the checkpoint at 16, and was written before the kill.
    assert _outputs(tmp_path / "resumed/argon/pair_remd")["energies"][-1] == 24
    assert {read_run_info(d).status for d in iter_runs(beamtime_logdir)} >= {"failed"}

    resumed = _engine(tmp_path / "resumed", config, resume=True).run()
    assert _outputs(resumed) == _outputs(straight)
    assert (resumed / "final_structure.pdb").exists()

    # one run per temperature for each of the two simulations: the resume continued, not added.
    names = sorted(read_run_info(d).name for d in iter_runs(beamtime_logdir))
    assert names == sorted(["T00_300K", "T01_400K", "T02_550K"] * 2)
    for d in iter_runs(beamtime_logdir):
        info = read_run_info(d)
        assert info.status == "finished"
        if len(info.attempts) == 2:
            steps = [r["_step"] for r in read_effective_metrics(d)]
            assert steps == straight_logs[info.name]


def test_resume_cuts_the_files_back(tmp_path: Path) -> None:
    """Text outputs and the DCD are cut back to the checkpointed step before appending.

    :param tmp_path: Pytest temporary directory."""
    root = _engine(tmp_path, _config(checkpoint_interval=24)).run()
    state = remd.read_checkpoint(root)
    assert state is not None and state["step"] == 24

    # the run reached step 40; the checkpoint is at 24, so resuming has 16 steps to redo.
    before = _outputs(root)
    assert before["energies"][-1] == 40
    _engine(tmp_path, _config(checkpoint_interval=24), resume=True).run()
    assert _outputs(root) == before
    assert remd.read_checkpoint(root)["step"] == 24  # type: ignore[index]


@pytest.mark.parametrize("name", [FORCES_FILE, "energies.csv", "exchanges.csv"])
def test_resume_rejects_files_behind_the_checkpoint(tmp_path: Path, name: str) -> None:
    """Files that lost rows the checkpoint counts on (a node crash) stop the resume.

    :param tmp_path: Pytest temporary directory.
    :param name: The file that is behind (parametrised)."""
    config = _config(checkpoint_interval=24)
    root = _engine(tmp_path, config).run()
    path = root / name
    if name == FORCES_FILE:
        writer = ForceWriter(path, np.load(path).shape[1], append=True)
        writer.truncate(2)
        writer.close()
    else:
        path.write_text("\n".join(_rows(path)[:3]) + "\n")
    with pytest.raises(ValueError):
        _engine(tmp_path, config, resume=True).run()


def test_resume_rejects_a_changed_output_grid(tmp_path: Path) -> None:
    """Changing an interval between run and resume would cut the files in the wrong place.

    :param tmp_path: Pytest temporary directory."""
    config = _config(checkpoint_interval=24)
    _engine(tmp_path, config).run()
    changed = config.model_copy(update={"output_freq": 8})
    with pytest.raises(ValueError, match="output grid"):
        _engine(tmp_path, changed, resume=True).run()


def test_resume_may_extend_num_steps(tmp_path: Path) -> None:
    """A resume with a larger num_steps continues past the original end.

    :param tmp_path: Pytest temporary directory."""
    config = _config(checkpoint_interval=40)
    root = _engine(tmp_path, config).run()
    longer = config.model_copy(update={"num_steps": 80})
    _engine(tmp_path, longer, resume=True).run()
    assert _outputs(root)["energies"] == list(range(4, 81, 4))
    assert _outputs(root)["exchanges"] == list(range(8, 81, 8))


def test_resume_rejects_another_ladder(tmp_path: Path) -> None:
    """A checkpoint of one ladder cannot be resumed with another.

    :param tmp_path: Pytest temporary directory."""
    _engine(tmp_path).run()
    other = _config(temperatures=[300.0, 400.0, 600.0])
    with pytest.raises(ValueError, match="ladder"):
        _engine(tmp_path, other, resume=True).run()


def test_barostat_is_rejected(tmp_path: Path) -> None:
    """An NPT system cannot be replica-exchanged with the potential-energy criterion.

    :param tmp_path: Pytest temporary directory."""
    built = _argon_pair()
    built.system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin))
    with pytest.raises(NotImplementedError, match="barostat"):
        ReplicaExchange(
            topology=built.topology,
            system=built.system,
            positions=built.positions,
            box_vectors=None,
            config=_config(),
            output_dir=tmp_path,
            runs=[None] * 3,  # type: ignore[list-item]
            logger=logging.getLogger("test"),
        )


class HarmonicEngine(OpenMMEngine):
    """Independent particles in 3D harmonic wells: no constraints, no centre-of-mass remover."""

    n_particles = 50

    def build_system(self) -> BuiltSystem:
        """Build the wells.

        :return: The built system."""
        topology = app.Topology()
        residue = topology.addResidue("AR", topology.addChain())
        system = openmm.System()
        well = openmm.CustomExternalForce("500*(x^2+y^2+z^2)")
        for i in range(self.n_particles):
            topology.addAtom(f"A{i}", app.Element.getBySymbol("Ar"), residue)
            system.addParticle(12.0)
            well.addParticle(i, [])
        system.addForce(well)
        return BuiltSystem(topology, system, np.zeros((self.n_particles, 3)) * unit.nanometer)


def test_samples_follow_the_canonical_ensemble_at_every_temperature(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Mean energy and swap acceptance match the exact harmonic values, end to end.

    A frame taken from the wrong replica, or a swap that does not change the temperature, shifts
    the mean energies; a wrong criterion shifts the acceptance.

    :param tmp_path: Pytest temporary directory.
    :param request: Pytest request, for the cleanup."""
    # one CPU thread is much faster than many for 50 particles.
    cpu = openmm.Platform.getPlatformByName("CPU")
    threads = cpu.getPropertyDefaultValue("Threads")
    cpu.setPropertyDefaultValue("Threads", "1")
    request.addfinalizer(lambda: cpu.setPropertyDefaultValue("Threads", threads))
    temperatures = [300.0, 330.0, 363.0, 400.0]
    config = AmberConfig(
        system="wells",
        output_subdir="wells/remd",
        input_pdb="a.pdb",
        num_steps=40_000,
        output_freq=100,
        report_interval=100,
        friction=10.0,
        temperature=300.0,
        replica_exchange=dict(
            temperatures=temperatures, exchange_interval=100, other_output="energies", seed=3
        ),
    )
    root = HarmonicEngine(config, tmp_path, logging.getLogger("test")).run()

    f = 3 * HarmonicEngine.n_particles
    kT = [remd._R * t for t in temperatures]
    for k, folder in enumerate([root, *sorted(root.glob("T0[1-9]_*K"))]):
        energies = [float(r.split(",")[1]) for r in _rows(folder / "energies.csv")[1:]]
        mean = np.mean(energies[len(energies) // 5 :])
        assert mean == pytest.approx(f / 2 * kT[k], rel=0.03), temperatures[k]

    # a harmonic system's potential energy is Gamma(f / 2, kT) distributed.
    rng = np.random.default_rng(0)
    draws = [rng.gamma(f / 2, kt, size=200_000) for kt in kT]
    rows = [row.split(",") for row in _rows(root / "exchanges.csv")[1:]]
    for k in range(len(temperatures) - 1):
        observed = [int(r[2 + len(temperatures) + k]) for r in rows if r[2 + len(temperatures) + k]]
        beta_i, beta_j = 1 / kT[k], 1 / kT[k + 1]
        exact = np.mean(np.minimum(1.0, np.exp((beta_i - beta_j) * (draws[k] - draws[k + 1]))))
        assert np.mean(observed) == pytest.approx(exact, abs=0.1), k


def _amber(**rex) -> AmberConfig:
    """Build an AMBER config with a replica-exchange block, left to pydantic to validate.

    :param rex: Fields of the block, or ``num_steps``/``output_freq``/``temperature`` overrides.
    :return: The config."""
    top = {k: rex.pop(k) for k in ("num_steps", "output_freq", "temperature") if k in rex}
    block = dict(exchange_interval=500) | rex
    return AmberConfig(
        system="t",
        output_subdir="t/run",
        input_pdb="a.pdb",
        num_steps=top.get("num_steps", 10_000),
        output_freq=top.get("output_freq", 1_000),
        temperature=top.get("temperature", 330.0),
        replica_exchange=block,
    )


def test_both_ladder_forms() -> None:
    """The ladder is given explicitly, or as a geometric one from the target temperature."""
    explicit = _amber(temperatures=[330, 400, 500])
    assert explicit.replica_exchange.temperatures == [330, 400, 500]  # type: ignore[union-attr]
    geometric = _amber(t_max=500, n_replicas=4)
    assert isinstance(geometric.replica_exchange, ReplicaExchangeConfig)
    assert remd.resolve_ladder(geometric)[0] == 330.0
    assert remd.resolve_ladder(geometric)[-1] == 500.0
    assert len(remd.resolve_ladder(geometric)) == 4
    assert remd.resolve_ladder(explicit) == [330, 400, 500]


@pytest.mark.parametrize(
    "fields",
    [
        dict(temperatures=[330, 400], t_max=500, n_replicas=3),
        dict(temperatures=[330, 400], n_replicas=3),
        dict(t_max=500),
        dict(),
        dict(temperatures=[330]),
        dict(t_max=500, n_replicas=1),
        dict(temperatures=[330, 330, 400]),
        dict(temperatures=[330, 500, 400]),
        dict(temperatures=[300, 400]),
        dict(t_max=330, n_replicas=3),
        dict(temperatures=[330, 400], exchange_interval=0),
        dict(temperatures=[330, 400], checkpoint_interval=750),
        dict(temperatures=[330, 400], exchange_interval=300),
        dict(temperatures=[330, 400], other_output_freq=700),
        dict(temperatures=[330, 400], num_steps=10_250),
        dict(temperatures=[330, 400], output_freq=750),
    ],
)
def test_invalid_replica_exchange_configs(fields: dict) -> None:
    """Every validation rule rejects its violation.

    :param fields: A violating replica-exchange block (parametrised)."""
    with pytest.raises(ValidationError):
        _amber(**fields)


def test_valid_intervals_are_accepted() -> None:
    """Intervals that divide one another pass, and `other_output_freq` is ignored when unused."""
    ok = _amber(
        temperatures=[330, 400],
        exchange_interval=500,
        other_output_freq=250,
        checkpoint_interval=1000,
        output_freq=250,
    )
    assert ok.replica_exchange.checkpoint_interval == 1000  # type: ignore[union-attr]
    _amber(temperatures=[330, 400], other_output="none", other_output_freq=700)


@pytest.mark.parametrize("engine", ["cgschnet", "gromacs_native"])
def test_non_openmm_engines_reject_replica_exchange(engine: str) -> None:
    """Engines that do not run through the OpenMM path say so.

    :param engine: Engine name (parametrised)."""
    payload = SAMPLE_CONFIGS[engine] | {
        "replica_exchange": {"temperatures": [300, 400], "exchange_interval": 500}
    }
    with pytest.raises(ValidationError, match="not supported by the"):
        TypeAdapter(SimulationConfig).validate_python(payload)


@pytest.mark.parametrize("engine", ["amber", "pure_liquid", "martini", "gromacs_input", "awsem"])
def test_openmm_engines_accept_replica_exchange(engine: str) -> None:
    """Every engine that runs through ``OpenMMEngine.run`` accepts the block.

    :param engine: Engine name (parametrised)."""
    payload = SAMPLE_CONFIGS[engine] | {
        "temperature": 300,
        "num_steps": 1000,
        "replica_exchange": {"temperatures": [300, 400], "exchange_interval": 500},
    }
    assert TypeAdapter(SimulationConfig).validate_python(payload).replica_exchange is not None
