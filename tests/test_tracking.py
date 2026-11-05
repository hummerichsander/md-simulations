import logging
from pathlib import Path

import openmm
import openmm.app as app
import openmm.unit as unit
import pytest
from beamtime.format.run import read_artifacts, read_config, read_effective_metrics, read_run_info
from beamtime.format.tree import iter_runs
from openmm import Vec3

from md_simulations.config.base import AmberConfig, CGSchNetConfig
from md_simulations.core.tracking import degrees_of_freedom, start_replica_runs
from md_simulations.engines.base import BuiltSystem, OpenMMEngine


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


class BrokenEngine(OpenMMEngine):
    """OpenMM engine whose system cannot be built."""

    def build_system(self) -> BuiltSystem:
        """Fail to build.

        :return: Never returns."""
        raise RuntimeError("no system")


def _config(**kwargs) -> AmberConfig:
    """Build a tiny AMBER config.

    :param kwargs: Fields to override.
    :return: The config."""
    fields = dict(
        system="argon",
        output_subdir="argon/pair_300K",
        input_pdb="a.pdb",
        num_steps=10,
        output_freq=2,
        report_interval=2,
    )
    return AmberConfig(**(fields | kwargs))


def _runs(logdir: Path) -> list[Path]:
    """List all run directories below the log dir.

    :param logdir: The log dir.
    :return: The run directories."""
    return list(iter_runs(logdir))


def test_beamtime_location_defaults() -> None:
    """A run goes into the system's group, named after its output folder."""
    config = _config()
    assert config.beamtime_group == "argon"
    assert config.beamtime_name == "pair_300K"

    replica = _config(beamtime={"group": "argon/pair_300K", "name": "r03"})
    assert replica.beamtime_group == "argon/pair_300K"
    assert replica.beamtime_name == "r03"


def test_openmm_engine_logs_one_run(tmp_path: Path, beamtime_logdir: Path) -> None:
    """An OpenMM simulation is one finished run with energies, its config and its outputs.

    :param tmp_path: Pytest temporary directory.
    :param beamtime_logdir: The temporary log dir."""
    ArgonEngine(_config(), tmp_path / "data", logging.getLogger("test")).run()

    (run_dir,) = _runs(beamtime_logdir)
    assert run_dir.parent == beamtime_logdir / "argon"
    info = read_run_info(run_dir)
    assert (info.name, info.status, info.tags) == ("pair_300K", "finished", ["amber"])
    assert read_config(run_dir)["output_dir"] == str((tmp_path / "data/argon/pair_300K"))

    rows = read_effective_metrics(run_dir)
    assert [r["_step"] for r in rows if "energy/potential" in r] == [2, 4, 6, 8, 10]
    assert all(r["temperature"] > 0 for r in rows if "temperature" in r)
    names = {a.name for a in read_artifacts(run_dir)}
    assert {"trajectory.dcd", "energies.csv", "final_structure.pdb"} <= names


def test_failed_simulation_marks_run_failed(tmp_path: Path, beamtime_logdir: Path) -> None:
    """An exception during the simulation ends the run as failed and propagates.

    :param tmp_path: Pytest temporary directory.
    :param beamtime_logdir: The temporary log dir."""
    with pytest.raises(RuntimeError, match="no system"):
        BrokenEngine(_config(), tmp_path, logging.getLogger("test")).run()

    (run_dir,) = _runs(beamtime_logdir)
    assert read_run_info(run_dir).status == "failed"


def test_replica_runs_form_a_group(tmp_path: Path, beamtime_logdir: Path) -> None:
    """Batched replicas are runs ``r00, r01, ...`` in a group named after the output folder.

    :param tmp_path: Pytest temporary directory.
    :param beamtime_logdir: The temporary log dir."""
    config = CGSchNetConfig(
        system="ala5",
        output_subdir="alaX/ala5_CGSchNet_330K",
        input_pdb="cg.pdb",
        model_file="m.pt",
        configurations_file="c.pt",
        n_replicas=3,
    )
    runs = start_replica_runs(config, tmp_path, ["r00", "r01", "r02"])
    for i, run in enumerate(runs):
        with run:
            run.log({"energy/potential": float(i)}, step=1)

    run_dirs = _runs(beamtime_logdir)
    assert {d.parent for d in run_dirs} == {beamtime_logdir / "ala5/ala5_CGSchNet_330K"}
    assert sorted(read_run_info(d).name for d in run_dirs) == ["r00", "r01", "r02"]
    for d in run_dirs:
        info = read_run_info(d)
        assert info.status == "finished"
        assert read_config(d)["replica_index"] == int(info.name[1:])
        assert read_effective_metrics(d)[0]["energy/potential"] == float(info.name[1:])


def test_degrees_of_freedom() -> None:
    """Constraints and a centre-of-mass remover each remove degrees of freedom."""
    system = _argon_pair().system
    assert degrees_of_freedom(system) == 6
    system.addConstraint(0, 1, 0.2)
    system.addForce(openmm.CMMotionRemover())
    assert degrees_of_freedom(system) == 2


def test_cgschnet_writer_logs_each_replica(tmp_path: Path, beamtime_logdir: Path) -> None:
    """The CGSchNet trajectory writer logs every saved frame to its replica's run.

    :param tmp_path: Pytest temporary directory.
    :param beamtime_logdir: The temporary log dir."""
    import numpy as np

    from md_simulations.engines.cgschnet import _IncrementalTrajectoryWriter

    config = CGSchNetConfig(
        system="ala5",
        output_subdir="alaX/ala5_CGSchNet_330K",
        input_pdb="cg.pdb",
        model_file="m.pt",
        configurations_file="c.pt",
        n_replicas=2,
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    np.save(raw / "sim_coords_0000.npy", np.zeros((2, 3, 4, 3)))
    np.save(raw / "sim_potential_0000.npy", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.save(raw / "sim_forces_0000.npy", np.ones((2, 3, 4, 3)))

    runs = start_replica_runs(config, tmp_path, ["r00", "r01"])
    writer = _IncrementalTrajectoryWriter(tmp_path, raw, "sim", 10, 1.0, 3, 2, runs)
    writer.close()
    for run in runs:
        run.finish()

    for d in _runs(beamtime_logdir):
        rows = read_effective_metrics(d)
        offset = 0.0 if read_run_info(d).name == "r00" else 3.0
        assert [r["_step"] for r in rows if "energy/potential" in r] == [10, 20, 30]
        assert [r["energy/potential"] for r in rows if "energy/potential" in r] == [
            offset + 1.0,
            offset + 2.0,
            offset + 3.0,
        ]
        # unit forces per Å are 10 kJ/mol/nm.
        assert all(r["force/rms"] == pytest.approx(10.0) for r in rows if "force/rms" in r)
