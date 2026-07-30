import logging
from pathlib import Path

import openmm
import openmm.app as app
import openmm.unit as unit
from openmm import Vec3

from md_simulations.config.base import AmberConfig
from md_simulations.core import (
    add_standard_reporters,
    create_integrator,
    run_simulation_loop,
    save_final_structure,
    save_initial_structure,
    write_simulation_summary,
)


def _tiny_simulation() -> app.Simulation:
    """Build a self-contained 2-particle harmonic system (no data files needed).

    :return: A ready-to-step OpenMM simulation."""
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("AR2", chain)
    argon = app.Element.getBySymbol("Ar")
    topology.addAtom("A1", argon, residue)
    topology.addAtom("A2", argon, residue)

    system = openmm.System()
    system.addParticle(39.948)
    system.addParticle(39.948)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.2, 1000.0)
    system.addForce(bond)

    integrator = create_integrator(300.0, 1.0, 0.002)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(
        [Vec3(0, 0, 0), Vec3(0.2, 0, 0)] * unit.nanometer
    )
    return simulation


def test_core_reporters_and_loop(tmp_path: Path) -> None:
    """Reporters + run loop + final-structure write produce the standard outputs.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    simulation = _tiny_simulation()
    simulation.minimizeEnergy()

    add_standard_reporters(
        simulation, tmp_path, report_interval=2, output_freq=2, num_steps=10
    )
    run_simulation_loop(simulation, num_steps=10, output_freq=2)
    save_final_structure(simulation, tmp_path, logging.getLogger("test"))

    for name in ("trajectory.dcd", "energies.csv", "forces.txt", "final_structure.pdb"):
        f = tmp_path / name
        assert f.exists(), f"missing output {name}"
        assert f.stat().st_size > 0


def test_initial_structure_and_summary(tmp_path: Path) -> None:
    """Initial frame and parameter summary are written with expected content.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    simulation = _tiny_simulation()
    config = AmberConfig(system="ala2", output_subdir="x", input_pdb="a.pdb")

    save_initial_structure(simulation, tmp_path)
    write_simulation_summary(tmp_path, config, simulation)

    initial = tmp_path / "initial_structure.pdb"
    summary = tmp_path / "simulation_summary.txt"
    assert initial.exists() and initial.stat().st_size > 0
    assert summary.exists()

    text = summary.read_text()
    assert "Simulation summary" in text
    assert "Ensemble" in text
    assert "Timestep" in text
    assert "Simulated time" in text
    assert "OpenMM" in text


def test_implicit_summary_omits_water_model(tmp_path: Path) -> None:
    """An implicit-solvent config reports GB, not a (meaningless) water model.

    :param tmp_path: Pytest temporary directory.
    :return: None."""
    simulation = _tiny_simulation()
    config = AmberConfig(
        system="cln",
        output_subdir="x",
        input_pdb="a.pdb",
        implicit_solvent=True,
        forcefield=["amber14-all.xml", "implicit/gbn2.xml"],
    )
    assert config.implicit_solvent

    write_simulation_summary(tmp_path, config, simulation)
    text = (tmp_path / "simulation_summary.txt").read_text()
    assert "Water model" not in text
    assert "Solvent padding" not in text
    assert "Implicit solvent" in text
