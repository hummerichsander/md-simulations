"""Smoke tests for the config -> calculator/potential build path.

Hermetic: engines are stubbed to a self-contained 2-particle system (mirroring
``test_core_smoke``), so no force-field files or data root contents are needed.
The directory-wide guard in ``conftest`` skips this without the
``torch-potentials`` extra."""

import logging
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pytest
from openmm import Vec3

from md_simulations.config.base import AmberConfig, GromacsNativeConfig
from md_simulations.engines.base import BuiltSystem, Engine, OpenMMEngine
from md_simulations.torch_potentials import build

_POS_NM = np.array([[0.0, 0.0, 0.0], [0.17, 0.01, 0.0]])
_LOG = logging.getLogger("test")


def _diatomic_system() -> tuple[app.Topology, openmm.System]:
    """A two-oxygen stiff-harmonic system (no data files needed).

    :return: ``(topology, system)`` for the diatomic."""
    topology = app.Topology()
    residue = topology.addResidue("OO2", topology.addChain())
    oxygen = app.Element.getBySymbol("O")
    topology.addAtom("O1", oxygen, residue)
    topology.addAtom("O2", oxygen, residue)

    system = openmm.System()
    system.addParticle(16.0)
    system.addParticle(16.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.15, 4.0e5)  # r0 = 0.15 nm
    system.addForce(bond)
    return topology, system


class _StubOpenMMEngine(OpenMMEngine):
    """OpenMM engine yielding the diatomic ``BuiltSystem``."""

    def build_system(self) -> BuiltSystem:
        topology, system = _diatomic_system()
        positions = [Vec3(*row) for row in _POS_NM] * unit.nanometer
        return BuiltSystem(topology, system, positions)


class _StubEngine(Engine):
    """A non-OpenMM engine (has no calculator bridge)."""

    def run(self) -> Path:
        raise NotImplementedError


def test_build_calculator_openmm(monkeypatch, tmp_path: Path) -> None:
    """The OpenMM branch yields a calculator matching a plain OpenMM context.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    import mdtraj as md
    import torch
    from md_simulations.torch_potentials import Potential

    monkeypatch.setattr(
        build, "build_engine", lambda config, data_root, logger: _StubOpenMMEngine(config, data_root, logger)
    )

    config = AmberConfig(system="t", output_subdir="t", input_pdb="a.pdb")
    calc = build.build_calculator(config, data_root=tmp_path)

    topology, system = _diatomic_system()
    pot = Potential(md.Topology.from_openmm(topology), calc)
    x = torch.tensor(_POS_NM, dtype=torch.float64, requires_grad=True)
    energy = pot(x)
    energy.backward()

    ref = openmm.Context(system, openmm.VerletIntegrator(0.001))
    ref.setPositions(_POS_NM * unit.nanometer)
    state = ref.getState(getEnergy=True, getForces=True)
    e_ref = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f_ref = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    assert np.isfinite(energy.item())
    assert np.isclose(energy.item(), e_ref, atol=1e-4)
    np.testing.assert_allclose(-x.grad.numpy(), f_ref, atol=1e-4)


def test_build_calculator_unsupported_engine(monkeypatch, tmp_path: Path) -> None:
    """A non-OpenMM engine raises a clear ``NotImplementedError``.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    monkeypatch.setattr(
        build, "build_engine", lambda config, data_root, logger: _StubEngine(config, data_root, logger)
    )
    config = GromacsNativeConfig(
        system="t", output_subdir="t", input_gro="a.gro", top="a.top", mdp="m.mdp"
    )
    with pytest.raises(NotImplementedError, match="gromacs_native"):
        build.build_calculator(config, data_root=tmp_path)


def test_build_potential_matches_manual_composition(monkeypatch, tmp_path: Path) -> None:
    """``build_potential`` equals ``Potential(top, build_calculator(...))``.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    import mdtraj as md
    import torch

    from md_simulations.torch_potentials import Potential

    monkeypatch.setattr(
        build, "build_engine", lambda config, data_root, logger: _StubOpenMMEngine(config, data_root, logger)
    )

    topology, _ = _diatomic_system()
    top = md.Topology.from_openmm(topology)
    config = AmberConfig(system="t", output_subdir="t", input_pdb="a.pdb")

    pot = build.build_potential(top, config, data_root=tmp_path)
    manual = Potential(top, build.build_calculator(config, data_root=tmp_path))

    assert isinstance(pot, Potential)
    assert pot.n_atoms == top.n_atoms

    x = torch.tensor(_POS_NM, dtype=torch.float64, requires_grad=True)
    energy = pot(x)
    energy.backward()
    assert np.isclose(energy.item(), manual(x.detach()).item())
    assert x.grad is not None


def test_build_potential_from_file(monkeypatch, tmp_path: Path, write_yaml) -> None:
    """``build_potential_from_file`` loads the YAML and binds the topology.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest temporary directory (used as the data root).
    :param write_yaml: Fixture writing a config dict to a YAML file.
    :return: None."""
    import mdtraj as md

    from md_simulations.torch_potentials import Potential

    monkeypatch.setattr(
        build, "build_engine", lambda config, data_root, logger: _StubOpenMMEngine(config, data_root, logger)
    )

    path = write_yaml(
        {
            "engine": "amber",
            "system": "t",
            "output_subdir": "t/run",
            "input_pdb": "a.pdb",
        }
    )
    topology, _ = _diatomic_system()
    top = md.Topology.from_openmm(topology)

    pot = build.build_potential_from_file(top, path, data_root=tmp_path)
    assert isinstance(pot, Potential)
    assert pot.n_atoms == top.n_atoms


def test_lazy_names_are_public_and_unknown_ones_raise() -> None:
    """The lazy ``__getattr__`` resolves ``__all__`` and rejects anything else.

    :return: None."""
    import md_simulations.torch_potentials as tp

    for name in tp.__all__:
        assert getattr(tp, name) is not None
    assert set(tp.__all__) == set(dir(tp))

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        tp.nope
