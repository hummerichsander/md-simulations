import logging
from pathlib import Path

import pytest

from md_simulations.config.base import AmberConfig, FrozenAtomsConfig
from md_simulations.core.freezing import freeze_atoms, resolve_frozen_atoms
from md_simulations.core.tracking import degrees_of_freedom
from md_simulations.engines.amber import AmberEngine
from test_engines import _ALA3_PDB

_LOG = logging.getLogger("test")
# The slow subspace of a 5-bead (N CA CB C O) coarse-graining, i.e. one bead per retained atom.
_SLOW_NAMES = ["N", "CA", "CB", "C", "O"]


def _build(tmp_path: Path, constraints: str):
    """Build the implicit-solvent AMBER system of the test tripeptide.

    :param tmp_path: Directory used as the data root; the input PDB is written into it.
    :param constraints: The ``constraints`` scheme to build with.
    :return: The engine's :class:`BuiltSystem`."""
    (tmp_path / "ala3.pdb").write_text(_ALA3_PDB)
    config = AmberConfig(
        system="ala3",
        output_subdir="ala3",
        input_pdb="ala3.pdb",
        forcefield=["amber14-all.xml", "implicit/gbn2.xml"],
        implicit_solvent=True,
        nonbonded_cutoff=2.0,
        constraints=constraints,  # type: ignore[arg-type]
    )
    try:
        return AmberEngine(config, tmp_path, _LOG).build_system()
    except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - needs the shipped XMLs
        pytest.skip(f"amber14 force field unavailable: {exc}")


def test_exactly_one_selection_form() -> None:
    """A frozen-atom selection is by name or by index, never both or neither.

    :return: None."""
    with pytest.raises(ValueError):
        FrozenAtomsConfig()
    with pytest.raises(ValueError):
        FrozenAtomsConfig(names=["CA"], indices=[0])
    with pytest.raises(ValueError):
        FrozenAtomsConfig(indices=[1, 1])
    assert FrozenAtomsConfig(names=["CA"]).names == ["CA"]


def test_names_resolve_against_the_built_topology(tmp_path: Path) -> None:
    """A name selection picks exactly the atoms carrying those names, hydrogens excluded.

    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    built = _build(tmp_path, "none")
    indices = resolve_frozen_atoms(built.topology, FrozenAtomsConfig(names=_SLOW_NAMES))
    atoms = list(built.topology.atoms())

    assert indices == sorted(indices)
    assert {atoms[i].name for i in indices} <= set(_SLOW_NAMES)
    assert not any(atoms[i].element.symbol == "H" for i in indices)
    # ALA3 with a terminal OXT: 5 named atoms per residue, 15 in total.
    assert len(indices) == 15


def test_expect_count_guards_the_wrong_topology(tmp_path: Path) -> None:
    """``expect_count`` catches a selection resolved against a topology it was not written for.

    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    built = _build(tmp_path, "none")
    spec = FrozenAtomsConfig(names=_SLOW_NAMES, expect_count=47)
    with pytest.raises(ValueError, match="expect_count"):
        resolve_frozen_atoms(built.topology, spec)


def test_unmatched_names_and_out_of_range_indices_raise(tmp_path: Path) -> None:
    """A selection that matches nothing, or points past the topology, is an error.

    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    built = _build(tmp_path, "none")
    with pytest.raises(ValueError, match="matched no atom"):
        resolve_frozen_atoms(built.topology, FrozenAtomsConfig(names=["ZZ"]))
    with pytest.raises(ValueError, match="out of range"):
        resolve_frozen_atoms(built.topology, FrozenAtomsConfig(indices=[10**6]))


def test_freezing_rejects_constraints_touching_a_frozen_atom(tmp_path: Path) -> None:
    """Freezing a build whose constraints reach a frozen atom fails with the fix in the message.

    OpenMM refuses such a constraint at context creation; catching it here keeps the failure off
    the GPU and names the remedy.

    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    built = _build(tmp_path, "hbonds")
    assert built.system.getNumConstraints() > 0
    with pytest.raises(ValueError, match="constraints: none"):
        freeze_atoms(built.system, built.topology, FrozenAtomsConfig(names=_SLOW_NAMES), _LOG)


def test_freezing_zeroes_masses_and_the_degree_of_freedom_count(tmp_path: Path) -> None:
    """Frozen atoms lose their mass, the COM remover goes, and the dof count is three per mover.

    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    import openmm
    import openmm.unit as unit

    built = _build(tmp_path, "none")
    n_atoms = built.system.getNumParticles()
    indices = freeze_atoms(built.system, built.topology, FrozenAtomsConfig(names=_SLOW_NAMES), _LOG)

    masses = [built.system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_atoms)]
    assert all(masses[i] == 0.0 for i in indices)
    assert all(masses[i] > 0.0 for i in set(range(n_atoms)) - set(indices))
    assert not any(isinstance(f, openmm.CMMotionRemover) for f in built.system.getForces())
    assert degrees_of_freedom(built.system) == 3 * (n_atoms - len(indices))


def test_frozen_atoms_do_not_move(tmp_path: Path) -> None:
    """A few steps of dynamics leave the frozen atoms exactly where they started.

    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    import numpy as np
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    built = _build(tmp_path, "none")
    indices = freeze_atoms(built.system, built.topology, FrozenAtomsConfig(names=_SLOW_NAMES), _LOG)

    integrator = openmm.LangevinMiddleIntegrator(
        330 * unit.kelvin, 1.0 / unit.picosecond, 0.0005 * unit.picoseconds
    )
    simulation = app.Simulation(
        built.topology, built.system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    simulation.context.setPositions(built.positions)
    simulation.context.setVelocitiesToTemperature(330 * unit.kelvin)
    before = np.array(built.positions.value_in_unit(unit.nanometer))
    simulation.step(10)
    after = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    after = after.value_in_unit(unit.nanometer)

    assert np.array_equal(after[indices], before[indices])
    mobile = sorted(set(range(len(before))) - set(indices))
    assert not np.array_equal(after[mobile], before[mobile])


def test_hydrogen_resampling_never_moves_a_frozen_atom(tmp_path: Path) -> None:
    """The X-H resampler displaces the lighter atom, so it never touches a frozen slow atom.

    The fiber trajectories are post-processed with ``md-sim-resample-hbonds`` to put them in the
    same convention as the reference ensemble; that step must leave the conditioning coordinates
    untouched.

    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    from md_simulations.sampling.resample_h_bonds import constrained_bond_parameters

    built = _build(tmp_path, "none")
    slow = set(resolve_frozen_atoms(built.topology, FrozenAtomsConfig(names=_SLOW_NAMES)))
    config = AmberConfig(
        system="ala3",
        output_subdir="ala3",
        input_pdb="ala3.pdb",
        forcefield=["amber14-all.xml", "implicit/gbn2.xml"],
        implicit_solvent=True,
        nonbonded_cutoff=2.0,
        constraints="none",
    )
    movers, anchors, _, _ = constrained_bond_parameters(config, tmp_path, logger=_LOG)

    assert len(movers) > 0
    assert not set(movers) & slow
    assert set(anchors) & slow  # the anchors are precisely the heavy atoms, some of them frozen
