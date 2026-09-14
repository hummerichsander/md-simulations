import logging
from pathlib import Path

import pytest

from md_simulations.config.base import (
    AmberConfig,
    GromacsInputConfig,
    MartiniConfig,
    PureLiquidConfig,
)
from md_simulations.engines import build_engine
from md_simulations.engines.amber import AmberEngine
from md_simulations.engines.gromacs_input import GromacsInputEngine
from md_simulations.engines.martini_openmm import MartiniEngine
from md_simulations.engines.pure_liquid import PureLiquidEngine

_LOG = logging.getLogger("test")
_ROOT = Path("/data")

_CASES = [
    (AmberConfig(system="t", output_subdir="t", input_pdb="a.pdb"), AmberEngine),
    (PureLiquidConfig(system="t", output_subdir="t", input_pdb="a.pdb"), PureLiquidEngine),
    (MartiniConfig(system="t", output_subdir="t", input_gro="a.gro", top="a.top"), MartiniEngine),
    (
        GromacsInputConfig(system="t", output_subdir="t", input_gro="a.gro", input_top="a.top"),
        GromacsInputEngine,
    ),
]


@pytest.mark.parametrize("config,cls", _CASES, ids=[c[0].engine for c in _CASES])
def test_build_engine_dispatch(config, cls: type) -> None:
    """The factory instantiates the engine class matching each config's engine.

    :param config: A config instance (parametrised).
    :param cls: Expected engine class (parametrised).
    :return: None."""
    assert isinstance(build_engine(config, _ROOT, _LOG), cls)


def test_resolve_constraints_maps_every_scheme() -> None:
    """Each constraints scheme maps to its OpenMM constant, and an unknown one raises.

    :return: None."""
    import openmm.app as app

    from md_simulations.engines.amber import resolve_constraints

    assert resolve_constraints("hbonds") is app.HBonds
    assert resolve_constraints("allbonds") is app.AllBonds
    assert resolve_constraints("none") is None
    with pytest.raises(ValueError):
        resolve_constraints("hangles")  # type: ignore[arg-type]


# ALA tripeptide, heavy atoms only; addHydrogens fills in the rest. Written out rather than
# shipped as a fixture file because the repo keeps no PDBs.
_ALA3_PDB = """ATOM      1  N   ALA A   1      -0.677   1.230  -0.491  1.00  0.00           N
ATOM      2  CA  ALA A   1      -0.001   0.000   0.000  1.00  0.00           C
ATOM      3  CB  ALA A   1      -0.523  -1.239  -0.503  1.00  0.00           C
ATOM      4  C   ALA A   1       1.499   0.000   0.000  1.00  0.00           C
ATOM      5  O   ALA A   1       2.100   1.055   0.000  1.00  0.00           O
ATOM      6  N   ALA A   2       2.130  -1.171   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.570  -1.322   0.000  1.00  0.00           C
ATOM      8  CB  ALA A   2       4.077  -2.128  -1.190  1.00  0.00           C
ATOM      9  C   ALA A   2       3.983  -2.007   1.295  1.00  0.00           C
ATOM     10  O   ALA A   2       3.184  -2.740   1.884  1.00  0.00           O
ATOM     11  N   ALA A   3       5.207  -1.782   1.752  1.00  0.00           N
ATOM     12  CA  ALA A   3       5.755  -2.365   2.965  1.00  0.00           C
ATOM     13  CB  ALA A   3       7.185  -1.899   3.212  1.00  0.00           C
ATOM     14  C   ALA A   3       5.694  -3.888   2.905  1.00  0.00           C
ATOM     15  O   ALA A   3       6.048  -4.491   1.891  1.00  0.00           O
ATOM     16  OXT ALA A   3       5.262  -4.492   3.899  1.00  0.00           O
END
"""


def _built_system(tmp_path: Path, constraints: str):
    """Build an implicit-solvent AMBER system for the test peptide.

    :param tmp_path: Directory to write the input PDB into, used as the data root.
    :param constraints: The ``constraints`` scheme to build with.
    :return: The engine's :class:`BuiltSystem`."""
    config = AmberConfig(
        system="ala3",
        output_subdir="ala3",
        input_pdb="ala3.pdb",
        forcefield=["amber14-all.xml", "implicit/gbn2.xml"],
        implicit_solvent=True,
        nonbonded_cutoff=2.0,
        constraints=constraints,  # type: ignore[arg-type]
    )
    return AmberEngine(config, tmp_path, _LOG).build_system()


def test_constraints_none_restores_the_hydrogen_bond_terms(tmp_path: Path) -> None:
    """``constraints: none`` drops the constraints and puts the X-H bond springs back.

    This is the whole point of the option: under ``hbonds`` OpenMM *deletes* the harmonic bond
    term of every constrained bond, so the energy barely depends on those bond lengths and a
    reweighting target built from it scores them not at all.

    :param tmp_path: Pytest temporary directory (used as the data root).
    :return: None."""
    import openmm

    # written outside the guard below, so a filesystem failure cannot pass for a missing XML
    (tmp_path / "ala3.pdb").write_text(_ALA3_PDB)

    try:
        constrained = _built_system(tmp_path, "hbonds")
        flexible = _built_system(tmp_path, "none")
    except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - needs the shipped XMLs
        pytest.skip(f"amber14 force field unavailable: {exc}")

    def bond_count(system) -> int:
        force = next(
            f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce)
        )
        return force.getNumBonds()

    n_constraints = constrained.system.getNumConstraints()
    assert n_constraints > 0
    assert flexible.system.getNumConstraints() == 0

    # every bond the constrained build turned into a constraint comes back as a spring
    assert bond_count(flexible.system) == bond_count(constrained.system) + n_constraints


def test_gromacs_input_velocity_flag_from_config() -> None:
    """The gromacs_input engine mirrors initialize_velocities from its config.

    :return: None."""
    on = GromacsInputConfig(
        system="t", output_subdir="t", input_gro="a.gro", input_top="a.top",
        initialize_velocities=True,
    )
    off = GromacsInputConfig(
        system="t", output_subdir="t", input_gro="a.gro", input_top="a.top",
        initialize_velocities=False,
    )
    assert build_engine(on, _ROOT, _LOG).initialize_velocities is True
    assert build_engine(off, _ROOT, _LOG).initialize_velocities is False
