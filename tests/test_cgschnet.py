import logging
from pathlib import Path

import numpy as np
import pytest

from md_simulations.config.base import CGSchNetConfig
from md_simulations.engines import build_engine
from md_simulations.prep.cgschnet_embeddings import extract_atom_types, residue_type_table
from md_simulations.prep.cgschnet_map import build_cg_structure, load_mapping


def _config() -> CGSchNetConfig:
    """Build a CGSchNet config for tests (no external files required).

    :return: A minimal CGSchNetConfig."""
    return CGSchNetConfig(
        system="cln",
        output_subdir="cln/CGSchNet",
        input_pdb="cg.pdb",
        model_file="model.pt",
        configurations_file="configs.pt",
    )

_TINY_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.332   1.548   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.999   2.842   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.510   2.720   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.055   1.629   0.000  1.00  0.00           O
TER
END
"""


def test_cgschnet_config() -> None:
    """The CGSchNet config skips minimization and defaults to one replica.

    :return: None."""
    config = _config()
    assert config.minimize_energy is False
    assert config.model_energy_unit == "kcal/mol"
    assert config.n_replicas == 1


def test_cg_slicing_map(tmp_path: Path, cg_mapping_file: Path) -> None:
    """The slicing mapper keeps one Cα bead per residue with its embedding.

    :param tmp_path: Pytest temporary directory.
    :param cg_mapping_file: Fixture writing a small mapping JSON.
    :return: None."""
    pdb = tmp_path / "tiny.pdb"
    pdb.write_text(_TINY_PDB)

    mapping = load_mapping(cg_mapping_file)
    cg, types = build_cg_structure(pdb, mapping)

    assert cg.n_atoms == 2  # one CA per residue
    assert types.tolist() == [mapping[("ALA", "CA")], mapping[("GLY", "CA")]]
    assert types.dtype == np.int64


class _FakeAtomicData:
    """Stand-in for an mlcg AtomicData exposing ``atom_types``."""

    def __init__(self, atom_types) -> None:
        self.atom_types = atom_types


@pytest.mark.parametrize(
    "obj",
    [
        [1, 2, 3],
        np.array([1, 2, 3]),
        np.array([[1], [2], [3]]),  # squeezes to 1-D
        {"atom_types": [1, 2, 3]},
        {"embeddings": np.array([1, 2, 3])},
        [_FakeAtomicData([1, 2, 3])],  # list of AtomicData → first element
        _FakeAtomicData(np.array([1, 2, 3])),
    ],
)
def test_extract_atom_types(obj) -> None:
    """atom types are recovered from the shapes a released config may take.

    :param obj: A configuration-like object (parametrised).
    :return: None."""
    out = extract_atom_types(obj)
    assert out.tolist() == [1, 2, 3]
    assert out.dtype == np.int64


def test_extract_atom_types_errors() -> None:
    """Unrecognised dict keys and non-1-D data raise clear errors.

    :return: None."""
    with pytest.raises(KeyError):
        extract_atom_types({"coords": [1, 2, 3]})
    with pytest.raises(ValueError):
        extract_atom_types(np.zeros((3, 3)))


def test_residue_type_table() -> None:
    """The residue→type table is built and per-residue conflicts are reported.

    :return: None."""
    table, conflicts = residue_type_table(
        ["ALA", "GLY", "ALA", "GLY"], np.array([1, 8, 1, 9])
    )
    assert table == {"ALA": 1, "GLY": 8}  # first-seen wins
    assert conflicts == {"GLY": [8, 9]}  # terminal/special variant flagged

    with pytest.raises(ValueError):
        residue_type_table(["ALA"], np.array([1, 2]))


def test_cgschnet_dispatch() -> None:
    """Selecting the cgschnet engine dispatches to CGSchNetEngine.

    torch/mlcg are imported lazily inside ``run()``, so dispatch itself needs
    neither — only the config and engine base class.

    :return: None."""
    from md_simulations.engines.cgschnet import CGSchNetEngine

    engine = build_engine(_config(), Path("/data"), logging.getLogger("test"))
    assert isinstance(engine, CGSchNetEngine)
