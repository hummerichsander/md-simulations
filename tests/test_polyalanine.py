from pathlib import Path

import numpy as np
import pytest

from md_simulations.prep.polyalanine import build_polyalanine, format_pdb

PDB_DIR = Path(__file__).resolve().parents[1] / "data" / "pdbs"


def _coords(text: str) -> np.ndarray:
    """Extract ATOM coordinates from PDB text.

    :param text: PDB file contents.
    :return: (n_atoms, 3) array of coordinates in angstrom."""
    return np.array(
        [
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for line in text.splitlines()
            if line.startswith("ATOM")
        ]
    )


def _labels(text: str) -> list[str]:
    """Extract per-atom name/resname/resseq labels from PDB text.

    :param text: PDB file contents.
    :return: List of atom label strings, one per ATOM record."""
    return [line[12:26] for line in text.splitlines() if line.startswith("ATOM")]


@pytest.mark.parametrize("n", range(3, 13))
def test_reproduces_reference_pdbs(n: int) -> None:
    """The builder reproduces the checked-in ala3-ala12 structures.

    :param n: Number of alanine residues.
    :return: None."""
    reference = PDB_DIR / f"ala{n}.pdb"
    if not reference.exists():
        pytest.skip(f"{reference} not present (data root is git-ignored)")

    ref_text = reference.read_text()
    new_text = format_pdb(build_polyalanine(n), n)

    assert _labels(new_text) == _labels(ref_text)
    # the references were written by an equivalent but differently-ordered float
    # path, so they agree only to the 0.001 A PDB print resolution
    assert np.abs(_coords(new_text) - _coords(ref_text)).max() < 0.0015


@pytest.mark.parametrize("n", [1, 2, 16, 24, 32])
def test_atom_count_and_residue_numbering(n: int) -> None:
    """Every chain length gets 3 + 5n + 2 heavy atoms and contiguous residue IDs.

    :param n: Number of alanine residues.
    :return: None."""
    atoms = build_polyalanine(n)
    assert len(atoms) == 3 + 5 * n + 2

    resnames = {resseq: resname for _, resname, resseq, _ in atoms}
    assert resnames[1] == "ACE"
    assert resnames[n + 2] == "NME"
    assert all(resnames[i] == "ALA" for i in range(2, n + 2))
    assert sorted(resnames) == list(range(1, n + 3))


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute the p0-p1-p2-p3 dihedral angle.

    :param p0: First atom position.
    :param p1: Second atom position.
    :param p2: Third atom position.
    :param p3: Fourth atom position.
    :return: The dihedral in degrees, in (-180, 180]."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def test_backbone_dihedrals_match_request() -> None:
    """Every residue carries the requested phi/psi/omega, caps included.

    :return: None."""
    n = 16
    xyz = {(resseq, name): pos for name, _, resseq, pos in build_polyalanine(n, -60.0, -45.0)}

    for r in range(2, n + 2):
        assert _dihedral(xyz[r - 1, "C"], xyz[r, "N"], xyz[r, "CA"], xyz[r, "C"]) == pytest.approx(
            -60.0, abs=1e-6
        )
        assert _dihedral(xyz[r, "N"], xyz[r, "CA"], xyz[r, "C"], xyz[r + 1, "N"]) == pytest.approx(
            -45.0, abs=1e-6
        )
        nxt = xyz.get((r + 1, "CA"), xyz.get((r + 1, "CH3")))
        assert abs(_dihedral(xyz[r, "CA"], xyz[r, "C"], xyz[r + 1, "N"], nxt)) == pytest.approx(
            180.0, abs=1e-6
        )


def test_alpha_helix_hydrogen_bond_pattern() -> None:
    """The default conformation shows i -> i+4 alpha-helical contacts.

    :return: None."""
    n = 24
    xyz = {(resseq, name): pos for name, _, resseq, pos in build_polyalanine(n)}

    hbonds = [np.linalg.norm(xyz[r + 4, "N"] - xyz[r, "O"]) for r in range(2, n - 2)]
    ca_ca = [np.linalg.norm(xyz[r + 1, "CA"] - xyz[r, "CA"]) for r in range(2, n + 1)]
    ca_i4 = [np.linalg.norm(xyz[r + 4, "CA"] - xyz[r, "CA"]) for r in range(2, n - 2)]

    assert np.allclose(hbonds, 3.03, atol=0.05)
    assert np.allclose(ca_ca, 3.80, atol=0.02)
    assert np.allclose(ca_i4, 6.31, atol=0.05)


def test_rejects_empty_chain() -> None:
    """A chain with no alanine residues is an error.

    :return: None."""
    with pytest.raises(ValueError):
        build_polyalanine(0)
