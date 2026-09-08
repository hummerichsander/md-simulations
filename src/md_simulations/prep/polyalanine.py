#!/usr/bin/env python3
"""Build capped ideal-alpha-helix polyalanine PDBs (ACE-(ALA)n-NME, heavy atoms only)."""

import argparse
from pathlib import Path

import numpy as np

N_CA_LENGTH = 1.46
CA_C_LENGTH = 1.52
C_O_LENGTH = 1.23
CA_CB_LENGTH = 1.52
PEPTIDE_BOND = 1.33

N_CA_C_ANGLE = 111.068
CA_C_O_ANGLE = 120.5
CA_C_N_ANGLE = 116.642
C_N_CA_ANGLE = 121.382
C_CA_CB_ANGLE = 109.5
N_C_CA_CB_DIANGLE = 122.686

ACE_O_C_N_ANGLE = 122.885
ACE_CH3_C_N_ANGLE = 116.65

PHI = -57.8
PSI = -47.0
OMEGA = 180.0

ATOM_FMT = (
    "ATOM  {serial:5d}  {name:<3s} {resname:>3s} {chain}{resseq:4d}"
    "    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}  "
)


def place_atom(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond: float,
    angle: float,
    torsion: float,
) -> np.ndarray:
    """Place a fourth atom from internal coordinates relative to three known atoms.

    :param a: Position of the first reference atom (defines the torsion).
    :param b: Position of the second reference atom (defines the angle).
    :param c: Position of the third reference atom (the new atom bonds to it).
    :param bond: c-d bond length in angstrom.
    :param angle: b-c-d angle in degrees.
    :param torsion: a-b-c-d dihedral in degrees.
    :return: Position of the new atom d."""
    angle_rad, torsion_rad = np.radians(angle), np.radians(torsion)

    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    M = np.stack([bc, np.cross(n, bc), n])

    d = np.array(
        [
            -bond * np.cos(angle_rad),
            bond * np.sin(angle_rad) * np.cos(torsion_rad),
            bond * np.sin(angle_rad) * np.sin(torsion_rad),
        ]
    )
    return c + d @ M


def build_polyalanine(
    n_residues: int,
    phi: float = PHI,
    psi: float = PSI,
    omega: float = OMEGA,
) -> list[tuple[str, str, int, np.ndarray]]:
    """Build ACE-(ALA)n-NME with a uniform backbone conformation.

    The first alanine is placed in the canonical PeptideBuilder frame (CA at the
    origin, C on +x, N in the xy-plane) so output is identical for every chain
    length. Caps are grown from that residue's backbone with idealized amide
    geometry; only heavy atoms are emitted.

    :param n_residues: Number of alanine residues between the caps.
    :param phi: Backbone phi dihedral in degrees.
    :param psi: Backbone psi dihedral in degrees.
    :param omega: Backbone omega dihedral in degrees.
    :return: List of (atom_name, residue_name, residue_seq, position) in output order."""
    if n_residues < 1:
        raise ValueError(f"n_residues must be >= 1, got {n_residues}")

    tau = np.radians(N_CA_C_ANGLE)
    backbone: list[dict[str, np.ndarray]] = [
        {
            "CA": np.zeros(3),
            "C": np.array([CA_C_LENGTH, 0.0, 0.0]),
            "N": np.array([N_CA_LENGTH * np.cos(tau), N_CA_LENGTH * np.sin(tau), 0.0]),
        }
    ]

    for _ in range(n_residues - 1):
        prev = backbone[-1]
        n = place_atom(prev["N"], prev["CA"], prev["C"], PEPTIDE_BOND, CA_C_N_ANGLE, psi)
        ca = place_atom(prev["CA"], prev["C"], n, N_CA_LENGTH, C_N_CA_ANGLE, omega)
        c = place_atom(prev["C"], n, ca, CA_C_LENGTH, N_CA_C_ANGLE, phi)
        backbone.append({"N": n, "CA": ca, "C": c})

    for res in backbone:
        res["O"] = place_atom(res["N"], res["CA"], res["C"], C_O_LENGTH, CA_C_O_ANGLE, psi + 180.0)
        res["CB"] = place_atom(
            res["N"], res["C"], res["CA"], CA_CB_LENGTH, C_CA_CB_ANGLE, N_C_CA_CB_DIANGLE
        )

    first, last = backbone[0], backbone[-1]
    ace_c = place_atom(first["C"], first["CA"], first["N"], PEPTIDE_BOND, C_N_CA_ANGLE, phi)
    ace_o = place_atom(first["CA"], first["N"], ace_c, C_O_LENGTH, ACE_O_C_N_ANGLE, 0.0)
    ace_ch3 = place_atom(first["CA"], first["N"], ace_c, CA_C_LENGTH, ACE_CH3_C_N_ANGLE, 180.0)
    nme_n = place_atom(last["N"], last["CA"], last["C"], PEPTIDE_BOND, CA_C_N_ANGLE, psi)
    nme_ch3 = place_atom(last["CA"], last["C"], nme_n, N_CA_LENGTH, C_N_CA_ANGLE, omega)

    atoms: list[tuple[str, str, int, np.ndarray]] = [
        ("CH3", "ACE", 1, ace_ch3),
        ("C", "ACE", 1, ace_c),
        ("O", "ACE", 1, ace_o),
    ]
    for i, res in enumerate(backbone):
        for name in ("N", "CA", "C", "O", "CB"):
            atoms.append((name, "ALA", i + 2, res[name]))
    atoms.append(("N", "NME", n_residues + 2, nme_n))
    atoms.append(("CH3", "NME", n_residues + 2, nme_ch3))

    return atoms


def format_pdb(
    atoms: list[tuple[str, str, int, np.ndarray]],
    n_residues: int,
    phi: float = PHI,
    psi: float = PSI,
    omega: float = OMEGA,
    chain: str = "A",
) -> str:
    """Render built atoms as a PDB file with provenance REMARKs.

    :param atoms: Output of :func:`build_polyalanine`.
    :param n_residues: Number of alanine residues, used in the REMARK header.
    :param phi: Backbone phi dihedral in degrees, used in the REMARK header.
    :param psi: Backbone psi dihedral in degrees, used in the REMARK header.
    :param omega: Backbone omega dihedral in degrees, used in the REMARK header.
    :param chain: Chain identifier.
    :return: The complete PDB text."""
    lines = [
        f"REMARK   1 CAPPED POLYALANINE, {n_residues} ALA RESIDUES, IDEAL ALPHA-HELIX",
        f"REMARK   1 SEQUENCE: ACE-(ALA){n_residues}-NME",
        f"REMARK   1 BACKBONE DIHEDRALS: PHI={phi} PSI={psi} OMEGA={omega} (DEGREES)",
        "REMARK   1 HEAVY ATOMS ONLY -- ADD HYDROGENS DURING FF PARAMETERIZATION",
        "REMARK   1 BUILT WITH MD_SIMULATIONS.PREP.POLYALANINE (PEPTIDEBUILDER IDEAL GEOMETRY)",
    ]

    for serial, (name, resname, resseq, xyz) in enumerate(atoms, start=1):
        lines.append(
            ATOM_FMT.format(
                serial=serial,
                name=name,
                resname=resname,
                chain=chain,
                resseq=resseq,
                x=xyz[0],
                y=xyz[1],
                z=xyz[2],
                element=name[0],
            )
        )

    _, resname, resseq, _ = atoms[-1]
    lines.append(f"TER   {len(atoms) + 1:5d}      {resname:>3s} {chain}{resseq:4d}" + " " * 55)
    lines.append("END   ")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "n_residues", type=int, nargs="+", help="Number(s) of alanine residues to build."
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="data/pdbs",
        help="Directory for the generated alaN.pdb files (default: data/pdbs).",
    )
    parser.add_argument("--phi", type=float, default=PHI, help="Backbone phi in degrees.")
    parser.add_argument("--psi", type=float, default=PSI, help="Backbone psi in degrees.")
    parser.add_argument("--omega", type=float, default=OMEGA, help="Backbone omega in degrees.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for n in args.n_residues:
        atoms = build_polyalanine(n, args.phi, args.psi, args.omega)
        text = format_pdb(atoms, n, args.phi, args.psi, args.omega)
        path = outdir / f"ala{n}.pdb"
        path.write_text(text)
        print(f"Written {len(atoms)} atoms ({n} ALA + caps) to {path}")


if __name__ == "__main__":
    main()
