#!/usr/bin/env python3
"""Add CONECT records to a PDB from bonds/constraints in a GROMACS ITP file."""

import argparse
import collections
from pathlib import Path


def parse_pdb(pdb_path: str) -> tuple[list[str], list[int]]:
    """Return (all_lines, serial_numbers) from a PDB.

    :param pdb_path: Path to the input PDB file.
    :return: All lines verbatim, and the ordered list of atom serial numbers."""
    lines = Path(pdb_path).read_text().splitlines(keepends=True)
    serials = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            serials.append(int(line[6:11].strip()))
    return lines, serials


def parse_itp_bonds(itp_path: str) -> list[tuple[int, int]]:
    """Parse [ bonds ] and [ constraints ] from an ITP file.

    Preprocessor guards (#ifdef / #ifndef / #endif) are ignored — all entries
    in the two sections are collected regardless of compile-time flags.

    :param itp_path: Path to the GROMACS ITP file.
    :return: Deduplicated list of (i, j) pairs, 1-indexed as in the ITP."""
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    section = None

    with open(itp_path) as f:
        for line in f:
            stripped = line.split(";", 1)[0].strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped[1:-1].strip()
                continue
            if section in ("bonds", "constraints"):
                fields = stripped.split()
                if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
                    i, j = int(fields[0]), int(fields[1])
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((i, j))

    return pairs


def build_conect_records(
    itp_pairs: list[tuple[int, int]],
    serial_numbers: list[int],
) -> list[str]:
    """Build PDB CONECT records from ITP atom indices and PDB serial numbers.

    ITP indices are 1-based positions into the atom list for a single molecule.
    The bond pattern is replicated for every molecule in the PDB by detecting
    atoms_per_mol as the maximum atom index that appears in the ITP bonds.

    :param itp_pairs: List of (i, j) pairs, 1-indexed (single-molecule numbering).
    :param serial_numbers: Ordered list of PDB serial numbers, one per ATOM record.
    :return: List of CONECT record strings (with newlines)."""
    if not itp_pairs:
        return []

    atoms_per_mol = max(max(i, j) for i, j in itp_pairs)
    n_total = len(serial_numbers)
    num_mols = n_total // atoms_per_mol
    remainder = n_total % atoms_per_mol

    if remainder:
        print(
            f"Warning: {n_total} atoms is not a multiple of {atoms_per_mol} "
            f"(atoms per molecule inferred from ITP). "
            f"Bonds will be written for {num_mols} complete molecules; "
            f"{remainder} trailing atom(s) will have no bonds."
        )

    adjacency: dict[int, list[int]] = collections.defaultdict(list)

    for mol in range(num_mols):
        offset = mol * atoms_per_mol
        for i, j in itp_pairs:
            si = serial_numbers[offset + i - 1]
            sj = serial_numbers[offset + j - 1]
            adjacency[si].append(sj)
            adjacency[sj].append(si)

    records = []
    for serial in sorted(adjacency):
        partners = adjacency[serial]
        # PDB CONECT allows up to 4 partners per record; wrap if more
        for start in range(0, len(partners), 4):
            chunk = partners[start : start + 4]
            line = f"CONECT{serial:5d}" + "".join(f"{p:5d}" for p in chunk)
            records.append(line + "\n")

    return records


def write_pdb_with_conect(
    input_lines: list[str],
    conect_records: list[str],
    output_path: str,
) -> None:
    """Write the input PDB with CONECT records inserted before END.

    :param input_lines: Lines of the original PDB.
    :param conect_records: CONECT lines to insert.
    :param output_path: Destination path."""
    # Drop any existing CONECT / END lines; we'll append fresh ones
    kept = [line for line in input_lines if not line.startswith(("CONECT", "END"))]
    with open(output_path, "w") as f:
        f.writelines(kept)
        f.writelines(conect_records)
        f.write("END\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdb", help="Input PDB file.")
    parser.add_argument("itp", help="GROMACS ITP file containing [ bonds ] / [ constraints ].")
    parser.add_argument("output", help="Output PDB file with CONECT records.")
    args = parser.parse_args()

    lines, serials = parse_pdb(args.pdb)
    print(f"Read {len(serials)} atoms from {args.pdb}")

    pairs = parse_itp_bonds(args.itp)
    print(f"Found {len(pairs)} unique bonds/constraints in {args.itp}")
    if pairs:
        atoms_per_mol = max(max(i, j) for i, j in pairs)
        num_mols = len(serials) // atoms_per_mol
        print(
            f"Detected {atoms_per_mol} atoms/molecule → replicating bonds for {num_mols} molecule(s)"
        )

    conect = build_conect_records(pairs, serials)
    write_pdb_with_conect(lines, conect, args.output)
    print(f"Written {len(conect)} CONECT records to {args.output}")


if __name__ == "__main__":
    main()
