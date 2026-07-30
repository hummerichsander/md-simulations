#!/usr/bin/env python3
"""Build a MARTINI coarse-graining mapping matrix from an all-atom PDB file."""

import argparse
import collections
import importlib.util
from pathlib import Path

import numpy as np


def _vermouth_mappings_base() -> Path:
    """Locate vermouth's bundled mappings directory via the installed package.

    :return: Path to ``<vermouth>/data/mappings``.
    :raises ModuleNotFoundError: If vermouth is not installed (``[martini]`` extra)."""
    spec = importlib.util.find_spec("vermouth")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "vermouth is required for CG mapping. Install it with the 'martini' extra: "
            "`uv sync --extra martini`."
        )
    return Path(spec.origin).parent / "data" / "mappings"


def _mapping_dir(to_ff: str) -> Path:
    """Return the mappings directory for a target force field.

    The top-level directory holds martini22/martini22p files; all others use
    per-FF subdirectories.

    :param to_ff: Target MARTINI version label.
    :return: Path to the directory containing that FF's ``.map`` files."""
    base = _vermouth_mappings_base()
    if to_ff in ("martini22", "martini22p"):
        return base
    return base / to_ff


def _parse_pdb_atoms(pdb_path: str) -> list[dict]:
    """Parse ATOM/HETATM records from a PDB file.

    :param pdb_path: Path to the PDB file.
    :return: Ordered list of atom dicts with keys atomname, resname, chain, resid."""

    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atoms.append(
                {
                    "atomname": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip() or "A",
                    "resid": int(line[22:26].strip()),
                }
            )
    return atoms


def _parse_map_file(path: Path) -> dict[str, tuple[str, bool]]:
    """Parse the [ atoms ] section of a vermouth .map file.

    Atoms prefixed with '!' in the bead column are null-weight: they appear in
    the topology but do not contribute to the bead's center of mass.

    :param path: Path to the .map file.
    :return: Dict mapping atom_name to (bead_name, is_null_weight)."""

    result = {}
    in_atoms = False
    with open(path) as f:
        for line in f:
            stripped = line.split(";", 1)[0].strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                in_atoms = stripped[1:-1].strip() == "atoms"
                continue
            if in_atoms:
                parts = stripped.split()
                if len(parts) >= 3:
                    atom_name, bead_name = parts[1], parts[2]
                    null = bead_name.startswith("!")
                    result[atom_name] = (bead_name.lstrip("!"), null)
    return result


def _bead_weights(atom_to_bead: dict[str, tuple[str, bool]]) -> dict[str, dict[str, float]]:
    """Compute equal per-bead atom weights from an atom-to-bead mapping.

    Null-weight atoms receive weight 0; the remaining atoms share weight 1/k
    where k is the number of non-null atoms assigned to that bead.

    :param atom_to_bead: Dict mapping atom_name to (bead_name, is_null_weight).
    :return: Ordered dict mapping bead_name to {atom_name: weight}, beads in
        first-encounter order."""

    bead_order = list(dict.fromkeys(bead for bead, _ in atom_to_bead.values()))
    counts = collections.Counter(bead for _, (bead, null) in atom_to_bead.items() if not null)
    weights: dict[str, dict[str, float]] = {bead: {} for bead in bead_order}
    for atom, (bead, null) in atom_to_bead.items():
        weights[bead][atom] = 0.0 if null else 1.0 / counts[bead]
    return weights


def build_mapping_matrix(
    pdb_path: str,
    from_ff: str = "charmm36",
    to_ff: str = "martini3001",
    extra_mappings: str | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the CG mapping matrix M of shape (N_CG, N_AA).

    Applying M to all-atom coordinates (N_AA, 3) yields CG bead positions
    (N_CG, 3). Residues without a vermouth map file are skipped with a warning.

    :param pdb_path: Path to the all-atom PDB file.
    :param from_ff: Source force field label as used in the map filenames,
        e.g. ``"charmm36"`` or ``"amber"``.
    :param to_ff: Target MARTINI version, e.g. ``"martini3001"`` or
        ``"martini22"``.
    :param extra_mappings: Optional path to a directory that mirrors the
        vermouth mappings layout (i.e. contains subdirs named after the
        target FF).  Files found here take precedence over vermouth built-ins.
    :return: A tuple (M, bead_labels, atom_labels) where M has shape
        (N_CG, N_AA), bead_labels are ordered CG bead identifiers
        e.g. ``["ALA1:BB", ...]``, and atom_labels are ordered all-atom
        identifiers e.g. ``["ALA1:N", "ALA1:CA", ...]``."""

    mapping_dir = _mapping_dir(to_ff)
    if not mapping_dir.exists():
        base = _vermouth_mappings_base()
        raise FileNotFoundError(
            f"Mapping directory not found: {mapping_dir}\n"
            f"Available targets: {[d.name for d in base.iterdir() if d.is_dir()]}"
        )

    extra_dir: Path | None = None
    if extra_mappings is not None:
        extra_dir = Path(extra_mappings) / to_ff

    atoms = _parse_pdb_atoms(pdb_path)
    n_aa = len(atoms)
    atom_labels = [f"{a['resname']}{a['resid']}:{a['atomname']}" for a in atoms]

    residues: dict[tuple, list[tuple[int, str]]] = collections.OrderedDict()
    for i, atom in enumerate(atoms):
        key = (atom["chain"], atom["resid"], atom["resname"])
        residues.setdefault(key, []).append((i, atom["atomname"]))

    rows: list[np.ndarray] = []
    bead_labels: list[str] = []
    skipped_resnames: set[str] = set()

    for (chain, resid, resname), res_atoms in residues.items():
        map_filename = f"{resname.lower()}.{from_ff}.map"
        map_file = mapping_dir / map_filename
        if extra_dir is not None:
            extra_file = extra_dir / map_filename
            if extra_file.exists():
                map_file = extra_file
        if not map_file.exists():
            skipped_resnames.add(resname)
            continue

        atom_to_bead = _parse_map_file(map_file)
        weights = _bead_weights(atom_to_bead)

        name_to_global_idx: dict[str, int] = {name: idx for idx, name in res_atoms}

        res_label = f"{resname}{resid}"
        for bead, bead_atom_weights in weights.items():
            row = np.zeros(n_aa)
            for atom_name, weight in bead_atom_weights.items():
                if atom_name in name_to_global_idx:
                    row[name_to_global_idx[atom_name]] = weight
            row_sum = row.sum()
            if row_sum > 0:
                row /= row_sum
            rows.append(row)
            bead_labels.append(f"{res_label}:{bead}")

    if skipped_resnames:
        print(
            f"Warning: no {from_ff}->{to_ff} mapping found for residue(s): "
            f"{', '.join(sorted(skipped_resnames))} — skipped."
        )

    M = np.stack(rows) if rows else np.zeros((0, n_aa))
    return M, bead_labels, atom_labels


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdb", help="Input all-atom PDB file.")
    parser.add_argument("output", help="Output path for the mapping matrix (.npy).")
    parser.add_argument(
        "--from-ff", default="charmm36", help="Source force field label (default: charmm36)."
    )
    parser.add_argument(
        "--to-ff", default="martini3001", help="Target MARTINI version (default: martini3001)."
    )
    parser.add_argument(
        "--extra-mappings",
        default=None,
        metavar="DIR",
        help=(
            "Directory with custom map files, organised like the vermouth mappings tree "
            "(i.e. DIR/<to-ff>/<resname>.<from-ff>.map).  "
            "Files here take precedence over vermouth built-ins."
        ),
    )
    parser.add_argument(
        "--labels", action="store_true", help="Also save bead and atom labels as .txt files."
    )
    args = parser.parse_args()

    M, bead_labels, atom_labels = build_mapping_matrix(
        args.pdb, args.from_ff, args.to_ff, args.extra_mappings
    )

    output = Path(args.output)
    np.save(output, M)
    print(f"Saved mapping matrix {M.shape} to {output}")
    print(f"  {M.shape[1]} all-atom atoms  →  {M.shape[0]} CG beads")

    if args.labels:
        bead_path = output.with_suffix(".beads.txt")
        atom_path = output.with_suffix(".atoms.txt")
        bead_path.write_text("\n".join(bead_labels))
        atom_path.write_text("\n".join(atom_labels))
        print(f"  Bead labels → {bead_path}")
        print(f"  Atom labels → {atom_path}")


if __name__ == "__main__":
    main()
