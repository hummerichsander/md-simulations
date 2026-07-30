#!/usr/bin/env python3
"""Build the CG structure + embeddings a CGSchNet run needs from an all-atom PDB.

Implements the *slicing* CG strategy (one retained atom → one bead), which is
what the transferable protein models use (e.g. per-residue Cα, optionally Cβ).
Reads a JSON mapping and writes two files consumed by the ``cgschnet`` engine:

- ``<out>.pdb``        — the CG structure (retained beads), a topology reference
- ``<out>.embeddings.npy`` — per-bead integer embedding types, in bead order

Center-of-mass / multi-atom bead mappings are out of scope here; use mlcg's
``build_cg_matrix`` for those.

Mapping JSON format::

    {"beads": [["ALA", "CA", 1], ["ARG", "CA", 2], ...]}

Each entry is ``[residue_name, atom_name, embedding_type]``. Atoms whose
(residue, atom) pair is not listed are dropped from the CG representation.
"""

import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np


def load_mapping(path: str | Path) -> dict[tuple[str, str], int]:
    """Load a slicing CG mapping from JSON.

    :param path: Path to the mapping JSON file.
    :return: Dict from (residue_name, atom_name) to embedding type."""
    data = json.loads(Path(path).read_text())
    return {(resname, atomname): int(t) for resname, atomname, t in data["beads"]}


def build_cg_structure(
    pdb_path: str | Path,
    mapping: dict[tuple[str, str], int],
) -> tuple[md.Trajectory, np.ndarray]:
    """Slice an all-atom structure down to its CG beads.

    :param pdb_path: Path to the all-atom PDB.
    :param mapping: (residue_name, atom_name) → embedding type.
    :return: A tuple (CG trajectory, per-bead embedding types)."""
    traj = md.load(str(pdb_path))
    keep, types = [], []
    for atom in traj.topology.atoms:
        t = mapping.get((atom.residue.name, atom.name))
        if t is not None:
            keep.append(atom.index)
            types.append(t)
    if not keep:
        raise ValueError(
            "No atoms matched the CG mapping — check residue/atom names and the mapping file."
        )
    return traj.atom_slice(keep), np.asarray(types, dtype=np.int64)


def main() -> None:
    """Entry point: write the CG PDB and embeddings for a given all-atom PDB."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdb", help="Input all-atom PDB file.")
    parser.add_argument("mapping", help="CG mapping JSON file.")
    parser.add_argument("-o", "--out", required=True, help="Output basename (without extension).")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    cg, types = build_cg_structure(args.pdb, mapping)

    out = Path(args.out)
    out_pdb = out.with_suffix(".pdb")
    out_emb = out.with_suffix(".embeddings.npy")
    cg.save_pdb(str(out_pdb))
    np.save(out_emb, types)

    print(f"Wrote CG structure ({cg.n_atoms} beads) → {out_pdb}")
    print(f"Wrote embeddings {types.shape} → {out_emb}")


if __name__ == "__main__":
    main()
