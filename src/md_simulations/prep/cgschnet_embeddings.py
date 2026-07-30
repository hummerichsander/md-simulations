#!/usr/bin/env python3
"""Recover CGSchNet bead embeddings (atom types) from a distributed artifact.

The released transferable model (e.g. ``pingzhili/cg-schnet``) ships per-protein
``configurations/*.pt`` objects that already carry the CG ``atom_types`` — the
exact embedding indices the model was trained with. This tool reads those out so
you never hand-guess them:

- write the atom types to a ``.npy`` the ``cgschnet`` engine consumes directly, and
- (optionally, with ``--pdb``) recover the residue → type-index table and write a
  ``cgschnet_map``-compatible mapping JSON, so you can map *new* proteins with the
  model's real embedding scheme.

Only the checkpoint loading needs PyTorch; the extraction logic is pure so it can
be reasoned about and tested without torch.
"""

import argparse
import json
from pathlib import Path

import numpy as np

_ATOM_TYPE_KEYS = ("atom_types", "embeddings", "types", "z", "atomic_numbers")


def _to_int_array(x) -> np.ndarray:
    """Coerce a tensor/array-like of per-bead types to a 1-D int64 array.

    :param x: A torch tensor, numpy array, or nested sequence.
    :return: 1-D ``np.int64`` array.
    :raises ValueError: If the data is not 1-D after squeezing."""
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    arr = np.asarray(x).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D atom types, got shape {np.asarray(x).shape}.")
    return arr.astype(np.int64)


def extract_atom_types(obj) -> np.ndarray:
    """Pull the per-bead embedding types out of a loaded configuration object.

    Handles the common shapes a released config may take: a dict keyed by
    ``atom_types``/``embeddings``/…, an mlcg ``AtomicData``-like object exposing
    ``.atom_types``/``.z``, a list/tuple of such objects (first is used), or a
    bare tensor/array.

    :param obj: The loaded configuration object.
    :return: 1-D ``np.int64`` array of bead types.
    :raises KeyError: If a dict carries none of the recognised keys."""
    if isinstance(obj, dict):
        for key in _ATOM_TYPE_KEYS:
            if key in obj:
                return _to_int_array(obj[key])
        raise KeyError(
            f"None of {_ATOM_TYPE_KEYS} found in config dict (keys: {list(obj)[:10]})."
        )
    if isinstance(obj, (list, tuple)):
        if not obj:
            raise ValueError("Empty configuration container.")
        # A list of scalars is the type array itself; a list of objects is a
        # container of configurations (use the first).
        if isinstance(obj[0], (int, float, np.integer, np.floating)):
            return _to_int_array(obj)
        return extract_atom_types(obj[0])
    for attr in ("atom_types", "z"):
        if hasattr(obj, attr):
            return _to_int_array(getattr(obj, attr))
    return _to_int_array(obj)


def residue_type_table(
    resnames: list[str], atom_types: np.ndarray
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Pair per-bead residue names with their embedding types.

    :param resnames: Residue name per bead (in bead order).
    :param atom_types: Embedding type per bead (same length/order).
    :return: (table, conflicts) where table maps residue name → type (first seen)
        and conflicts maps residue names seen with more than one type (e.g.
        terminal beads) to the sorted list of observed types.
    :raises ValueError: If the lengths differ."""
    if len(resnames) != len(atom_types):
        raise ValueError(
            f"Bead count mismatch: {len(resnames)} residues vs {len(atom_types)} types. "
            "Make sure the PDB matches the configuration's CG scheme (Cα-only)."
        )
    seen: dict[str, set[int]] = {}
    table: dict[str, int] = {}
    for name, t in zip(resnames, atom_types):
        t = int(t)
        seen.setdefault(name, set()).add(t)
        table.setdefault(name, t)
    conflicts = {name: sorted(ts) for name, ts in seen.items() if len(ts) > 1}
    return table, conflicts


def _load_config_object(path: str | Path):
    """Load a configuration/checkpoint object with a lazy torch import.

    :param path: Path to the ``.pt`` file.
    :return: The unpickled object."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required to read CGSchNet configuration files. Install it in "
            "your cgschnet conda env (see README)."
        ) from e
    # weights_only=False: these files contain full objects (AtomicData, priors).
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _ca_residue_names(pdb_path: str | Path) -> list[str]:
    """List residue names for residues containing a Cα, in topology order.

    :param pdb_path: Path to a structure with Cα atoms.
    :return: Residue names, one per Cα bead."""
    import mdtraj as md

    top = md.load(str(pdb_path)).topology
    return [res.name for res in top.residues if any(a.name == "CA" for a in res.atoms)]


def main() -> None:
    """Entry point: extract embeddings (and optionally the mapping table)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="Distributed configuration/checkpoint .pt file.")
    parser.add_argument(
        "-o", "--out", required=True, help="Output .npy path for the bead embeddings."
    )
    parser.add_argument(
        "--pdb",
        default=None,
        help="Matching structure (Cα per residue) to recover the residue→type table.",
    )
    parser.add_argument(
        "--mapping-out",
        default=None,
        help="Write a cgschnet_map-compatible mapping JSON (requires --pdb).",
    )
    args = parser.parse_args()

    atom_types = extract_atom_types(_load_config_object(args.config))
    np.save(args.out, atom_types)
    print(f"Wrote {len(atom_types)} bead embeddings → {args.out}")

    if args.pdb is None:
        return

    table, conflicts = residue_type_table(_ca_residue_names(args.pdb), atom_types)
    print("\nResidue → embedding type:")
    for name, t in sorted(table.items(), key=lambda kv: kv[1]):
        print(f"  {name:4s} {t}")
    if conflicts:
        print(
            "\nWARNING: these residues appear with multiple types (likely terminal/special "
            "beads); a residue-only slicing map cannot capture position-dependent types:"
        )
        for name, ts in conflicts.items():
            print(f"  {name}: {ts}")

    if args.mapping_out:
        beads = [[name, "CA", t] for name, t in table.items()]
        Path(args.mapping_out).write_text(json.dumps({"beads": beads}, indent=2))
        print(f"\nWrote mapping → {args.mapping_out}")


if __name__ == "__main__":
    main()
