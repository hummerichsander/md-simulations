#!/usr/bin/env python3
"""Fit the fixed priors of a CGSchNet trained from scratch, and write its mlcg inputs.

Fine-tuning reuses the transferable checkpoint's priors (:mod:`.cgschnet_finetune`); a
molecule outside that model's vocabulary -- alanine dipeptide on the ``AL_CG_MAP`` beads,
say -- needs its own. This module takes a force-matching dataset from
:mod:`.cgschnet_dataset` (built *without* ``--cgschnet-config``) and writes the three
things ``mlcg-train_h5`` and :class:`~md_simulations.torch_potentials.cgschnet.CGSchNetPotential`
then need:

* ``priors.pt`` -- a ``ModuleDict`` of ``GradientsOut``-wrapped bond, angle, dihedral and
  repulsion priors, Boltzmann-inverted from the dataset. ``mlcg-combine_model --prior``
  reads it as is.
* ``<name>_bysegment.h5`` -- the delta forces (total minus prior), cut into contiguous time
  blocks so a whole block can be held out for validation.
* ``configurations.pt`` -- a few dataset frames as ``AtomicData`` carrying ``atom_types``,
  masses and the prior neighbour lists, keyed by prior name.

The CG topology is derived from the all-atom bond graph rather than from bead order: two
beads are bonded when the all-atom path between them runs through non-bead atoms only.
mlcg's ``add_chain_*`` helpers assume a linear chain in index order, which is wrong as
soon as a bead branches off it (CB on CA), and silently so for the dihedrals.

Usage::

    md-sim-cgschnet-priors --dataset data/alaX/ala2_fm_330K
"""

import argparse
import json
import logging
from pathlib import Path

import mdtraj as md
import numpy as np
import torch
from torch import Tensor

from md_simulations.prep.cgschnet_finetune import write_multi_h5

logger = logging.getLogger("md_simulations.prep.cgschnet_priors")

KB_KCAL = 0.0019872041
# pairs this many bonds apart or closer are covered by the bond, angle and dihedral terms.
MAX_BONDED_SEPARATION = 3


def cg_bonds(aa_topology: md.Topology, indices: np.ndarray) -> list[tuple[int, int]]:
    """Bond the beads whose all-atom path runs through non-bead atoms only.

    :param aa_topology: All-atom topology with bonds.
    :param indices: All-atom index of each bead, in bead order.
    :return: Sorted bead pairs ``(i, j)`` with ``i < j``."""
    neighbours: dict[int, set[int]] = {atom.index: set() for atom in aa_topology.atoms}
    for a, b in aa_topology.bonds:
        neighbours[a.index].add(b.index)
        neighbours[b.index].add(a.index)
    bead_of = {int(atom): bead for bead, atom in enumerate(indices)}

    bonds = set()
    for bead, start in enumerate(indices):
        seen, frontier = {int(start)}, [int(start)]
        while frontier:
            atom = frontier.pop()
            for nxt in neighbours[atom] - seen:
                seen.add(nxt)
                if nxt in bead_of:
                    bonds.add(tuple(sorted((bead, bead_of[nxt]))))
                else:
                    frontier.append(nxt)

    return sorted(bonds)


def cg_topology(aa_topology: md.Topology, indices: np.ndarray):
    """Build the mlcg CG topology: one type per bead, with bonds, angles and dihedrals.

    :param aa_topology: All-atom topology with bonds.
    :param indices: All-atom index of each bead, in bead order.
    :return: An ``mlcg.geometry.topology.Topology``."""
    from mlcg.geometry.topology import Topology

    topology = Topology()
    for bead, index in enumerate(indices):
        atom = aa_topology.atom(int(index))
        topology.add_atom(bead, atom.name, atom.residue.name, atom.residue.resSeq)

    bonds = cg_bonds(aa_topology, indices)
    neighbours: dict[int, set[int]] = {bead: set() for bead in range(len(indices))}
    for i, j in bonds:
        topology.add_bond(i, j)
        neighbours[i].add(j)
        neighbours[j].add(i)

    for j in range(len(indices)):
        for i in neighbours[j]:
            for k in neighbours[j]:
                if i < k:
                    topology.add_angle(i, j, k)

    for j, k in bonds:
        for i in neighbours[j] - {k}:
            for m in neighbours[k] - {j, i}:
                topology.add_dihedral(i, j, k, m)

    return topology


def repulsion_pairs(topology) -> Tensor:
    """Bead pairs more than ``MAX_BONDED_SEPARATION`` bonds apart (or disconnected).

    :param topology: The mlcg CG topology from :func:`cg_topology`.
    :return: ``(2, n_pairs)`` index tensor with ``i < j`` in each column."""
    n = topology.n_atoms
    hops = np.full((n, n), np.inf)
    np.fill_diagonal(hops, 0)
    for i, j in zip(*topology.bonds):
        hops[i, j] = hops[j, i] = 1
    for k in range(n):
        hops = np.minimum(hops, hops[:, [k]] + hops[[k], :])

    i, j = np.triu_indices(n, k=1)
    keep = hops[i, j] > MAX_BONDED_SEPARATION
    return torch.tensor(np.stack([i[keep], j[keep]]), dtype=torch.long)


def prior_neighbor_lists(topology) -> dict[str, dict]:
    """Neighbour lists of every prior term, keyed by the prior's name.

    :param topology: The mlcg CG topology from :func:`cg_topology`.
    :return: Name -> mlcg neighbour-list dict, for the terms that have interactions."""
    from mlcg.neighbor_list.neighbor_list import make_neighbor_list
    from mlcg.nn import Dihedral, HarmonicAngles, HarmonicBonds, Repulsion

    neighbor_lists = {
        HarmonicBonds.name: topology.neighbor_list("bonds"),
        HarmonicAngles.name: topology.neighbor_list("angles"),
        Dihedral.name: topology.neighbor_list("dihedrals"),
        Repulsion.name: make_neighbor_list(
            tag=Repulsion.name, order=2, index_mapping=repulsion_pairs(topology)
        ),
    }
    return {k: v for k, v in neighbor_lists.items() if v["index_mapping"].shape[1] > 0}


def atomic_data(coords: np.ndarray, types: Tensor, masses: Tensor, neighbor_lists: dict) -> list:
    """Wrap CG frames as mlcg ``AtomicData``.

    :param coords: ``(F, N, 3)`` CG coordinates in Angstrom.
    :param types: ``(N,)`` bead types.
    :param masses: ``(N,)`` bead masses.
    :param neighbor_lists: Prior neighbour lists from :func:`prior_neighbor_lists`.
    :return: One ``AtomicData`` per frame."""
    from mlcg.data import AtomicData

    return [
        AtomicData.from_points(
            pos=torch.as_tensor(frame, dtype=torch.float32),
            atom_types=types,
            masses=masses,
            neighborlist=neighbor_lists,
        )
        for frame in coords
    ]


def fit_priors(
    coords: np.ndarray, types: Tensor, masses: Tensor, neighbor_lists: dict, temperature: float
) -> torch.nn.ModuleDict:
    """Boltzmann-invert every prior term from the frames' feature histograms.

    :param coords: ``(F, N, 3)`` CG coordinates in Angstrom.
    :param types: ``(N,)`` bead types.
    :param masses: ``(N,)`` bead masses.
    :param neighbor_lists: Prior neighbour lists from :func:`prior_neighbor_lists`.
    :param temperature: Temperature of the source ensemble in K.
    :return: Prior name -> ``GradientsOut``-wrapped prior."""
    from mlcg.geometry.statistics import fit_baseline_models
    from mlcg.nn import Dihedral, GradientsOut, HarmonicAngles, HarmonicBonds, Repulsion
    from torch_geometric.data.collate import collate

    data_list = atomic_data(coords, types, masses, neighbor_lists)
    data, _, _ = collate(
        data_list[0].__class__, data_list=data_list, increment=True, add_batch=True
    )
    classes = [HarmonicBonds, HarmonicAngles, Dihedral, Repulsion]
    priors, _ = fit_baseline_models(
        data, 1.0 / (KB_KCAL * temperature), [c for c in classes if c.name in neighbor_lists]
    )

    return torch.nn.ModuleDict({k: GradientsOut(v, targets="forces") for k, v in priors.items()})


def prior_forces(
    priors: torch.nn.ModuleDict,
    coords: np.ndarray,
    types: Tensor,
    masses: Tensor,
    neighbor_lists: dict,
    chunk: int = 20000,
) -> np.ndarray:
    """Evaluate the summed prior forces on every frame.

    :param priors: Output of :func:`fit_priors`.
    :param coords: ``(F, N, 3)`` CG coordinates in Angstrom.
    :param types: ``(N,)`` bead types.
    :param masses: ``(N,)`` bead masses.
    :param neighbor_lists: Prior neighbour lists from :func:`prior_neighbor_lists`.
    :param chunk: Frames per forward.
    :return: ``(F, N, 3)`` prior forces in kcal/mol/Angstrom."""
    from torch_geometric.loader import DataLoader

    forces = []
    for start in range(0, len(coords), chunk):
        block = coords[start : start + chunk]
        data = next(iter(DataLoader(atomic_data(block, types, masses, neighbor_lists), len(block))))
        total = torch.zeros_like(data.pos)
        for name, prior in priors.items():
            data = prior(data)
            total += data.out[name]["forces"].detach()
        forces.append(total.reshape(block.shape).numpy())

    return np.concatenate(forces)


def build_priors(
    dataset_dir: Path,
    fit_frames: int = 50000,
    n_blocks: int = 10,
    n_configurations: int = 32,
) -> dict[str, object]:
    """Fit the priors of a dataset and write ``priors.pt``, the h5 and ``configurations.pt``.

    :param dataset_dir: Output directory of :mod:`.cgschnet_dataset` (one run, no prior).
    :param fit_frames: Frames, evenly strided, the priors are fit on.
    :param n_blocks: Contiguous time blocks the h5 is cut into.
    :param n_configurations: Evenly spaced frames written as starting configurations.
    :return: Summary of what was written.
    :raises ValueError: If the dataset pools several runs or already has delta forces."""
    meta = json.loads((dataset_dir / "dataset.json").read_text())
    if len(meta["runs"]) != 1 or meta["delta_forces"]:
        raise ValueError(
            f"{dataset_dir} must hold one run without prior subtraction; time blocks of "
            "pooled runs would straddle run boundaries."
        )

    aa_topology = md.load_topology(meta["aa_topology"])
    indices = np.load(dataset_dir / "bead_indices.npy")
    coords = np.load(dataset_dir / "cg_coords.npy")
    forces = np.load(dataset_dir / "cg_forces.npy")

    topology = cg_topology(aa_topology, indices)
    neighbor_lists = prior_neighbor_lists(topology)
    types = torch.arange(len(indices))
    masses = torch.tensor([aa_topology.atom(int(i)).element.mass for i in indices])
    for name, nl in neighbor_lists.items():
        logger.info(f"{name}: {nl['index_mapping'].t().tolist()}")

    stride = max(1, len(coords) // fit_frames)
    priors = fit_priors(coords[::stride], types, masses, neighbor_lists, meta["temperature"])
    torch.save(priors, dataset_dir / "priors.pt")

    delta = forces - prior_forces(priors, coords, types, masses, neighbor_lists)
    explained = 1 - (delta**2).mean() / (forces**2).mean()
    logger.info(
        f"delta forces: RMS {np.sqrt((delta**2).mean()):.2f} vs total "
        f"{np.sqrt((forces**2).mean()):.2f} kcal/mol/A (prior explains {100 * explained:.1f}%)"
    )
    np.save(dataset_dir / "cg_delta_forces.npy", delta)

    lengths = np.diff(np.linspace(0, len(coords), n_blocks + 1).astype(np.int64))
    names = write_multi_h5(
        dataset_dir / f"{meta['name']}_bysegment.h5",
        meta["name"],
        coords,
        delta,
        types.numpy(),
        lengths,
    )

    picks = np.linspace(0, len(coords) - 1, n_configurations).astype(np.int64)
    torch.save(
        atomic_data(coords[picks], types, masses, neighbor_lists),
        dataset_dir / "configurations.pt",
    )

    summary = {
        "priors": sorted(priors),
        "neighbor_lists": {k: v["index_mapping"].t().tolist() for k, v in neighbor_lists.items()},
        "fit_frames": int(len(coords[::stride])),
        "temperature": meta["temperature"],
        "prior_explained_force_variance": float(explained),
        "segments": names,
        "segment_lengths": lengths.tolist(),
        "embedding_size": int(len(indices)),
    }
    (dataset_dir / "priors.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote priors, {len(names)} segments and {n_configurations} configurations")

    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", required=True, type=Path, help="md-sim-cgschnet-dataset output."
    )
    parser.add_argument(
        "--fit-frames", type=int, default=50000, help="Frames the priors are fit on."
    )
    parser.add_argument("--n-blocks", type=int, default=10, help="Contiguous h5 time blocks.")
    parser.add_argument("--n-configurations", type=int, default=32, help="Starting configurations.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_priors(args.dataset, args.fit_frames, args.n_blocks, args.n_configurations)


if __name__ == "__main__":
    main()
