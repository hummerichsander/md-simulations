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
  blocks within each pooled run so a whole block, or a whole run, can be held out for
  validation.
* ``configurations.pt`` -- a few dataset frames as ``AtomicData`` carrying ``atom_types``,
  masses and the prior neighbour lists, keyed by prior name.

The CG topology is derived from the all-atom bond graph rather than from bead order: two
beads are bonded when the all-atom path between them runs through non-bead atoms only.
mlcg's ``add_chain_*`` helpers assume a linear chain in index order, which is wrong as
soon as a bead branches off it (CB on CA), and silently so for the dihedrals.

That generic ``graph`` scheme is the default. It fits omega (CA-C-N-CA across a peptide bond) as
its own term at Fourier degree 1 and every other dihedral at 3, which covers polyalanine on
N CA CB C. On a protein mapped to N CA CB C O its dihedrals still include stiff ones (the carbonyl
plane, the CG proline ring; ~1e11 coefficients on chignolin), so ``--scheme protein`` instead
reproduces the terms and Fourier degrees of mlcg's transferable model
(:func:`protein_neighbor_lists`).

Usage::

    md-sim-cgschnet-priors --dataset data/alaX/ala2_fm_330K
    md-sim-cgschnet-priors --dataset data/cln/cln_fm_330K --scheme protein
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import mdtraj as md
import numpy as np
import torch
from torch import Tensor

from md_simulations.prep.cgschnet_finetune import write_multi_h5

logger = logging.getLogger("md_simulations.prep.cgschnet_priors")

KB_KCAL = 0.0019872041
# pairs this many bonds apart or closer are covered by the bond, angle and dihedral terms.
MAX_BONDED_SEPARATION = 3
CAP_RESIDUES = ("ACE", "NME")


def bead_types(aa_topology: md.Topology, indices: np.ndarray) -> Tensor:
    """Embedding type of each bead: CA by residue, cap beads by residue and name, the rest by name.

    The transferable models' scheme, which ``CGSchNetPotential`` checks for: N, C and O share one
    type across residues and CA carries the residue identity. Types are numbered in order of first
    appearance, so a molecule whose beads are all distinct (ala2) gets ``arange``.

    :param aa_topology: All-atom topology.
    :param indices: All-atom index of each bead, in bead order.
    :return: ``(N,)`` integer bead types."""
    keys = []
    for index in indices:
        atom = aa_topology.atom(int(index))
        if atom.residue.name in CAP_RESIDUES:
            keys.append((atom.residue.name, atom.name))
        elif atom.name == "CA":
            keys.append(("CA", atom.residue.name))
        else:
            keys.append((atom.name,))
    labels = {key: label for label, key in enumerate(dict.fromkeys(keys))}
    return torch.tensor([labels[key] for key in keys])


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


def separated_pairs(n: int, bonds: list[tuple[int, int]], min_separation: int) -> Tensor:
    """Bead pairs at least ``min_separation`` bonds apart (or disconnected).

    :param n: Number of beads.
    :param bonds: Bead pairs bonded in the CG topology.
    :param min_separation: Fewest bonds between the two beads of a kept pair.
    :return: ``(2, n_pairs)`` index tensor with ``i < j`` in each column."""
    hops = np.full((n, n), np.inf)
    np.fill_diagonal(hops, 0)
    for i, j in bonds:
        hops[i, j] = hops[j, i] = 1
    for k in range(n):
        hops = np.minimum(hops, hops[:, [k]] + hops[[k], :])

    i, j = np.triu_indices(n, k=1)
    keep = hops[i, j] >= min_separation
    return torch.tensor(np.stack([i[keep], j[keep]]), dtype=torch.long)


def repulsion_pairs(topology) -> Tensor:
    """Bead pairs more than ``MAX_BONDED_SEPARATION`` bonds apart (or disconnected).

    :param topology: The mlcg CG topology from :func:`cg_topology`.
    :return: ``(2, n_pairs)`` index tensor with ``i < j`` in each column."""
    bonds = list(zip(*topology.bonds))
    return separated_pairs(topology.n_atoms, bonds, MAX_BONDED_SEPARATION + 1)


def is_omega(topology, dihedral: tuple[int, int, int, int]) -> bool:
    """Whether a dihedral is CA-C-N-CA across a peptide bond, in either direction.

    :param topology: The mlcg CG topology from :func:`cg_topology`.
    :param dihedral: Bead indices ``(i, j, k, m)``.
    :return: True for an omega dihedral."""
    names = [topology.names[b] for b in dihedral]
    _, j, k, _ = dihedral
    across = topology.resids[j] != topology.resids[k]
    return across and names in (["CA", "C", "N", "CA"], ["CA", "N", "C", "CA"])


def prior_neighbor_lists(topology) -> dict[str, dict]:
    """Neighbour lists of every prior term, keyed by the prior's name.

    Omega gets its own term, so it can be fit at a lower Fourier degree than the other dihedrals.

    :param topology: The mlcg CG topology from :func:`cg_topology`.
    :return: Name -> mlcg neighbour-list dict, for the terms that have interactions."""
    from mlcg.neighbor_list.neighbor_list import make_neighbor_list
    from mlcg.nn import Dihedral, HarmonicAngles, HarmonicBonds, Repulsion

    dihedrals = torch.tensor(topology.dihedrals, dtype=torch.long).reshape(4, -1)
    omega = torch.tensor([is_omega(topology, tuple(d)) for d in dihedrals.t().tolist()])
    neighbor_lists = {
        HarmonicBonds.name: topology.neighbor_list("bonds"),
        HarmonicAngles.name: topology.neighbor_list("angles"),
        Dihedral.name: make_neighbor_list(
            tag=Dihedral.name, order=4, index_mapping=dihedrals[:, ~omega]
        ),
        "omega": make_neighbor_list(tag="omega", order=4, index_mapping=dihedrals[:, omega]),
        Repulsion.name: make_neighbor_list(
            tag=Repulsion.name, order=2, index_mapping=repulsion_pairs(topology)
        ),
    }
    return {k: v for k, v in neighbor_lists.items() if v["index_mapping"].shape[1] > 0}


def protein_neighbor_lists(aa_topology: md.Topology, indices: np.ndarray) -> dict[str, dict]:
    """Neighbour lists of mlcg's transferable protein model on N CA CB C O beads.

    The terms, names and atom orders of the transferable checkpoint's ``configurations.pt``:
    backbone-tree bonds and angles split into N-terminal, bulk and C-terminal terms, per-residue
    ``<RES>_phi`` and ``<RES>_psi``, ``pro_omega`` / ``non_pro_omega`` (by the residue after the
    peptide bond), the impropers ``gamma_1`` (N CB C CA, the CA chirality) and ``gamma_2``
    (CA O N+ C, the carbonyl plane), and ``non_bonded`` for pairs five or more bonds apart. The
    proline ring is not closed: N-CB is not a bond here, unlike in :func:`cg_bonds`.

    :param aa_topology: All-atom topology.
    :param indices: All-atom index of each bead, in bead order.
    :return: Name -> mlcg neighbour-list dict, for the terms that have interactions."""
    from mlcg.neighbor_list.neighbor_list import make_neighbor_list

    bead, resname = {}, {}
    for b, index in enumerate(indices):
        atom = aa_topology.atom(int(index))
        bead[(atom.residue.index, atom.name)] = b
        resname[atom.residue.index] = atom.residue.name
    residues = sorted(resname)
    first, last = residues[0], residues[-1]

    bonds = []
    for r in residues:
        bonds += [(bead[(r, "N")], bead[(r, "CA")]), (bead[(r, "CA")], bead[(r, "C")])]
        bonds.append((bead[(r, "C")], bead[(r, "O")]))
        if (r, "CB") in bead:
            bonds.append((bead[(r, "CA")], bead[(r, "CB")]))
    for r, nxt in zip(residues, residues[1:]):
        bonds.append((bead[(r, "C")], bead[(nxt, "N")]))

    neighbours: dict[int, set[int]] = {b: set() for b in range(len(indices))}
    for i, j in bonds:
        neighbours[i].add(j)
        neighbours[j].add(i)
    angles = [(i, j, k) for j in neighbours for i in neighbours[j] for k in neighbours[j] if i < k]

    residue_of = {b: r for (r, _), b in bead.items()}
    terms: dict[str, list[tuple[int, ...]]] = {}
    for kind, features in (("bonds", bonds), ("angles", angles)):
        for feature in features:
            touched = {residue_of[b] for b in feature}
            end = "n_term" if first in touched else "c_term" if last in touched else "bulk"
            terms.setdefault(f"{end}_{kind}", []).append(feature)

    for prev, r in zip(residues, residues[1:]):
        terms.setdefault(f"{resname[r]}_phi", []).append(
            (bead[(prev, "C")], bead[(r, "N")], bead[(r, "CA")], bead[(r, "C")])
        )
        terms.setdefault(f"{resname[prev]}_psi", []).append(
            (bead[(prev, "N")], bead[(prev, "CA")], bead[(prev, "C")], bead[(r, "N")])
        )
        omega = "pro_omega" if resname[r] == "PRO" else "non_pro_omega"
        terms.setdefault(omega, []).append(
            (bead[(prev, "CA")], bead[(prev, "C")], bead[(r, "N")], bead[(r, "CA")])
        )
        terms.setdefault("gamma_2", []).append(
            (bead[(prev, "CA")], bead[(prev, "O")], bead[(r, "N")], bead[(prev, "C")])
        )
    for r in residues:
        if (r, "CB") in bead:
            terms.setdefault("gamma_1", []).append(
                (bead[(r, "N")], bead[(r, "CB")], bead[(r, "C")], bead[(r, "CA")])
            )

    neighbor_lists = {
        name: make_neighbor_list(
            tag=name, order=len(features[0]), index_mapping=torch.tensor(features).t()
        )
        for name, features in terms.items()
    }
    neighbor_lists["non_bonded"] = make_neighbor_list(
        tag="non_bonded", order=2, index_mapping=separated_pairs(len(indices), bonds, 5)
    )
    return neighbor_lists


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


# Omega is a ~6 deg wide peak at 180 deg; the 6-degree fit gave it |k| ~ 3e5 on ala5.
GRAPH_DIHEDRAL_DEGREES = {"dihedrals": 3, "omega": 1}


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
    from mlcg.geometry.statistics import compute_statistics, fit_baseline_models
    from mlcg.nn import Dihedral, GradientsOut, HarmonicAngles, HarmonicBonds, Repulsion

    data = collated(coords, types, masses, neighbor_lists)
    beta = 1.0 / (KB_KCAL * temperature)
    classes = [HarmonicBonds, HarmonicAngles, Repulsion]
    priors, _ = fit_baseline_models(data, beta, [c for c in classes if c.name in neighbor_lists])

    for name, n_degs in GRAPH_DIHEDRAL_DEGREES.items():
        if name in neighbor_lists:
            statistics = compute_statistics(
                data, name, beta, TargetPrior=Dihedral, target_fit_kwargs={"n_degs": n_degs}
            )
            priors[name] = Dihedral(statistics, n_degs=n_degs, name=name)

    return torch.nn.ModuleDict({k: GradientsOut(v, targets="forces") for k, v in priors.items()})


# Fourier degrees of the transferable checkpoint's dihedral terms; every other phi / psi has 3.
# Stiff terms get few degrees: a 4-11 deg wide peak fills too few histogram bins for more.
PROTEIN_DIHEDRAL_DEGREES = {
    "PRO_phi": 1,
    "pro_omega": 2,
    "non_pro_omega": 1,
    "gamma_1": 1,
    "gamma_2": 1,
}


def fit_protein_priors(
    coords: np.ndarray, types: Tensor, masses: Tensor, neighbor_lists: dict, temperature: float
) -> torch.nn.ModuleDict:
    """Boltzmann-invert the terms of :func:`protein_neighbor_lists` with mlcg's protein classes.

    :param coords: ``(F, N, 3)`` CG coordinates in Angstrom.
    :param types: ``(N,)`` bead types.
    :param masses: ``(N,)`` bead masses.
    :param neighbor_lists: Prior neighbour lists from :func:`protein_neighbor_lists`.
    :param temperature: Temperature of the source ensemble in K.
    :return: Prior name -> ``GradientsOut``-wrapped prior."""
    from mlcg.geometry.statistics import compute_statistics
    from mlcg.nn import Dihedral, GradientsOut, Repulsion
    from mlcg.nn.prior.harmonic import GeneralAngles, GeneralBonds

    data = collated(coords, types, masses, neighbor_lists)
    beta = 1.0 / (KB_KCAL * temperature)

    priors = torch.nn.ModuleDict()
    for name in neighbor_lists:
        match name:
            case "non_bonded":
                prior = Repulsion(compute_statistics(data, name, beta, TargetPrior=Repulsion))
                prior.name = name
            case _ if name.endswith("_bonds"):
                statistics = compute_statistics(data, name, beta, TargetPrior=GeneralBonds)
                prior = GeneralBonds(statistics, name)
            case _ if name.endswith("_angles"):
                statistics = compute_statistics(data, name, beta, TargetPrior=GeneralAngles)
                prior = GeneralAngles(statistics, name)
            case _ if name.endswith(("_phi", "_psi", "_omega")) or name.startswith("gamma_"):
                n_degs = PROTEIN_DIHEDRAL_DEGREES.get(name, 3)
                statistics = compute_statistics(
                    data, name, beta, TargetPrior=Dihedral, target_fit_kwargs={"n_degs": n_degs}
                )
                prior = Dihedral(statistics, n_degs=n_degs, name=name)
            case _:
                raise ValueError(f"no protein prior class for term {name!r}.")
        priors[name] = GradientsOut(prior, targets="forces")

    return priors


def collated(coords: np.ndarray, types: Tensor, masses: Tensor, neighbor_lists: dict):
    """Collate CG frames into one batched mlcg ``AtomicData``, as the prior fits expect.

    :param coords: ``(F, N, 3)`` CG coordinates in Angstrom.
    :param types: ``(N,)`` bead types.
    :param masses: ``(N,)`` bead masses.
    :param neighbor_lists: Prior neighbour lists.
    :return: The collated ``AtomicData``."""
    from torch_geometric.data.collate import collate

    data_list = atomic_data(coords, types, masses, neighbor_lists)
    data, _, _ = collate(
        data_list[0].__class__, data_list=data_list, increment=True, add_batch=True
    )
    return data


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
    scheme: Literal["graph", "protein"] = "graph",
    chunk: int = 20000,
) -> dict[str, object]:
    """Fit the priors of a dataset and write ``priors.pt``, the h5 and ``configurations.pt``.

    :param dataset_dir: Output directory of :mod:`.cgschnet_dataset` (no prior subtracted).
    :param fit_frames: Frames, evenly strided, the priors are fit on.
    :param n_blocks: Contiguous time blocks each run is cut into.
    :param n_configurations: Evenly spaced frames written as starting configurations.
    :param scheme: ``"graph"`` for terms read off the all-atom bond graph (any bead set, e.g.
        ala2's ``AL_CG_MAP``), ``"protein"`` for the transferable model's terms on N CA CB C O.
    :param chunk: Frames per prior-force forward; lower it when a large molecule runs out of memory.
    :return: Summary of what was written.
    :raises ValueError: If the dataset already has delta forces, or on an unknown scheme."""
    meta = json.loads((dataset_dir / "dataset.json").read_text())
    if meta["delta_forces"]:
        raise ValueError(f"{dataset_dir} already has delta forces; build it without a prior.")

    aa_topology = md.load_topology(meta["aa_topology"])
    indices = np.load(dataset_dir / "bead_indices.npy")
    coords = np.load(dataset_dir / "cg_coords.npy")
    forces = np.load(dataset_dir / "cg_forces.npy")

    match scheme:
        case "graph":
            neighbor_lists = prior_neighbor_lists(cg_topology(aa_topology, indices))
            fit = fit_priors
        case "protein":
            neighbor_lists = protein_neighbor_lists(aa_topology, indices)
            fit = fit_protein_priors
        case _:
            raise ValueError(f"unknown prior scheme {scheme!r}; expected 'graph' or 'protein'.")
    types = bead_types(aa_topology, indices)
    masses = torch.tensor([aa_topology.atom(int(i)).element.mass for i in indices])
    for name, nl in neighbor_lists.items():
        logger.info(f"{name}: {nl['index_mapping'].t().tolist()}")

    stride = max(1, len(coords) // fit_frames)
    priors = fit(coords[::stride], types, masses, neighbor_lists, meta["temperature"])
    torch.save(priors, dataset_dir / "priors.pt")

    delta = forces - prior_forces(priors, coords, types, masses, neighbor_lists, chunk)
    explained = 1 - (delta**2).mean() / (forces**2).mean()
    logger.info(
        f"delta forces: RMS {np.sqrt((delta**2).mean()):.2f} vs total "
        f"{np.sqrt((forces**2).mean()):.2f} kcal/mol/A (prior explains {100 * explained:.1f}%)"
    )
    np.save(dataset_dir / "cg_delta_forces.npy", delta)

    # blocks are cut within each run, so none straddles a run boundary.
    lengths = np.concatenate(
        [
            np.diff(np.linspace(0, run["n_frames"], n_blocks + 1).astype(np.int64))
            for run in meta["runs"]
        ]
    )
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
        "scheme": scheme,
        "prior_explained_force_variance": float(explained),
        "segments": names,
        "segment_lengths": lengths.tolist(),
        "embedding_size": int(types.max()) + 1,
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
    parser.add_argument(
        "--scheme",
        default="graph",
        choices=["graph", "protein"],
        help="Prior terms: off the all-atom bond graph, or the transferable protein model's.",
    )
    parser.add_argument(
        "--chunk", type=int, default=20000, help="Frames per prior-force forward pass."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_priors(
        args.dataset,
        args.fit_frames,
        args.n_blocks,
        args.n_configurations,
        args.scheme,
        args.chunk,
    )


if __name__ == "__main__":
    main()
