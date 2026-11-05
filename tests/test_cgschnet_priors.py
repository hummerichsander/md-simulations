import json
from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")
torch = pytest.importorskip("torch")
pytest.importorskip("mlcg")

from md_simulations.prep.cgschnet_dataset import bead_indices
from md_simulations.prep.cgschnet_priors import (
    bead_types,
    build_priors,
    cg_bonds,
    cg_topology,
    fit_priors,
    fit_protein_priors,
    prior_neighbor_lists,
    protein_neighbor_lists,
)

# AL_CG_MAP on the heavy atoms of _alanine_dipeptide(): ACE C, ALA N CA CB C, NME N.
BEADS = np.array([1, 3, 4, 5, 6, 8])


def _alanine_dipeptide() -> "md.Topology":
    """Heavy-atom ACE-ALA-NME with bonds, in AMBER order.

    :return: An mdtraj ``Topology`` of 10 atoms."""
    topology = md.Topology()
    chain = topology.add_chain()
    atoms = {}
    layout = (
        ("ACE", ["CH3", "C", "O"]),
        ("ALA", ["N", "CA", "CB", "C", "O"]),
        ("NME", ["N", "C"]),
    )
    for resname, names in layout:
        residue = topology.add_residue(resname, chain)
        for name in names:
            element = {"N": md.element.nitrogen, "O": md.element.oxygen}.get(
                name, md.element.carbon
            )
            atoms[(resname, name)] = topology.add_atom(name, element, residue)
    bonds = [
        (("ACE", "CH3"), ("ACE", "C")),
        (("ACE", "C"), ("ACE", "O")),
        (("ACE", "C"), ("ALA", "N")),
        (("ALA", "N"), ("ALA", "CA")),
        (("ALA", "CA"), ("ALA", "CB")),
        (("ALA", "CA"), ("ALA", "C")),
        (("ALA", "C"), ("ALA", "O")),
        (("ALA", "C"), ("NME", "N")),
        (("NME", "N"), ("NME", "C")),
    ]
    for a, b in bonds:
        topology.add_bond(atoms[a], atoms[b])
    return topology


def test_bonds_follow_the_branch() -> None:
    """CB hangs off CA, and C bonds to CA rather than to CB, its predecessor in bead order.

    :return: None."""
    assert cg_bonds(_alanine_dipeptide(), BEADS) == [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)]


def test_bonds_route_through_non_bead_atoms_only() -> None:
    """Dropping CA bonds N to CB and C through it, but never across a bead.

    :return: None."""
    beads = np.array([1, 3, 5, 6, 8])
    assert cg_bonds(_alanine_dipeptide(), beads) == [(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)]


def test_ala2_interaction_lists() -> None:
    """Angles, the four dihedrals (phi, psi and their CB twins) and one 1-5 repulsion pair.

    :return: None."""
    lists = prior_neighbor_lists(cg_topology(_alanine_dipeptide(), BEADS))
    as_sets = {k: {tuple(c) for c in v["index_mapping"].t().tolist()} for k, v in lists.items()}

    assert as_sets["angles"] == {(0, 1, 2), (1, 2, 3), (1, 2, 4), (3, 2, 4), (2, 4, 5)}
    dihedrals = {d if d[1] < d[2] else d[::-1] for d in as_sets["dihedrals"]}
    assert dihedrals == {(0, 1, 2, 3), (0, 1, 2, 4), (1, 2, 4, 5), (3, 2, 4, 5)}
    assert as_sets["repulsion"] == {(0, 5)}


def _polyalanine(n: int) -> tuple["md.Topology", np.ndarray]:
    """Heavy-atom ACE-(ALA)n-NME with bonds, and its beads: ACE C, N CA CB C per Ala, NME N.

    :param n: Number of Ala residues.
    :return: The topology and the bead indices, in atom order."""
    topology = md.Topology()
    chain = topology.add_chain()
    layout = (
        [("ACE", ["CH3", "C", "O"], [("CH3", "C"), ("C", "O")])]
        + [
            (
                "ALA",
                ["N", "CA", "CB", "C", "O"],
                [("N", "CA"), ("CA", "CB"), ("CA", "C"), ("C", "O")],
            )
        ]
        * n
        + [("NME", ["N", "C"], [("N", "C")])]
    )
    previous = None
    for resname, names, bonds in layout:
        residue = topology.add_residue(resname, chain)
        atoms = {name: topology.add_atom(name, md.element.carbon, residue) for name in names}
        if previous is not None:
            topology.add_bond(previous, atoms["N"])
        for a, b in bonds:
            topology.add_bond(atoms[a], atoms[b])
        previous = atoms["C"]

    beads = [
        atom.index
        for atom in topology.atoms
        if (atom.residue.name, atom.name) in {("ACE", "C"), ("NME", "N")}
        or (atom.residue.name == "ALA" and atom.name in ("N", "CA", "CB", "C"))
    ]
    return topology, np.array(beads)


def test_omega_is_its_own_term() -> None:
    """Each peptide bond between two Ala gives one CA-C-N-CA omega, kept out of the dihedrals.

    :return: None."""
    topology, beads = _polyalanine(3)
    lists = _as_sets(prior_neighbor_lists(cg_topology(topology, beads)))

    # ACE C 0, then N CA CB C at 1-4, 5-8, 9-12, NME N 13.
    assert {min(d, d[::-1]) for d in lists["omega"]} == {(2, 4, 5, 6), (6, 8, 9, 10)}
    assert not lists["omega"] & lists["dihedrals"]
    assert len(lists["dihedrals"]) == 4 * 3


def test_graph_fit_is_finite_with_a_stiff_omega() -> None:
    """A planar peptide bond, jittered by 0.02 A, fits to bounded forces.

    :return: None."""
    topology, beads = _polyalanine(3)
    rng = np.random.default_rng(0)
    reference = 1.5 * np.cumsum(rng.standard_normal((len(beads), 3)), axis=0)
    coords = (reference + 0.02 * rng.standard_normal((2000, len(beads), 3))).astype(np.float32)
    cg = cg_topology(topology, beads)
    lists = prior_neighbor_lists(cg)
    types = bead_types(topology, beads)
    masses = torch.full((len(beads),), 12.0)

    priors = fit_priors(coords, types, masses, lists, 330.0)

    from md_simulations.prep.cgschnet_priors import prior_forces

    forces = prior_forces(priors, coords, types, masses, lists)
    assert set(priors) == {"bonds", "angles", "dihedrals", "omega", "repulsion"}
    assert np.isfinite(forces).all()
    assert np.abs(forces).max() < 1e4


def _build_toy_dataset(tmp_path, run_lengths: list[int], n_blocks: int) -> dict:
    """Write a jittered ala2 dataset pooling runs of the given lengths, and fit its priors.

    :param tmp_path: pytest's temporary directory.
    :param run_lengths: Frames contributed by each pooled run.
    :param n_blocks: Time blocks each run is cut into.
    :return: The summary :func:`build_priors` returns."""
    topology = _alanine_dipeptide()
    md.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology).save_pdb(str(tmp_path / "aa.pdb"))

    rng = np.random.default_rng(0)
    reference = np.array(
        [
            [0.0, 0, 0],
            [1.33, 0, 0],
            [1.9, 1.35, 0],
            [1.4, 2.1, 1.2],
            [3.4, 1.4, 0.1],
            [4.0, 2.5, 0.6],
        ]
    )
    n_frames = sum(run_lengths)
    coords = (reference + 0.05 * rng.standard_normal((n_frames, 6, 3))).astype(np.float32)
    forces = rng.standard_normal((n_frames, 6, 3)).astype(np.float32)
    np.save(tmp_path / "cg_coords.npy", coords)
    np.save(tmp_path / "cg_forces.npy", forces)
    np.save(tmp_path / "bead_indices.npy", BEADS)
    meta = {
        "runs": [{"n_frames": n} for n in run_lengths],
        "delta_forces": False,
        "aa_topology": str(tmp_path / "aa.pdb"),
        "temperature": 330.0,
        "name": "ala2",
    }
    (tmp_path / "dataset.json").write_text(json.dumps(meta))

    return build_priors(tmp_path, fit_frames=n_frames, n_blocks=n_blocks, n_configurations=3)


def test_build_priors_writes_consistent_outputs(tmp_path) -> None:
    """Delta forces are total minus prior, the h5 blocks cover every frame once.

    :param tmp_path: pytest's temporary directory.
    :return: None."""
    h5py = pytest.importorskip("h5py")
    summary = _build_toy_dataset(tmp_path, [400], n_blocks=4)
    forces = np.load(tmp_path / "cg_forces.npy")

    delta = np.load(tmp_path / "cg_delta_forces.npy")
    assert not np.allclose(delta, forces)
    with h5py.File(tmp_path / "ala2_bysegment.h5") as handle:
        stacked = np.concatenate(
            [handle["ala2"][n]["cg_delta_forces"][:] for n in summary["segments"]]
        )
    np.testing.assert_allclose(stacked, delta, rtol=1e-6)
    assert summary["segment_lengths"] == [100, 100, 100, 100]

    configurations = torch.load(tmp_path / "configurations.pt", weights_only=False)
    assert len(configurations) == 3
    assert set(configurations[0].neighbor_list) == set(
        torch.load(tmp_path / "priors.pt", weights_only=False)
    )


def test_build_priors_blocks_each_pooled_run(tmp_path) -> None:
    """Pooled runs are cut into blocks within each run, so no block straddles a boundary.

    :param tmp_path: pytest's temporary directory.
    :return: None."""
    summary = _build_toy_dataset(tmp_path, [250, 150], n_blocks=2)
    assert summary["segment_lengths"] == [125, 125, 75, 75]


def test_build_priors_rejects_delta_forces(tmp_path) -> None:
    """A dataset with the priors already subtracted cannot be fit again.

    :param tmp_path: pytest's temporary directory.
    :return: None."""
    (tmp_path / "dataset.json").write_text(json.dumps({"runs": [{}], "delta_forces": True}))
    with pytest.raises(ValueError, match="delta forces"):
        build_priors(tmp_path)


def test_ala2_beads_each_get_their_own_type() -> None:
    """The caps' C and N differ from ALA's, so ala2's six beads are six types, in bead order.

    :return: None."""
    assert bead_types(_alanine_dipeptide(), BEADS).tolist() == [0, 1, 2, 3, 4, 5]


def test_protein_types_share_the_backbone() -> None:
    """N, CB, C and O are one type each across residues; CA is typed by its residue.

    :return: None."""
    topology = md.Topology()
    chain = topology.add_chain()
    for resname in ("ALA", "GLY", "ALA"):
        residue = topology.add_residue(resname, chain)
        names = ["N", "CA", "C", "O"] if resname == "GLY" else ["N", "CA", "CB", "C", "O"]
        for name in names:
            topology.add_atom(name, md.element.carbon, residue)

    types = bead_types(topology, np.arange(topology.n_atoms)).tolist()
    ala, gly, ala2 = types[:5], types[5:9], types[9:]
    assert ala == ala2
    assert gly[1] != ala[1]
    assert [gly[0], gly[2], gly[3]] == [ala[0], ala[3], ala[4]]
    assert max(types) + 1 == 6


TRPCAGE = Path(__file__).resolve().parents[1] / "data" / "trp-cage"


def _tripeptide() -> "md.Topology":
    """GLY-PRO-ALA on its N CA CB C O beads, in mlcg order.

    :return: An mdtraj ``Topology`` of 13 beads."""
    topology = md.Topology()
    chain = topology.add_chain()
    for resname in ("GLY", "PRO", "ALA"):
        residue = topology.add_residue(resname, chain)
        names = ["N", "CA", "C", "O"] if resname == "GLY" else ["N", "CA", "CB", "C", "O"]
        for name in names:
            topology.add_atom(name, md.element.carbon, residue)
    return topology


def _as_sets(neighbor_lists: dict) -> dict[str, set]:
    """Neighbour lists as sets of index tuples, keyed by term name.

    :param neighbor_lists: mlcg neighbour-list dicts.
    :return: Term name -> set of tuples."""
    return {
        k: {tuple(c) for c in v["index_mapping"].t().tolist()} for k, v in neighbor_lists.items()
    }


def test_protein_terms_on_a_tripeptide() -> None:
    """Per-residue phi/psi, a proline omega, and no bond closing the proline ring.

    :return: None."""
    topology = _tripeptide()
    lists = _as_sets(protein_neighbor_lists(topology, np.arange(topology.n_atoms)))

    assert set(lists) == {
        "n_term_bonds",
        "bulk_bonds",
        "c_term_bonds",
        "n_term_angles",
        "bulk_angles",
        "c_term_angles",
        "PRO_phi",
        "ALA_phi",
        "GLY_psi",
        "PRO_psi",
        "pro_omega",
        "non_pro_omega",
        "gamma_1",
        "gamma_2",
        "non_bonded",
    }
    # GLY 0-3, PRO 4-8 (N CA CB C O), ALA 9-13.
    assert lists["pro_omega"] == {(1, 2, 4, 5)}
    assert lists["non_pro_omega"] == {(5, 7, 9, 10)}
    assert lists["gamma_1"] == {(4, 6, 7, 5), (9, 11, 12, 10)}
    bonds = lists["n_term_bonds"] | lists["bulk_bonds"] | lists["c_term_bonds"]
    assert len(bonds) == topology.n_atoms - 1
    assert (4, 6) not in bonds


def test_protein_fit_is_finite() -> None:
    """Every protein term fits to finite, bounded forces on jittered frames.

    :return: None."""
    topology = _tripeptide()
    indices = np.arange(topology.n_atoms)
    rng = np.random.default_rng(0)
    reference = 1.5 * np.cumsum(rng.standard_normal((topology.n_atoms, 3)), axis=0)
    coords = (reference + 0.1 * rng.standard_normal((500, topology.n_atoms, 3))).astype(np.float32)
    lists = protein_neighbor_lists(topology, indices)
    types = bead_types(topology, indices)
    masses = torch.full((topology.n_atoms,), 12.0)

    priors = fit_protein_priors(coords, types, masses, lists, 330.0)

    from md_simulations.prep.cgschnet_priors import prior_forces

    forces = prior_forces(priors, coords, types, masses, lists)
    assert np.isfinite(forces).all()
    assert np.abs(forces).max() < 1e4


@pytest.mark.skipif(
    not (TRPCAGE / "fm_330K" / "dataset.json").exists(), reason="needs the trp-cage dataset"
)
def test_protein_terms_match_the_transferable_model() -> None:
    """On trp-cage the protein lists equal the transferable ``configurations.pt``'s, term by term.

    :return: None."""
    meta = json.loads((TRPCAGE / "fm_330K" / "dataset.json").read_text())
    root = TRPCAGE.parents[1]
    aa_topology = md.load_topology(str(root / meta["aa_topology"]))
    indices = bead_indices(aa_topology, md.load_topology(str(root / meta["bead_pdb"])))

    ours = _as_sets(protein_neighbor_lists(aa_topology, indices))
    reference = torch.load(TRPCAGE / "configurations.pt", weights_only=False)[0].neighbor_list
    theirs = {k: v for k, v in _as_sets(reference).items() if v}

    def canonical(term: set) -> set:
        """Undirected form of each feature, as mlcg's symmetrised types treat them.

        :param term: Set of index tuples.
        :return: Set with each tuple replaced by min(tuple, reversed tuple)."""
        return {min(f, f[::-1]) for f in term}

    assert set(ours) == set(theirs)
    for name in theirs:
        assert canonical(ours[name]) == canonical(theirs[name]), name
