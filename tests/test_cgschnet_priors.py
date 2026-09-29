import json

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")
torch = pytest.importorskip("torch")
pytest.importorskip("mlcg")

from md_simulations.prep.cgschnet_priors import (
    build_priors,
    cg_bonds,
    cg_topology,
    prior_neighbor_lists,
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


def test_build_priors_writes_consistent_outputs(tmp_path) -> None:
    """Delta forces are total minus prior, the h5 blocks cover every frame once.

    :param tmp_path: pytest's temporary directory.
    :return: None."""
    h5py = pytest.importorskip("h5py")
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
    n_frames = 400
    coords = (reference + 0.05 * rng.standard_normal((n_frames, 6, 3))).astype(np.float32)
    forces = rng.standard_normal((n_frames, 6, 3)).astype(np.float32)
    np.save(tmp_path / "cg_coords.npy", coords)
    np.save(tmp_path / "cg_forces.npy", forces)
    np.save(tmp_path / "bead_indices.npy", BEADS)
    meta = {
        "runs": [{"n_frames": n_frames}],
        "delta_forces": False,
        "aa_topology": str(tmp_path / "aa.pdb"),
        "temperature": 330.0,
        "name": "ala2",
    }
    (tmp_path / "dataset.json").write_text(json.dumps(meta))

    summary = build_priors(tmp_path, fit_frames=n_frames, n_blocks=4, n_configurations=3)

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


def test_build_priors_rejects_pooled_runs(tmp_path) -> None:
    """Time blocks of several pooled runs would straddle run boundaries.

    :param tmp_path: pytest's temporary directory.
    :return: None."""
    meta = {"runs": [{}, {}], "delta_forces": False}
    (tmp_path / "dataset.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="one run"):
        build_priors(tmp_path)
