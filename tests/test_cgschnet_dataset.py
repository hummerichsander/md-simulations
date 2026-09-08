import numpy as np
import pytest

mdtraj = pytest.importorskip("mdtraj")

from md_simulations.prep.cgschnet_dataset import (
    NM_TO_ANGSTROM,
    KJNM_TO_KCALA,
    _summary_constraints,
    _summary_temperature,
    bead_indices,
    cg_native,
    cg_trajectory,
    count_lines,
    frame_selection,
    project_run,
    read_all_atom,
)

BEAD_NAMES = ["N", "CA", "CB", "C", "O"]


def _topologies(n_res: int = 3):
    """An all-atom topology in AMBER order and a CG one in mlcg order.

    AMBER emits ``N, CA, C, O, CB`` per residue while mlcg wants ``N, CA, CB, C, O``, so
    the two orders genuinely differ and the mapping has to be by name, not position.
    """
    aa = mdtraj.Topology()
    chain = aa.add_chain()
    for i in range(n_res):
        res = aa.add_residue("ALA", chain, resSeq=i + 1)
        for name in ["N", "CA", "C", "O", "CB"]:
            element = mdtraj.element.nitrogen if name == "N" else mdtraj.element.carbon
            aa.add_atom(name, mdtraj.element.oxygen if name == "O" else element, res)

    cg = mdtraj.Topology()
    chain = cg.add_chain()
    for i in range(n_res):
        res = cg.add_residue("ALA", chain, resSeq=i + 1)
        for name in BEAD_NAMES:
            element = mdtraj.element.nitrogen if name == "N" else mdtraj.element.carbon
            cg.add_atom(name, mdtraj.element.oxygen if name == "O" else element, res)
    return aa, cg


def _fake_run(tmp_path, n_frames: int = 40, n_res: int = 3):
    """Write a run directory whose frame *k* is uniquely identifiable by its values."""
    aa, _ = _topologies(n_res)
    n_atoms = aa.n_atoms
    run = tmp_path / "run"
    run.mkdir()

    # Frame k has every coordinate equal to k (nm) and every force to -k (kJ/mol/nm),
    # so any misalignment between the two streams is immediately visible.
    xyz = np.repeat(np.arange(n_frames, dtype=np.float32), n_atoms * 3)
    xyz = xyz.reshape(n_frames, n_atoms, 3)
    traj = mdtraj.Trajectory(xyz, aa)
    traj.save_dcd(str(run / "trajectory.dcd"))
    traj[0].save_pdb(str(run / "initial_structure.pdb"))

    with (run / "forces.txt").open("w") as handle:
        for k in range(n_frames):
            handle.write(" ".join([f"{-float(k):g}"] * (n_atoms * 3)) + "\n")

    (run / "simulation_summary.txt").write_text(
        "Particles:        15\nConstraints:      1,234\nTemperature: 330.0 K\n"
    )
    return run, aa, n_atoms


def test_bead_indices_follows_cg_order_not_all_atom_order():
    aa, cg = _topologies(n_res=2)
    indices = bead_indices(aa, cg)
    # CG slot order is N, CA, CB, C, O; AMBER stores N, CA, C, O, CB.
    assert indices.tolist() == [0, 1, 4, 2, 3, 5, 6, 9, 7, 8]


def test_bead_indices_raises_on_missing_bead():
    aa, cg = _topologies(n_res=2)
    extra = mdtraj.Topology()
    chain = extra.add_chain()
    res = extra.add_residue("ALA", chain, resSeq=1)
    extra.add_atom("ZZ", mdtraj.element.carbon, res)
    with pytest.raises(ValueError, match="matched"):
        bead_indices(aa, extra)


def test_frame_selection_stride_start_stop():
    assert frame_selection(10, stride=3).tolist() == [0, 3, 6, 9]
    assert frame_selection(10, stride=2, start=4).tolist() == [4, 6, 8]
    assert frame_selection(10, stride=1, start=2, stop=5).tolist() == [2, 3, 4]
    assert frame_selection(10, stride=1, stop=100).tolist() == list(range(10))


def test_frame_selection_explicit_indices(tmp_path):
    path = tmp_path / "idx.npy"
    np.save(path, np.array([7, 1, 3], dtype=np.int64))
    assert frame_selection(10, frame_indices=path).tolist() == [1, 3, 7]


def test_frame_selection_rejects_conflicting_options(tmp_path):
    path = tmp_path / "idx.npy"
    np.save(path, np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="cannot be combined"):
        frame_selection(10, stride=2, frame_indices=path)


def test_frame_selection_rejects_bad_indices(tmp_path):
    path = tmp_path / "idx.npy"
    np.save(path, np.array([1, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="duplicate"):
        frame_selection(10, frame_indices=path)

    np.save(path, np.array([1, 99], dtype=np.int64))
    with pytest.raises(ValueError, match="outside"):
        frame_selection(10, frame_indices=path)

    np.save(path, np.array([1.5, 2.5]))
    with pytest.raises(ValueError, match="integers"):
        frame_selection(10, frame_indices=path)


def test_frame_selection_rejects_empty_and_invalid():
    with pytest.raises(ValueError, match="empty"):
        frame_selection(10, stride=1, start=20)
    with pytest.raises(ValueError, match="at least 1"):
        frame_selection(10, stride=0)
    with pytest.raises(ValueError, match="non-negative"):
        frame_selection(10, start=-1)


def test_count_lines(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("a\nb\nc\n")
    assert count_lines(path) == 3


@pytest.mark.parametrize("chunk", [3, 7, 100])
@pytest.mark.parametrize("stride", [1, 3, 5])
def test_coordinates_and_forces_stay_paired_under_stride(tmp_path, chunk, stride):
    """The regression test for mdtraj's iterload stride semantics.

    ``iterload(chunk=c, stride=s)`` yields ``c``-frame blocks spanning ``c * s`` file
    frames, so striding the trajectory reader while taking every ``s``-th force line
    misaligns the two by a factor of ``s``. Selection must happen after pairing.
    """
    run, aa, n_atoms = _fake_run(tmp_path, n_frames=40)
    selection = frame_selection(40, stride=stride)
    coords, forces = read_all_atom(run, run / "initial_structure.pdb", n_atoms, chunk, selection)

    assert len(coords) == len(selection)
    expected_coord = selection * NM_TO_ANGSTROM
    expected_force = -selection * KJNM_TO_KCALA
    assert np.allclose(coords[:, 0, 0], expected_coord)
    assert np.allclose(forces[:, 0, 0], expected_force)
    # The pairing itself: force must be the negative partner of its own coordinate.
    assert np.allclose(forces[:, 0, 0] / KJNM_TO_KCALA, -coords[:, 0, 0] / NM_TO_ANGSTROM)


def test_explicit_frame_indices_stay_paired(tmp_path):
    run, aa, n_atoms = _fake_run(tmp_path, n_frames=40)
    wanted = np.array([0, 1, 17, 38, 39], dtype=np.int64)
    path = tmp_path / "idx.npy"
    np.save(path, wanted)
    selection = frame_selection(40, frame_indices=path)
    coords, forces = read_all_atom(run, run / "initial_structure.pdb", n_atoms, 7, selection)
    assert np.allclose(coords[:, 0, 0], wanted * NM_TO_ANGSTROM)
    assert np.allclose(forces[:, 0, 0], -wanted * KJNM_TO_KCALA)


def test_project_run_maps_into_cg_bead_order(tmp_path):
    run, aa, n_atoms = _fake_run(tmp_path, n_frames=10)
    _, cg = _topologies(n_res=3)
    indices = bead_indices(aa, cg)
    cg_matrix = np.zeros((len(indices), n_atoms))
    cg_matrix[np.arange(len(indices)), indices] = 1.0

    selection = frame_selection(10, stride=2)
    coords, forces = project_run(
        run, run / "initial_structure.pdb", cg_matrix, cg_matrix, n_atoms, 4, selection
    )
    assert coords.shape == (len(selection), len(indices), 3)
    assert np.allclose(coords[:, 0, 0], selection * NM_TO_ANGSTROM)


def test_cg_trajectory_converts_angstrom_to_nm(tmp_path):
    _, cg = _topologies(n_res=2)
    bead_pdb = tmp_path / "cg.pdb"
    mdtraj.Trajectory(np.zeros((1, cg.n_atoms, 3), dtype=np.float32), cg).save_pdb(str(bead_pdb))

    coords_angstrom = np.full((4, cg.n_atoms, 3), 10.0, dtype=np.float32)
    traj = cg_trajectory(coords_angstrom, bead_pdb)
    assert traj.n_frames == 4
    assert np.allclose(traj.xyz, 1.0)  # 10 Angstrom is 1 nm


def test_cg_native_projects_all_atom_reference(tmp_path):
    aa, cg = _topologies(n_res=2)
    indices = bead_indices(aa, cg)
    bead_pdb = tmp_path / "cg.pdb"
    mdtraj.Trajectory(np.zeros((1, cg.n_atoms, 3), dtype=np.float32), cg).save_pdb(str(bead_pdb))

    xyz = np.arange(aa.n_atoms * 3, dtype=np.float32).reshape(1, aa.n_atoms, 3) / 100.0
    native_pdb = tmp_path / "native.pdb"
    mdtraj.Trajectory(xyz, aa).save_pdb(str(native_pdb))

    projected = cg_native(native_pdb, indices, bead_pdb)
    assert projected.n_frames == 1
    assert projected.n_atoms == cg.n_atoms
    reference = mdtraj.load(str(native_pdb))
    assert np.allclose(projected.xyz[0], reference.xyz[0][indices], atol=1e-3)


def test_summary_constraints_handles_thousands_separator(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "simulation_summary.txt").write_text("Constraints:      1,234\n")
    assert _summary_constraints(run) == 1234

    (run / "simulation_summary.txt").write_text("Constraints:      136\n")
    assert _summary_constraints(run) == 136


def test_summary_temperature_is_parsed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "simulation_summary.txt").write_text("Temperature: 330.0 K\n")
    assert _summary_temperature(run) == 330.0

    (run / "simulation_summary.txt").write_text("nothing here\n")
    assert _summary_temperature(run) is None


def test_write_h5_sets_the_attributes_mlcg_requires(tmp_path):
    h5py = pytest.importorskip("h5py")
    from md_simulations.prep.cgschnet_dataset import write_h5

    coords = np.zeros((7, 5, 3), dtype=np.float32)
    delta = np.ones((7, 5, 3), dtype=np.float32)
    embeds = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    path = tmp_path / "mol.h5"
    write_h5(path, "mol", coords, delta, embeds)

    with h5py.File(path) as handle:
        group = handle["mol/mol"]
        # Both are read unconditionally by mlcg's H5Dataset; without them it raises.
        assert group.attrs["N_frames"] == 7
        assert np.array_equal(group.attrs["cg_embeds"], embeds)
        assert group["cg_coords"].shape == (7, 5, 3)
        assert group["cg_delta_forces"].shape == (7, 5, 3)


def test_write_h5_rejects_mismatched_embeddings(tmp_path):
    pytest.importorskip("h5py")
    from md_simulations.prep.cgschnet_dataset import write_h5

    with pytest.raises(ValueError, match="entries but"):
        write_h5(
            tmp_path / "mol.h5",
            "mol",
            np.zeros((2, 5, 3), dtype=np.float32),
            np.zeros((2, 5, 3), dtype=np.float32),
            np.array([1, 2], dtype=np.int64),
        )


def test_written_h5_loads_through_mlcg(tmp_path):
    """The end-to-end gate: mlcg's own loader must accept the file we write.

    The previous dataset passed every shape and dtype check and still failed here with
    ``KeyError: cg_embeds``, so this is the only assertion that actually establishes the
    file is trainable.
    """
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("mlcg")
    from mlcg.datasets.h5_dataset import MetaSet

    from md_simulations.prep.cgschnet_dataset import write_h5

    n_frames, n_beads = 12, 5
    rng = np.random.default_rng(0)
    coords = rng.normal(0.0, 5.0, (n_frames, n_beads, 3)).astype(np.float32)
    delta = rng.normal(0.0, 1.0, (n_frames, n_beads, 3)).astype(np.float32)
    embeds = np.array([21, 22, 3, 23, 24], dtype=np.int64)

    path = tmp_path / "mol.h5"
    write_h5(path, "mol", coords, delta, embeds)

    with h5py.File(path) as handle:
        metaset = MetaSet.create_from_hdf5_group(handle["mol"], mol_list=["mol"])
        assert len(metaset) == n_frames
        sample = metaset[0]
        assert tuple(sample.pos.shape) == (n_beads, 3)
        assert sample.atom_types.tolist() == embeds.tolist()


def test_embedding_order_uses_an_inverse_scatter():
    """Pins the direction of the bead_permutation application.

    ``perm[j]`` is the caller-bead index at model slot ``j``, so caller order is
    recovered by scattering (``out[perm] = types``). On trp-cage the model and dataset
    topologies are the same file, ``perm`` is the identity, and a gather would look
    identical — so the direction has to be pinned on a permuted case.
    """
    bead_permutation = pytest.importorskip(
        "md_simulations.torch_potentials.cgschnet"
    ).bead_permutation

    model_top = mdtraj.Topology()
    chain = model_top.add_chain()
    res = model_top.add_residue("ALA", chain, resSeq=1)
    for name in ["N", "CA", "CB"]:
        model_top.add_atom(name, mdtraj.element.carbon, res)

    caller_top = mdtraj.Topology()
    chain = caller_top.add_chain()
    res = caller_top.add_residue("ALA", chain, resSeq=1)
    for name in ["CB", "N", "CA"]:  # deliberately permuted
        caller_top.add_atom(name, mdtraj.element.carbon, res)

    perm = np.asarray(bead_permutation(model_top, caller_top))
    assert perm.tolist() == [1, 2, 0]  # model slots N, CA, CB -> caller 1, 2, 0

    model_types = np.array([10, 20, 30], dtype=np.int64)  # types for N, CA, CB
    out = np.empty(3, dtype=np.int64)
    out[perm] = model_types
    assert out.tolist() == [30, 10, 20]  # caller order CB, N, CA
    # A gather would silently give the wrong answer here.
    assert model_types[perm].tolist() != out.tolist()
