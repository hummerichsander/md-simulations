import numpy as np
import pytest

mdtraj = pytest.importorskip("mdtraj")

from md_simulations.analysis.cvs import (
    compute_cvs,
    end_to_end_distance,
    fraction_native_contacts,
    radius_of_gyration,
    rmsd_to_native,
)


def _ca_trajectory(n_res: int = 8, n_frames: int = 50, seed: int = 0):
    """Build a synthetic Cα-only trajectory laid out as an extended chain.

    :param n_res: Number of Cα beads (one per residue).
    :param n_frames: Number of frames.
    :param seed: RNG seed for reproducibility.
    :return: An ``mdtraj.Trajectory`` with one CA per residue."""
    top = mdtraj.Topology()
    chain = top.add_chain()
    for i in range(n_res):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", mdtraj.element.carbon, res)
    rng = np.random.default_rng(seed)
    base = np.zeros((n_res, 3))
    base[:, 0] = np.arange(n_res) * 0.38
    xyz = base[None] + 0.02 * rng.standard_normal((n_frames, n_res, 3))
    return mdtraj.Trajectory(xyz.astype(np.float32), top)


def _compact_ca_trajectory(n_res: int = 8, n_frames: int = 50, seed: int = 1):
    """Build a synthetic Cα-only trajectory collapsed into a tight blob.

    Beads sit within a small box so residues far apart in sequence are still
    close in space, giving a non-empty native-contact set.

    :param n_res: Number of Cα beads.
    :param n_frames: Number of frames.
    :param seed: RNG seed for reproducibility.
    :return: An ``mdtraj.Trajectory`` with one CA per residue."""
    top = mdtraj.Topology()
    chain = top.add_chain()
    for i in range(n_res):
        res = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", mdtraj.element.carbon, res)
    rng = np.random.default_rng(seed)
    base = 0.3 * rng.standard_normal((n_res, 3))
    xyz = base[None] + 0.02 * rng.standard_normal((n_frames, n_res, 3))
    return mdtraj.Trajectory(xyz.astype(np.float32), top)


def test_rg_shape_and_positive():
    """Radius of gyration is one finite positive value per frame."""
    traj = _ca_trajectory()
    rg = radius_of_gyration(traj)
    assert rg.shape == (traj.n_frames,)
    assert np.all(np.isfinite(rg)) and np.all(rg > 0)


def test_end_to_end_matches_chain_span():
    """End-to-end distance tracks the first–last Cα separation."""
    traj = _ca_trajectory(n_res=8)
    ete = end_to_end_distance(traj)
    assert ete.shape == (traj.n_frames,)
    # extended chain of 8 beads at ~0.38 nm spacing -> ~2.66 nm
    assert np.all(np.abs(ete - 0.38 * 7) < 0.3)


def test_rmsd_to_native_zero_at_reference():
    """RMSD of a frame to itself is ~0 and all values are non-negative.

    Uses a compact (3-D) blob: QCP's closed-form RMSD is numerically unstable
    for the near-collinear extended chain."""
    traj = _compact_ca_trajectory()
    rmsd = rmsd_to_native(traj, traj[0])
    assert rmsd.shape == (traj.n_frames,)
    assert rmsd[0] < 1e-5
    assert np.all(rmsd >= 0)


def test_native_contacts_near_one_at_reference():
    """Q of the native frame is ~1 and every frame stays in [0, 1]."""
    traj = _compact_ca_trajectory()
    q = fraction_native_contacts(traj, traj[0], native_cutoff=0.6, min_seq_sep=2)
    assert q.shape == (traj.n_frames,)
    assert q[0] > 0.95
    assert np.all((q >= 0) & (q <= 1))


def test_native_contacts_requires_contacts():
    """An extended chain has no native contacts and is rejected."""
    traj = _ca_trajectory(n_res=8)
    with pytest.raises(ValueError):
        fraction_native_contacts(traj, traj[0], native_cutoff=0.45, min_seq_sep=3)


def test_compute_cvs_returns_named_arrays():
    """compute_cvs returns one array per requested CV, in order."""
    traj = _compact_ca_trajectory()
    cvs = compute_cvs(
        traj, traj[0], ["rg", "end_to_end", "rmsd"], q_native_cutoff=0.6
    )
    assert list(cvs) == ["rg", "end_to_end", "rmsd"]
    for arr in cvs.values():
        assert arr.shape == (traj.n_frames,)


def test_unknown_cv_raises():
    """An unrecognised CV name is rejected."""
    traj = _ca_trajectory()
    with pytest.raises(KeyError):
        compute_cvs(traj, None, ["nope"])


def test_native_required_without_reference_raises():
    """A native-requiring CV without a reference is rejected."""
    traj = _ca_trajectory()
    with pytest.raises(ValueError):
        compute_cvs(traj, None, ["rmsd"])


def test_pairwise_plot_writes_pdf(tmp_path):
    """Plotting three CVs writes a single multi-panel PDF."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    from md_simulations.analysis.cvs import plot_cv_free_energy

    traj = _compact_ca_trajectory(n_frames=200)
    cvs = compute_cvs(
        traj, traj[0], ["rg", "end_to_end", "rmsd"], q_native_cutoff=0.6
    )
    out = tmp_path / "cv_fes.pdf"
    plot_cv_free_energy(cvs, out)
    assert out.exists() and out.stat().st_size > 0
