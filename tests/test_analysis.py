import numpy as np
import pytest

mdtraj = pytest.importorskip("mdtraj")

from md_simulations.analysis.tica import (
    ca_distance_features,
    compute_features,
)


def _ca_trajectory(n_res: int = 8, n_frames: int = 200, seed: int = 0):
    """Build a synthetic Cα-only trajectory as a wandering chain.

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
    # Chain laid along x (~0.38 nm spacing) with small per-frame jitter.
    base = np.zeros((n_res, 3))
    base[:, 0] = np.arange(n_res) * 0.38
    xyz = base[None] + 0.02 * rng.standard_normal((n_frames, n_res, 3))
    return mdtraj.Trajectory(xyz.astype(np.float32), top)


def test_ca_distance_feature_count():
    """Cα-distance featurisation yields the expected pair count and is finite."""
    traj = _ca_trajectory(n_res=8)
    feats = ca_distance_features(traj, exclude_neighbors=2)
    # pairs with j - i > 2 among 8 beads
    expected = sum(1 for i in range(8) for j in range(i + 1, 8) if j - i > 2)
    assert feats.shape == (traj.n_frames, expected)
    assert np.all(np.isfinite(feats)) and np.all(feats > 0)


def test_exclude_neighbors_reduces_features():
    """A smaller exclusion window keeps more Cα pairs."""
    traj = _ca_trajectory(n_res=8)
    few = compute_features(traj, ["ca_distances"], exclude_neighbors=3)
    many = compute_features(traj, ["ca_distances"], exclude_neighbors=1)
    assert many.shape[1] > few.shape[1]


def test_unknown_feature_raises():
    """An unrecognised feature name is rejected."""
    traj = _ca_trajectory()
    with pytest.raises(KeyError):
        compute_features(traj, ["nope"])


def test_save_load_model_roundtrip(tmp_path):
    """A pickled model reloads and reproduces its own .transform."""
    pytest.importorskip("deeptime")
    from md_simulations.analysis.tica import fit_tica, load_tica_model, save_tica_model

    traj = _ca_trajectory(n_res=10, n_frames=500)
    feats = compute_features(traj, ["ca_distances"])
    model = fit_tica(feats, lagtime=5, dim=2, scaling="kinetic_map")
    path = tmp_path / "model.pkl"
    save_tica_model(model, path)
    reloaded = load_tica_model(path)
    np.testing.assert_allclose(
        reloaded.transform(feats)[:, :2], model.transform(feats)[:, :2], atol=1e-8
    )
