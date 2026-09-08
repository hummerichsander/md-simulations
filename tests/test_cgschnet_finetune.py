import numpy as np
import pytest

pytest.importorskip("h5py")
# cgschnet_finetune imports torch at module scope, so the whole file needs it.
torch = pytest.importorskip("torch")

from md_simulations.prep.cgschnet_finetune import write_multi_h5


class _Dummy(torch.nn.Module):
    """A picklable stand-in for a model term.

    Defined at module level because ``torch.save`` pickles by qualified name and cannot
    reach a class declared inside a test function.
    """

    name = "dummy"

    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))

    def forward(self, data):
        return data


def _dataset(n_beads: int = 5, lengths=(4, 6, 5)):
    total = sum(lengths)
    rng = np.random.default_rng(0)
    coords = rng.normal(0.0, 5.0, (total, n_beads, 3)).astype(np.float32)
    delta = rng.normal(0.0, 1.0, (total, n_beads, 3)).astype(np.float32)
    embeds = np.arange(1, n_beads + 1, dtype=np.int64)
    return coords, delta, embeds, np.array(lengths, dtype=np.int64)


def test_write_multi_h5_splits_by_segment(tmp_path):
    import h5py

    coords, delta, embeds, lengths = _dataset()
    path = tmp_path / "multi.h5"
    names = write_multi_h5(path, "trpcage", coords, delta, embeds, lengths)
    assert names == ["seg00", "seg01", "seg02"]

    with h5py.File(path) as handle:
        assert sorted(handle["trpcage"].keys()) == names
        bounds = np.r_[0, np.cumsum(lengths)]
        for name, a, b in zip(names, bounds[:-1], bounds[1:]):
            group = handle[f"trpcage/{name}"]
            # Every molecule group carries the two attributes mlcg reads unconditionally.
            assert group.attrs["N_frames"] == b - a
            assert np.array_equal(group.attrs["cg_embeds"], embeds)
            # Frames land in their own segment, in order.
            assert np.allclose(group["cg_coords"][:], coords[a:b])
            assert np.allclose(group["cg_delta_forces"][:], delta[a:b])


def test_write_multi_h5_honours_custom_names(tmp_path):
    import h5py

    coords, delta, embeds, lengths = _dataset()
    path = tmp_path / "multi.h5"
    write_multi_h5(path, "mol", coords, delta, embeds, lengths, names=["equil", "r00", "r01"])
    with h5py.File(path) as handle:
        assert sorted(handle["mol"].keys()) == ["equil", "r00", "r01"]


def test_write_multi_h5_rejects_inconsistent_lengths(tmp_path):
    coords, delta, embeds, _ = _dataset()
    with pytest.raises(ValueError, match="sum to"):
        write_multi_h5(tmp_path / "m.h5", "mol", coords, delta, embeds, np.array([1, 2]))


def test_write_multi_h5_rejects_name_count_mismatch(tmp_path):
    coords, delta, embeds, lengths = _dataset()
    with pytest.raises(ValueError, match="names for"):
        write_multi_h5(tmp_path / "m.h5", "mol", coords, delta, embeds, lengths, names=["only"])


def test_segment_split_loads_as_separate_molecules_in_mlcg(tmp_path):
    """A held-out segment must be selectable as its own molecule.

    This is what makes the train/val split meaningful: frames 5 ps apart are far inside the
    correlation time, so a random frame split would leave near-duplicates on both sides.
    """
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("mlcg")
    from mlcg.datasets.h5_dataset import MetaSet

    coords, delta, embeds, lengths = _dataset(n_beads=5, lengths=(20, 30, 25))
    path = tmp_path / "multi.h5"
    names = write_multi_h5(path, "trpcage", coords, delta, embeds, lengths, ["equil", "r00", "r01"])

    with h5py.File(path) as handle:
        train = MetaSet.create_from_hdf5_group(handle["trpcage"], mol_list=["r00", "r01"])
        val = MetaSet.create_from_hdf5_group(handle["trpcage"], mol_list=["equil"])
        assert len(train) == 30 + 25
        assert len(val) == 20
        sample = val[0]
        assert tuple(sample.pos.shape) == (5, 3)
        assert sample.atom_types.tolist() == embeds.tolist()
    assert names[0] == "equil"


def test_write_prior_model_drops_only_schnet(tmp_path):
    """The merge step needs the priors alone; SchNet is the one parametric term."""
    pytest.importorskip("mlcg")
    import torch
    from mlcg.nn.gradients import SumOut

    from md_simulations.prep.cgschnet_finetune import write_prior_model

    model = SumOut(torch.nn.ModuleDict({"SchNet": _Dummy(1.0), "Bonds": _Dummy(2.0)}))
    src = tmp_path / "model.pt"
    torch.save(model, src)

    out = tmp_path / "prior.pt"
    assert write_prior_model(src, out) == 1
    reloaded = torch.load(out, map_location="cpu", weights_only=False)
    assert list(reloaded.models.keys()) == ["Bonds"]


def test_write_prior_model_raises_without_schnet(tmp_path):
    pytest.importorskip("mlcg")
    import torch
    from mlcg.nn.gradients import SumOut

    from md_simulations.prep.cgschnet_finetune import write_prior_model

    model = SumOut(torch.nn.ModuleDict({"Bonds": torch.nn.Linear(1, 1)}))
    src = tmp_path / "model.pt"
    torch.save(model, src)
    with pytest.raises(KeyError, match="SchNet"):
        write_prior_model(src, tmp_path / "prior.pt")
