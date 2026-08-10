"""Hermetic tests for the native CGSchNet potential -- no mlcg, no data files.

The fakes here are not conveniences, they are the contract: production code must reach
an mlcg ``AtomicData`` through plain attribute access and ``copy.deepcopy`` only (never
``AtomicData(**kw)``, ``.clone()``, ``.to()`` or ``in``), and must survive a prior that
``scatter``s without ``dim_size``. If either assumption breaks, these tests fail long
before anything touches a 390 MB checkpoint.

The directory ``conftest`` already skips on a missing torch and defaults to float64."""

import copy
import importlib
import subprocess
import sys

import mdtraj as md
import pytest
import torch
from torch import Tensor, nn

from md_simulations.torch_potentials.cgschnet import (
    CGSchNetPotential,
    bead_permutation,
    prune_empty_terms,
    unwrap_energy_models,
)
from md_simulations.torch_potentials.protocol import PotentialLike

MODEL_ORDER = ("N", "CA", "CB", "C", "O")
CALLER_ORDER = ("N", "CA", "C", "O", "CB")


def _scatter_sum(src: Tensor, index: Tensor) -> Tensor:
    """Sum ``src`` into bins given by ``index``, sizing the output from the index.

    Faithfully reproduces mlcg's ``scatter(..., reduce="sum")`` call *without*
    ``dim_size``: an empty index yields a length-0 result, which is exactly the
    failure mode :func:`prune_empty_terms` exists to prevent.

    :param src: Values to accumulate.
    :param index: Destination bin per value.
    :return: The accumulated tensor, of length ``index.max() + 1`` (0 if empty)."""
    size = int(index.max()) + 1 if index.numel() else 0
    return torch.zeros(size, dtype=src.dtype, device=src.device).index_add(0, index, src)


class _FakeAtomicData:
    """Stand-in for ``mlcg.data.AtomicData`` exposing only plain attributes."""

    def __init__(
        self,
        pos: Tensor,
        atom_types: Tensor,
        n_atoms: Tensor,
        neighbor_list: dict,
        masses: Tensor | None = None,
    ) -> None:
        """Store the fields a potential is allowed to touch.

        :param pos: ``(N, 3)`` positions.
        :param atom_types: ``(N,)`` embedding indices.
        :param n_atoms: ``(1,)`` bead count.
        :param neighbor_list: Prior interaction lists.
        :param masses: Integrator-only field, present to prove it gets dropped.
        :return: None."""
        self.pos = pos
        self.atom_types = atom_types
        self.n_atoms = n_atoms
        self.neighbor_list = neighbor_list
        self.masses = torch.ones(pos.shape[0]) if masses is None else masses
        self.out: dict = {}


class _FakeSchNet(nn.Module):
    """A smooth, batch-scattered energy term standing in for mlcg's SchNet."""

    name = "SchNet"

    def __init__(self) -> None:
        """Create the term with one parameter, so dtype casting is observable.

        :return: None."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, data: _FakeAtomicData) -> _FakeAtomicData:
        """Accumulate a per-atom energy into per-frame totals via ``data.batch``.

        :param data: The batched system.
        :return: ``data``, with ``out['SchNet']`` populated."""
        per_atom = self.scale * (data.pos**2).sum(-1)
        data.out[self.name] = {"energy": _scatter_sum(per_atom, data.batch)}
        return data


class _FakeBondPrior(nn.Module):
    """A harmonic pair term reading its interactions from ``data.neighbor_list``."""

    def __init__(self, name: str, k: float = 2.0, r0: float = 1.0) -> None:
        """Bind the term to a neighbour-list key.

        :param name: The ``neighbor_list`` key and ``out`` key this term uses.
        :param k: Force constant.
        :param r0: Equilibrium distance.
        :return: None."""
        super().__init__()
        self.name = name
        self.register_buffer("k", torch.tensor(k))
        self.register_buffer("r0", torch.tensor(r0))

    def forward(self, data: _FakeAtomicData) -> _FakeAtomicData:
        """Sum ``k (r - r0)^2`` over this term's pairs, per frame.

        :param data: The batched system.
        :return: ``data``, with ``out[self.name]`` populated."""
        term = data.neighbor_list[self.name]
        i, j = term["index_mapping"]
        distance = (data.pos[i] - data.pos[j]).pow(2).sum(-1).sqrt()
        energy = self.k * (distance - self.r0) ** 2
        data.out[self.name] = {"energy": _scatter_sum(energy, term["mapping_batch"])}
        return data


class _FakeGradientsOut(nn.Module):
    """Stand-in for ``mlcg.nn.gradients.GradientsOut``: a force head that detaches.

    Reproduces the two behaviours that make the real checkpoint uncallable -- an
    internal ``torch.autograd.grad`` (which raises under ``no_grad``) and
    ``data.pos = data.pos.detach()`` (which severs every later term)."""

    def __init__(self, model: nn.Module) -> None:
        """Wrap an energy module.

        :param model: The bare energy module to wrap.
        :return: None."""
        super().__init__()
        self.model = model
        self.name = model.name

    def forward(self, data: _FakeAtomicData) -> _FakeAtomicData:
        """Run the inner model, take its position gradient, then detach ``pos``.

        :param data: The batched system.
        :return: ``data``, with ``pos`` detached."""
        data.pos.requires_grad_(True)
        data = self.model(data)
        energy = data.out[self.name]["energy"]
        data.out[self.name]["forces"] = -torch.autograd.grad(
            energy.sum(), data.pos, create_graph=self.training
        )[0]
        data.pos = data.pos.detach()
        return data


class _FakeSumOut(nn.Module):
    """Stand-in for ``mlcg.nn.gradients.SumOut``: a ``ModuleDict`` of wrapped terms."""

    def __init__(self, models: dict[str, nn.Module]) -> None:
        """Wrap each term in a force head, as the distributed checkpoint does.

        :param models: Term name -> bare energy module.
        :return: None."""
        super().__init__()
        self.targets = ["energy", "forces"]
        self.models = nn.ModuleDict({k: _FakeGradientsOut(v) for k, v in models.items()})

    def forward(self, data: _FakeAtomicData) -> _FakeAtomicData:
        """Run every wrapped term in sequence.

        :param data: The batched system.
        :return: ``data``, with each term's output populated."""
        for module in self.models.values():
            data = module(data)
        return data


def _topology(residues: list[str], order: tuple[str, ...]) -> "md.Topology":
    """Build a CG bead topology with a fixed within-residue bead order.

    :param residues: Residue names, one per residue.
    :param order: Bead names to emit per residue, in order. ``CB`` is skipped for GLY.
    :return: An mdtraj ``Topology``."""
    topology = md.Topology()
    chain = topology.add_chain()
    for index, name in enumerate(residues):
        residue = topology.add_residue(name, chain, resSeq=index + 1)
        for bead in order:
            if bead == "CB" and name == "GLY":
                continue
            topology.add_atom(bead, md.element.carbon, residue)
    return topology


def _system(n_atoms: int, empty_term: bool = False) -> tuple[dict, _FakeAtomicData]:
    """Build fake energy terms and a matching one-frame template.

    :param n_atoms: Number of beads.
    :param empty_term: Also include a prior with zero interactions.
    :return: ``(models, template)`` ready for :class:`CGSchNetPotential`."""
    pairs = torch.tensor([[i for i in range(n_atoms - 1)], [i + 1 for i in range(n_atoms - 1)]])
    neighbor_list = {
        "bonds": {
            "tag": "bonds",
            "order": 2,
            "index_mapping": pairs,
            "rcut": None,
            "mapping_batch": torch.zeros(pairs.shape[1], dtype=torch.long),
        }
    }
    models: dict[str, nn.Module] = {"SchNet": _FakeSchNet(), "bonds": _FakeBondPrior("bonds")}

    if empty_term:
        neighbor_list["ghost"] = {
            "tag": "ghost",
            "order": 2,
            "index_mapping": torch.zeros((2, 0), dtype=torch.long),
            "rcut": None,
            "mapping_batch": torch.zeros(0, dtype=torch.long),
        }
        models["ghost"] = _FakeBondPrior("ghost")

    template = _FakeAtomicData(
        pos=torch.zeros(n_atoms, 3),
        atom_types=torch.arange(1, n_atoms + 1),
        n_atoms=torch.tensor([n_atoms]),
        neighbor_list=neighbor_list,
    )
    return models, template


@pytest.fixture
def potential() -> CGSchNetPotential:
    """A 6-bead potential with no permutation and float64 terms.

    :return: A ready :class:`CGSchNetPotential`."""
    models, template = _system(6)
    return CGSchNetPotential(
        models,
        template,
        _topology(["ALA", "ALA"], MODEL_ORDER[:3]),
        dtype=torch.float64,
        energy_to_kjmol=1.0,
    )


@pytest.fixture
def coords() -> Tensor:
    """Four well-separated frames of 6 beads, in nm.

    :return: A ``(4, 6, 3)`` tensor."""
    torch.manual_seed(0)
    base = torch.arange(6, dtype=torch.float64).unsqueeze(-1).repeat(1, 3) * 0.15
    return base.unsqueeze(0) + 0.02 * torch.randn(4, 6, 3, dtype=torch.float64)


def test_satisfies_potential_like(potential: CGSchNetPotential) -> None:
    """The potential is structurally interchangeable with ``Potential``.

    :param potential: The fixture potential.
    :return: None."""
    assert isinstance(potential, PotentialLike)
    assert potential.n_atoms == 6
    assert potential.terms == ("SchNet", "bonds")
    assert potential.permutation is None


def test_batched_matches_per_frame(potential: CGSchNetPotential, coords: Tensor) -> None:
    """A batched call equals looping over frames one at a time.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    batched = potential(coords)
    assert batched.shape == (4,)
    per_frame = torch.stack([potential(frame) for frame in coords])
    assert torch.allclose(batched, per_frame)


def test_single_frame_returns_scalar(potential: CGSchNetPotential, coords: Tensor) -> None:
    """A ``(N, 3)`` input yields a 0-dim energy, not a length-1 vector.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    assert potential(coords[0]).shape == torch.Size([])


def test_frames_are_independent(potential: CGSchNetPotential, coords: Tensor) -> None:
    """Perturbing one frame leaves the others' energies untouched.

    This is what catches an off-by-one in the per-frame index offsets: a wrong offset
    still gives plausible numbers, but couples frames together.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    before = potential(coords)
    moved = coords.clone()
    moved[2] += 0.3
    after = potential(moved)

    assert not torch.allclose(before[2], after[2])
    assert torch.equal(before[[0, 1, 3]], after[[0, 1, 3]])


@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 16])
def test_chunking_does_not_change_the_answer(
    potential: CGSchNetPotential, coords: Tensor, chunk: int
) -> None:
    """Every ``max_batch_frames`` gives the same energies.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :param chunk: Frames per forward.
    :return: None."""
    reference = potential(coords)
    potential._max_batch_frames = chunk
    assert torch.allclose(potential(coords), reference)


def test_gradcheck(potential: CGSchNetPotential, coords: Tensor) -> None:
    """The energy's analytic gradient matches finite differences.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    x = coords[:2].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(potential, (x,), atol=1e-6)


def test_double_backward(potential: CGSchNetPotential, coords: Tensor) -> None:
    """Second derivatives exist -- the payoff of not going through the ASE bridge.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    x = coords[:2].clone().requires_grad_(True)
    probe = torch.randn(x.shape, dtype=x.dtype, generator=torch.Generator().manual_seed(0))

    gradient = torch.autograd.grad(potential(x).sum(), x, create_graph=True)[0]
    hessian_vector = torch.autograd.grad((gradient * probe).sum(), x)[0]
    assert torch.isfinite(hessian_vector).all()
    assert hessian_vector.abs().sum() > 0


def test_works_under_no_grad(potential: CGSchNetPotential, coords: Tensor) -> None:
    """The potential is callable inside ``torch.no_grad()``.

    The unwrapped model is what makes this possible; the wrapped checkpoint raises
    (see :func:`test_wrapped_model_is_unusable_but_unwrapped_is_not`).

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    with torch.no_grad():
        energy = potential(coords)
    assert energy.shape == (4,) and torch.isfinite(energy).all()
    assert not energy.requires_grad


def test_float32_input_round_trips(potential: CGSchNetPotential, coords: Tensor) -> None:
    """A float32 input returns a float32 energy close to the float64 answer.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    energy = potential(coords.to(torch.float32))
    assert energy.dtype == torch.float32
    assert torch.allclose(energy.to(torch.float64), potential(coords), atol=1e-5)


def test_unitcell_is_rejected(potential: CGSchNetPotential, coords: Tensor) -> None:
    """A supplied box raises rather than being silently ignored.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    with pytest.raises(NotImplementedError, match="non-periodic"):
        potential(coords, unitcell_lengths=torch.ones(3, dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="non-periodic"):
        potential(coords, unitcell_angles=torch.full((3,), 90.0, dtype=torch.float64))


@pytest.mark.parametrize("shape", [(6,), (4, 5, 3), (4, 6, 2), (2, 4, 6, 3)])
def test_bad_shapes_are_rejected(potential: CGSchNetPotential, shape: tuple[int, ...]) -> None:
    """Malformed position shapes raise ``ValueError``.

    :param potential: The fixture potential.
    :param shape: A shape that does not describe frames of 6 beads.
    :return: None."""
    with pytest.raises(ValueError):
        potential(torch.zeros(shape, dtype=torch.float64))


def test_to_moves_dtype_and_clears_the_cache(
    potential: CGSchNetPotential, coords: Tensor
) -> None:
    """``to()`` re-casts the model and drops cached batches pinned to the old dtype.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    reference = potential(coords)
    assert potential._batch_cache  # populated by the call above

    potential.to(dtype=torch.float32)
    assert not potential._batch_cache

    energy = potential(coords)
    assert torch.allclose(energy, reference, atol=1e-5)


def test_parameters_are_frozen(potential: CGSchNetPotential) -> None:
    """Model parameters do not require grad, so a caller's backward cannot touch them.

    :param potential: The fixture potential.
    :return: None."""
    for module in potential._models.values():
        assert all(not p.requires_grad for p in module.parameters())


def test_masses_are_dropped(potential: CGSchNetPotential, coords: Tensor) -> None:
    """The integrator-only ``masses`` field is removed from the batch.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    potential(coords)
    batch = potential._batch_cache[4]
    assert getattr(batch, "masses", None) is None


def test_static_batch_is_cached(potential: CGSchNetPotential, coords: Tensor) -> None:
    """Repeated calls at one frame count reuse the same scaffold object.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    potential(coords)
    first = potential._batch_cache[4]
    potential(coords + 0.1)
    assert potential._batch_cache[4] is first


def test_batch_index_fields(potential: CGSchNetPotential, coords: Tensor) -> None:
    """The hand-built batch carries the fields mlcg's SchNet requires.

    :param potential: The fixture potential.
    :param coords: Input frames.
    :return: None."""
    potential(coords)
    batch = potential._batch_cache[4]

    assert torch.equal(batch.batch, torch.arange(4).repeat_interleave(6))
    assert torch.equal(batch.ptr, torch.arange(5) * 6)
    assert torch.equal(batch.n_atoms, torch.full((4,), 6))
    assert batch.atom_types.shape == (24,)

    bonds = batch.neighbor_list["bonds"]
    assert bonds["index_mapping"].shape == (2, 4 * 5)
    assert torch.equal(bonds["mapping_batch"], torch.arange(4).repeat_interleave(5))
    # frame-major blocks, each offset by frame * n_atoms
    assert torch.equal(
        bonds["index_mapping"][:, 5:10], potential._template.neighbor_list["bonds"]["index_mapping"] + 6
    )


def test_unwrap_strips_force_heads() -> None:
    """``unwrap_energy_models`` returns the bare inner modules.

    :return: None."""
    models, _ = _system(6)
    unwrapped = unwrap_energy_models(_FakeSumOut(models))

    assert set(unwrapped) == {"SchNet", "bonds"}
    assert not any(isinstance(m, _FakeGradientsOut) for m in unwrapped.values())
    assert isinstance(unwrapped["SchNet"], _FakeSchNet)


def test_unwrap_rejects_a_non_checkpoint() -> None:
    """A module without a ``models`` mapping is rejected.

    :return: None."""
    with pytest.raises(ValueError, match="models"):
        unwrap_energy_models(nn.Linear(1, 1))


def test_unwrap_requires_the_schnet_term() -> None:
    """A priors-only checkpoint is rejected.

    :return: None."""
    with pytest.raises(ValueError, match="SchNet"):
        unwrap_energy_models(_FakeSumOut({"bonds": _FakeBondPrior("bonds")}))


def test_wrapped_model_is_unusable_but_unwrapped_is_not() -> None:
    """The premise of this module: the wrapped checkpoint fails under ``no_grad``.

    :return: None."""
    models, template = _system(6)
    wrapped = _FakeSumOut(models)

    data = copy.deepcopy(template)
    data.batch = torch.zeros(6, dtype=torch.long)
    data.pos = torch.randn(6, 3, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="does not require grad"):
        with torch.no_grad():
            wrapped(data)

    potential = CGSchNetPotential(
        unwrap_energy_models(wrapped),
        template,
        _topology(["ALA", "ALA"], MODEL_ORDER[:3]),
        dtype=torch.float64,
    )
    with torch.no_grad():
        assert torch.isfinite(potential(torch.randn(6, 3, dtype=torch.float64)))


def test_prune_drops_empty_terms() -> None:
    """Zero-interaction terms and their modules are removed.

    :return: None."""
    models, template = _system(6, empty_term=True)
    kept_models, kept_nl, pruned = prune_empty_terms(models, template.neighbor_list)

    assert pruned == ["ghost"]
    assert set(kept_models) == {"SchNet", "bonds"}
    assert set(kept_nl) == {"bonds"}


def test_prune_requires_a_surviving_prior() -> None:
    """An all-empty neighbour list is an error, not a silent SchNet-only potential.

    :return: None."""
    models, template = _system(6)
    template.neighbor_list["bonds"]["index_mapping"] = torch.zeros((2, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="no prior term survived"):
        prune_empty_terms(models, template.neighbor_list)


def test_unpruned_empty_term_raises_rather_than_collapsing() -> None:
    """An empty term that escapes pruning is caught by the per-term shape check.

    Without that check mlcg's ``dim_size``-less ``scatter`` would silently collapse the
    running sum to a length-0 tensor.

    :return: None."""
    models, template = _system(6, empty_term=True)
    potential = CGSchNetPotential(
        models,
        template,
        _topology(["ALA", "ALA"], MODEL_ORDER[:3]),
        dtype=torch.float64,
    )
    with pytest.raises(RuntimeError, match="empty interaction list"):
        potential(torch.randn(2, 6, 3, dtype=torch.float64))


def test_permutation_is_identity_when_orders_agree() -> None:
    """Matching topologies give the identity permutation.

    :return: None."""
    topology = _topology(["ALA", "GLY", "TRP"], MODEL_ORDER)
    assert bead_permutation(topology, topology) == list(range(topology.n_atoms))


def test_permutation_direction() -> None:
    """``perm[j]`` is the caller index feeding model slot ``j``.

    The real trp-cage pair differs exactly this way (model ``N,CA,CB,C,O`` versus a
    projected ``N,CA,C,O,CB``), so this pins the convention that keeps energies right.

    :return: None."""
    residues = ["ASP", "ALA", "GLY"]
    model_top = _topology(residues, MODEL_ORDER)
    caller_top = _topology(residues, CALLER_ORDER)

    perm = bead_permutation(model_top, caller_top)
    assert perm[:5] == [0, 1, 4, 2, 3]
    assert sorted(perm) == list(range(model_top.n_atoms))

    model_keys = [(a.residue.index, a.name) for a in model_top.atoms]
    caller_keys = [(a.residue.index, a.name) for a in caller_top.atoms]
    assert model_keys == [caller_keys[i] for i in perm]


def test_permutation_round_trip(coords: Tensor) -> None:
    """Permuting the input is equivalent to letting the potential do it.

    Also checks the force path: autograd must hand gradients back in the caller's bead
    order, un-permuted.

    :param coords: Input frames (unused shape-wise; rebuilt for 14 beads).
    :return: None."""
    residues = ["ASP", "ALA", "GLY"]
    model_top = _topology(residues, MODEL_ORDER)
    caller_top = _topology(residues, CALLER_ORDER)
    n_atoms = model_top.n_atoms
    perm = bead_permutation(model_top, caller_top)

    models, template = _system(n_atoms)
    shared = dict(dtype=torch.float64, energy_to_kjmol=1.0)
    identity = CGSchNetPotential(models, copy.deepcopy(template), model_top, None, **shared)
    permuted = CGSchNetPotential(models, copy.deepcopy(template), caller_top, perm, **shared)

    torch.manual_seed(1)
    x_caller = torch.randn(3, n_atoms, 3, dtype=torch.float64, requires_grad=True)
    x_model = x_caller.detach()[:, perm, :].clone().requires_grad_(True)

    e_caller = permuted(x_caller)
    e_model = identity(x_model)
    assert torch.allclose(e_caller, e_model)

    e_caller.sum().backward()
    e_model.sum().backward()
    assert torch.allclose(x_caller.grad[:, perm, :], x_model.grad)


def test_permutation_rejects_a_bead_count_mismatch() -> None:
    """Different bead counts raise.

    :return: None."""
    with pytest.raises(ValueError, match="bead count mismatch"):
        bead_permutation(_topology(["ALA"], MODEL_ORDER), _topology(["ALA", "ALA"], MODEL_ORDER))


def test_permutation_rejects_a_missing_bead() -> None:
    """A bead the model expects but the caller lacks raises, naming it.

    :return: None."""
    model_top = _topology(["ALA", "ALA"], MODEL_ORDER)
    caller_top = _topology(["ALA", "ALA"], ("N", "CA", "CB", "C", "OXT"))
    with pytest.raises(ValueError, match="'O'"):
        bead_permutation(model_top, caller_top)


def test_permutation_rejects_a_residue_name_mismatch() -> None:
    """The same bead names on a different sequence raise.

    :return: None."""
    model_top = _topology(["ALA", "TRP"], MODEL_ORDER)
    caller_top = _topology(["ALA", "PHE"], MODEL_ORDER)
    with pytest.raises(ValueError, match="different systems"):
        bead_permutation(model_top, caller_top)


def test_permutation_rejects_duplicate_beads() -> None:
    """A caller topology with a repeated bead name in one residue raises.

    :return: None."""
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALA", chain)
    for name in ("N", "CA", "CA"):
        topology.add_atom(name, md.element.carbon, residue)
    with pytest.raises(ValueError, match="duplicate bead"):
        bead_permutation(_topology(["ALA"], ("N", "CA", "CB")), topology)


def test_module_does_not_import_mlcg() -> None:
    """Importing the module must not pull in mlcg or torch-geometric.

    ``__init__`` advertises ``CGSchNetPotential`` in ``__all__``, and
    ``test_build.py`` resolves every such name, so a module-level mlcg import would
    break the suite on any environment without the cgschnet extra. Run in a
    subprocess because this session may already have imported mlcg.

    :return: None."""
    code = (
        "import sys, importlib;"
        "importlib.import_module('md_simulations.torch_potentials.cgschnet');"
        "assert 'mlcg' not in sys.modules, 'mlcg imported';"
        "assert 'torch_geometric' not in sys.modules, 'torch_geometric imported';"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_lazy_export_resolves() -> None:
    """``CGSchNetPotential`` is reachable from the package root.

    :return: None."""
    package = importlib.import_module("md_simulations.torch_potentials")
    assert package.CGSchNetPotential is CGSchNetPotential
    assert set(package.__all__) == set(dir(package))
