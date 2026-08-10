"""Batched, differentiable CG potential from a CGSchNet/mlcg checkpoint.

Unlike every other engine here, cgschnet's model is *already* a torch module, so it
gets a native :class:`PotentialLike` instead of an ASE calculator: no numpy round
trip, frames batched into one forward, and real second derivatives.

Three things about the distributed checkpoint make a thin wrapper impossible, and
they are the reason this module exists:

1. The pickled object is an ``mlcg.nn.gradients.SumOut`` over ~52 terms, each wrapped
   in a ``GradientsOut`` force head whose ``forward`` calls ``torch.autograd.grad``
   and then does ``data.pos = data.pos.detach()``. So calling it directly (a) raises
   ``RuntimeError: element 0 of tensors does not require grad`` under
   ``torch.no_grad()``, (b) leaves every term after the first disconnected from the
   caller's positions, and (c) runs ~52 internal backward passes for a caller who
   asked only for energy. :func:`unwrap_energy_models` strips those shells.
2. Prior terms for residue types absent from the molecule carry an empty
   ``index_mapping``, and mlcg's priors ``scatter`` without ``dim_size`` -- an empty
   term returns a length-0 energy, which silently collapses the running sum to
   length 0. :func:`prune_empty_terms` drops them (18 of 51 for trp-cage). mlcg
   avoids this with ``specialize_priors``, which is unusable here because it
   concatenates static buffers across a fixed ``data_list``, baking the frame count
   into the model.
3. The prior ``index_mapping`` arrays index bead *positions*, so a caller whose CG
   topology orders beads differently from the model's own gets silently wrong
   energies. :func:`bead_permutation` derives the reordering and
   :class:`CGSchNetPotential` applies it as an autograd op, so forces come back in
   the caller's order for free.

The model is native Angstrom + kcal/mol and strictly non-periodic (every prior
neighbour list has ``rcut=None``/``cell_shifts=None``), so a supplied unit cell is
rejected rather than ignored.

This module imports ``torch`` only. mlcg is pulled in implicitly by ``torch.load``
inside :meth:`CGSchNetPotential.from_config`, never at import time, so the package
stays importable (and the class stays unit-testable) without mlcg installed."""

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from md_simulations.torch_potentials.bridge import KCALMOL_TO_KJMOL, NM_TO_ANG

if TYPE_CHECKING:
    from md_simulations.config import CGSchNetConfig

logger = logging.getLogger(__name__)

# Key mlcg's models write their energy under in ``data.out[name]``.
_ENERGY_KEY = "energy"
# The one term that is not a prior: it has no neighbour-list entry of its own and
# rebuilds its 15 A radius graph from ``data.pos`` on every forward.
_SCHNET_KEY = "SchNet"
# Backbone beads whose embedding index must be residue-independent; used as a cheap,
# permutation-independent check that input_pdb and configurations_file agree.
_BACKBONE_BEADS = ("N", "C", "O")

_DTYPES: dict[str, torch.dtype] = {"float32": torch.float32, "float64": torch.float64}


def bead_permutation(
    bead_topology: "mdtraj.Topology",  # noqa: F821
    topology: "mdtraj.Topology",  # noqa: F821
) -> list[int]:
    """Index array reordering the caller's beads into the model's bead order.

    Beads are matched on ``(residue.index, atom.name)``, so ``result[j]`` is the index
    in *topology* of the bead occupying model slot ``j``. Applied as a gather
    (``positions[..., perm, :]``), which lets autograd invert it on the force path.

    There is deliberately no identity fallback: the model's priors index bead
    positions, so an unresolvable topology must fail loudly rather than produce
    plausible-looking nonsense.

    :param bead_topology: The model's own CG topology (``config.input_pdb``).
    :param topology: The caller's CG topology.
    :return: A permutation of ``range(n_atoms)``; ``list(range(n))`` if the orders
        already agree.
    :raises ValueError: On an atom-count, duplicate-key, unmatched-bead or
        residue-name mismatch, naming the offending bead."""
    if bead_topology.n_atoms != topology.n_atoms:
        raise ValueError(
            f"bead count mismatch: the model's topology has {bead_topology.n_atoms} beads, "
            f"the supplied topology has {topology.n_atoms}."
        )

    caller: dict[tuple[int, str], int] = {}
    for i, atom in enumerate(topology.atoms):
        key = (atom.residue.index, atom.name)
        if key in caller:
            raise ValueError(
                f"duplicate bead {atom.name!r} in residue {atom.residue.index} of the supplied "
                "topology; bead matching needs (residue index, atom name) to be unique."
            )
        caller[key] = i

    permutation: list[int] = []
    for atom in bead_topology.atoms:
        key = (atom.residue.index, atom.name)
        if (index := caller.get(key)) is None:
            raise ValueError(
                f"the supplied topology has no bead {atom.name!r} in residue {atom.residue.index} "
                f"({atom.residue.name}), which the model's topology expects."
            )
        matched = topology.atom(index).residue.name
        if matched != atom.residue.name:
            raise ValueError(
                f"residue {atom.residue.index} is {atom.residue.name!r} in the model's topology "
                f"but {matched!r} in the supplied one; these are different systems."
            )
        permutation.append(index)

    return permutation


def unwrap_energy_models(model: nn.Module) -> dict[str, nn.Module]:
    """Strip mlcg's ``GradientsOut`` force heads, keeping the bare energy modules.

    See this module's docstring (point 1) for why the wrapped model cannot be called
    directly. Each child is unwrapped via ``getattr(child, "model", child)``, so an
    already-bare module passes through unchanged.

    :param model: The loaded checkpoint, an ``mlcg.nn.gradients.SumOut``.
    :return: Term name -> bare energy module, in the checkpoint's order.
    :raises ValueError: If ``model`` has no ``models`` mapping, or if the SchNet term
        is absent (which would mean the checkpoint is priors-only)."""
    if (children := getattr(model, "models", None)) is None:
        raise ValueError(
            f"expected a checkpoint with a 'models' mapping (mlcg SumOut), got "
            f"{type(model).__name__}."
        )

    unwrapped = {name: getattr(child, "model", child) for name, child in children.items()}
    if not unwrapped:
        raise ValueError("the checkpoint's 'models' mapping is empty.")
    if _SCHNET_KEY not in unwrapped:
        raise ValueError(
            f"the checkpoint has no {_SCHNET_KEY!r} term (only {sorted(unwrapped)}); "
            "this does not look like a CGSchNet model + prior."
        )
    return unwrapped


def prune_empty_terms(
    models: dict[str, nn.Module], neighbor_list: dict[str, dict[str, Any]]
) -> tuple[dict[str, nn.Module], dict[str, dict[str, Any]], list[str]]:
    """Drop prior terms that have no interactions in this molecule.

    See this module's docstring (point 2): an empty term's ``scatter`` returns a
    length-0 energy and collapses the sum. Terms without a neighbour-list entry are
    dropped too -- only ``SchNet`` builds its own graph.

    :param models: Term name -> bare energy module, from :func:`unwrap_energy_models`.
    :param neighbor_list: The configuration's ``neighbor_list`` mapping.
    :return: ``(kept_models, kept_neighbor_list, pruned_names)``, the last sorted.
    :raises ValueError: If no prior term survives."""
    kept_nl = {
        name: term for name, term in neighbor_list.items() if term["index_mapping"].shape[1] > 0
    }
    kept_models = {
        name: module for name, module in models.items() if name == _SCHNET_KEY or name in kept_nl
    }
    pruned = sorted(set(models) - set(kept_models))

    if len(kept_models) < 2:
        raise ValueError(
            "no prior term survived pruning; the configuration's neighbor_list is empty for "
            f"every term in the checkpoint ({sorted(models)[:5]}...)."
        )
    return kept_models, kept_nl, pruned


class CGSchNetPotential:
    """Batched, differentiable CG energy of a CGSchNet/mlcg model (nm in, kJ/mol out).

    Deliberately **not** an ``nn.Module``: the checkpoint carries ~390 MB of
    parameters that must not be registered as submodules of a caller's model, where
    they would land in ``state_dict()``, checkpoints and optimizer param groups.
    Mirrors :class:`~md_simulations.torch_potentials.potential.Potential`'s call
    signature, so both satisfy
    :class:`~md_simulations.torch_potentials.protocol.PotentialLike`.

    Construction is expensive (a ~390 MB unpickle); build it once and reuse it."""

    def __init__(
        self,
        energy_models: dict[str, nn.Module],
        template: Any,
        topology: "mdtraj.Topology",  # noqa: F821
        permutation: list[int] | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        max_batch_frames: int = 64,
        energy_to_kjmol: float = KCALMOL_TO_KJMOL,
    ) -> None:
        """Bind pruned energy terms and a static system to a topology.

        Takes already-loaded objects so it stays testable without mlcg; see
        :meth:`from_config` for the config-driven path.

        :param energy_models: Term name -> bare energy module (pruned).
        :param template: One mlcg ``AtomicData`` supplying ``atom_types``,
            ``neighbor_list`` and ``n_atoms``. Its ``pos`` and ``masses`` are unused.
        :param topology: The caller's CG topology, whose bead order the call signature
            follows.
        :param permutation: Caller -> model bead order, or ``None`` for identity.
        :param device: Torch device the model and batches live on.
        :param dtype: Floating-point dtype for the model and the forward pass.
        :param max_batch_frames: Frames per mlcg forward; the frame axis is chunked.
        :param energy_to_kjmol: The model's energy unit expressed in kJ/mol."""
        if max_batch_frames < 1:
            raise ValueError(f"max_batch_frames must be >= 1 (got {max_batch_frames}).")

        self._topology = topology
        self._n_atoms = topology.n_atoms
        self._max_batch_frames = max_batch_frames
        self._energy_to_kjmol = energy_to_kjmol
        self._device = torch.device(device)
        self._dtype = dtype

        # Match mlcg's own Simulation._attach_model: inference mode, params frozen so
        # a caller's backward pass cannot accumulate grads into 390 MB of weights.
        self._models = {}
        for name, module in energy_models.items():
            module = module.to(device=self._device, dtype=dtype).eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            self._models[name] = module

        self._template = self._prepare_template(template)
        # Normalized so an identity reordering is indistinguishable from none at all --
        # both mean "the caller's bead order is already the model's".
        if permutation is not None and list(permutation) == list(range(self._n_atoms)):
            permutation = None
        self._permutation = None if permutation is None else list(permutation)
        self._perm_index = (
            None
            if self._permutation is None
            else torch.as_tensor(self._permutation, dtype=torch.long, device=self._device)
        )
        self._batch_cache: dict[int, Any] = {}

    @classmethod
    def from_config(
        cls,
        config: "CGSchNetConfig",
        data_root: Path,
        topology: "mdtraj.Topology",  # noqa: F821
        *,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        max_batch_frames: int = 64,
    ) -> "CGSchNetPotential":
        """Build the potential from a cgschnet simulation config.

        Reads ``model_file`` (the re-exported checkpoint), ``configurations_file``
        (the only source of ``atom_types`` and the prior neighbour lists) and
        ``input_pdb`` (the model's bead order, used to derive the permutation).

        :param config: A validated ``CGSchNetConfig``.
        :param data_root: Root the config's relative paths resolve against.
        :param topology: The caller's CG topology.
        :param device: Torch device override; defaults to ``config.device``.
        :param dtype: Dtype override, a torch dtype or ``"float32"``/``"float64"``;
            defaults to ``config.dtype``.
        :param max_batch_frames: Frames per mlcg forward.
        :return: A ready :class:`CGSchNetPotential`.
        :raises FileNotFoundError: If any of the three inputs is missing.
        :raises ValueError: On a bead-count, permutation or embedding inconsistency."""
        import mdtraj as md

        model_path = config.resolve(data_root, config.model_file)
        configs_path = config.resolve(data_root, config.configurations_file)
        bead_pdb = config.resolve(data_root, config.input_pdb)
        for path, role in (
            (model_path, "model_file (the re-exported model + prior checkpoint)"),
            (configs_path, "configurations_file (atom types and prior neighbour lists)"),
            (bead_pdb, "input_pdb (the model's coarse-grained bead order)"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{role} not found: {path}")

        logger.info(f"Loading CG model:       {model_path}")
        # map_location='cpu' then .to(device): loading a ~390 MB pickle straight onto
        # the GPU can OOM a device that is already holding a training run.
        model = torch.load(str(model_path), map_location="cpu", weights_only=False)

        logger.info(f"Loading configurations: {configs_path}")
        configurations = torch.load(str(configs_path), map_location="cpu", weights_only=False)
        if not isinstance(configurations, (list, tuple)):
            configurations = [configurations]
        if not 0 <= config.replica < len(configurations):
            raise ValueError(
                f"replica {config.replica} out of range for {len(configurations)} configurations."
            )
        template = copy.deepcopy(configurations[config.replica])

        n_beads = int(template.n_atoms.reshape(-1)[0])
        if n_beads != topology.n_atoms:
            raise ValueError(
                f"the configuration has {n_beads} beads but the supplied topology has "
                f"{topology.n_atoms}; these are different systems."
            )

        bead_topology = md.load_topology(str(bead_pdb))
        permutation = bead_permutation(bead_topology, topology)
        _check_backbone_embeddings(bead_topology, template.atom_types)
        if permutation != list(range(n_beads)):
            logger.info(
                f"Reordering beads to the model's order (from {bead_pdb.name}): "
                f"first indices {permutation[:5]}"
            )

        models, neighbor_list, pruned = prune_empty_terms(
            unwrap_energy_models(model), template.neighbor_list
        )
        template.neighbor_list = neighbor_list
        if pruned:
            logger.info(f"Pruned {len(pruned)} zero-interaction prior terms: {', '.join(pruned)}")

        if dtype is None:
            dtype = config.dtype
        if isinstance(dtype, str):
            if dtype not in _DTYPES:
                raise ValueError(f"unsupported dtype {dtype!r}; expected one of {sorted(_DTYPES)}.")
            dtype = _DTYPES[dtype]

        return cls(
            models,
            template,
            topology,
            permutation,
            device=config.device if device is None else device,
            dtype=dtype,
            max_batch_frames=max_batch_frames,
            energy_to_kjmol=KCALMOL_TO_KJMOL if config.model_energy_unit == "kcal/mol" else 1.0,
        )

    @property
    def n_atoms(self) -> int:
        """Number of coarse-grained beads in the bound system."""
        return self._n_atoms

    @property
    def topology(self) -> "mdtraj.Topology":  # noqa: F821
        """The bound mdtraj ``Topology`` (the caller's bead order)."""
        return self._topology

    @property
    def terms(self) -> tuple[str, ...]:
        """Names of the energy terms that survived pruning."""
        return tuple(self._models)

    @property
    def permutation(self) -> list[int] | None:
        """Caller -> model bead order, or ``None`` if the orders already agree."""
        return self._permutation

    def to(
        self, device: str | torch.device | None = None, dtype: torch.dtype | None = None
    ) -> "CGSchNetPotential":
        """Move the model and static system to another device and/or dtype, in place.

        :param device: Target device, or ``None`` to keep the current one.
        :param dtype: Target floating-point dtype, or ``None`` to keep the current one.
        :return: ``self``, for chaining."""
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype

        for name, module in self._models.items():
            self._models[name] = module.to(device=self._device, dtype=self._dtype)
        self._template = self._prepare_template(self._template)
        if self._perm_index is not None:
            self._perm_index = self._perm_index.to(self._device)
        self._batch_cache.clear()  # cached batches pin the old device/dtype
        return self

    def __call__(
        self,
        positions: Tensor,
        unitcell_lengths: Tensor | None = None,
        unitcell_angles: Tensor | None = None,
    ) -> Tensor:
        """Return the differentiable potential energy at ``positions``.

        :param positions: ``(N, 3)`` (one frame) or ``(F, N, 3)`` (a batch of frames)
            bead positions in nanometre, in the bound topology's bead order. Set
            ``requires_grad=True`` to recover forces as ``-positions.grad``.
        :param unitcell_lengths: Must be ``None`` -- the model is non-periodic.
        :param unitcell_angles: Must be ``None`` -- the model is non-periodic.
        :return: Scalar energy in kJ/mol for ``(N, 3)`` input, else a ``(F,)`` tensor,
            differentiable (twice) w.r.t. ``positions``.
        :raises NotImplementedError: If a unit cell is supplied.
        :raises ValueError: On a malformed ``positions`` shape."""
        if unitcell_lengths is not None or unitcell_angles is not None:
            raise NotImplementedError(
                "the CGSchNet model is non-periodic (every prior neighbour list has "
                "rcut=None), so a unit cell cannot be honoured; pass unitcell_lengths=None."
            )
        if positions.dim() not in (2, 3):
            raise ValueError(f"positions must be (N, 3) or (F, N, 3), got {tuple(positions.shape)}")
        if positions.shape[-2:] != (self._n_atoms, 3):
            raise ValueError(
                f"positions must end in ({self._n_atoms}, 3) to match the bound topology, "
                f"got {tuple(positions.shape)}"
            )

        in_device, in_dtype = positions.device, positions.dtype
        single_frame = positions.dim() == 2
        x = positions.unsqueeze(0) if single_frame else positions

        x = x.to(device=self._device, dtype=self._dtype)
        if self._perm_index is not None:
            x = x.index_select(-2, self._perm_index)
        x = x * NM_TO_ANG  # nm -> A; autograd turns this into the force conversion

        chunks = [
            self._energy(x[start : start + self._max_batch_frames])
            for start in range(0, x.shape[0], self._max_batch_frames)
        ]
        energy = (chunks[0] if len(chunks) == 1 else torch.cat(chunks)) * self._energy_to_kjmol

        energy = energy.to(device=in_device, dtype=in_dtype)
        return energy.squeeze(0) if single_frame else energy

    def _energy(self, positions: Tensor) -> Tensor:
        """Sum the surviving energy terms over a batch of frames.

        :param positions: ``(F, N, 3)`` positions in Angstrom, in the model's bead order.
        :return: ``(F,)`` energies in the model's energy unit.
        :raises RuntimeError: If a term returns an energy that is not per-frame, which
            is how a zero-interaction term that escaped pruning would manifest."""
        n_frames = positions.shape[0]
        data = self._static_batch(n_frames)
        data.pos = positions.reshape(-1, 3)
        data.out = {}

        total = None
        for name, module in self._models.items():
            data = module(data)
            energy = data.out[name][_ENERGY_KEY].reshape(-1)
            if energy.shape != (n_frames,):
                raise RuntimeError(
                    f"term {name!r} returned an energy of shape "
                    f"{tuple(data.out[name][_ENERGY_KEY].shape)} for {n_frames} frames; "
                    "an empty interaction list would do this (see prune_empty_terms)."
                )
            total = energy if total is None else total + energy
        return total

    def _static_batch(self, n_frames: int) -> Any:
        """Return the position-independent batch scaffold for ``n_frames`` frames.

        Everything except ``pos`` is fixed by the topology, so this is built once per
        distinct frame count and cached (chunking means at most two ever occur).
        Reuse is safe because mlcg's SchNet rebuilds its radius graph from ``pos`` on
        every forward and writes nothing back into ``neighbor_list``.

        The batch is assembled by hand rather than via torch-geometric's ``collate``:
        that would mean importing torch-geometric at module scope, and it has never
        been exercised on these unspecialized 51-key neighbour lists.

        :param n_frames: Number of frames in the batch.
        :return: An ``AtomicData``-shaped object with batched static fields."""
        if (cached := self._batch_cache.get(n_frames)) is not None:
            return cached

        n_atoms, device = self._n_atoms, self._device
        frames = torch.arange(n_frames, device=device)

        data = copy.deepcopy(self._template)
        data.atom_types = self._template.atom_types.repeat(n_frames)
        data.n_atoms = torch.full((n_frames,), n_atoms, dtype=torch.long, device=device)
        # SchNet needs `batch` (it scatters energies by it and reads data.batch[-1]).
        data.batch = frames.repeat_interleave(n_atoms)
        data.ptr = torch.arange(n_frames + 1, device=device) * n_atoms

        offsets = (frames * n_atoms).view(1, n_frames, 1)
        neighbor_list = {}
        for name, term in self._template.neighbor_list.items():
            index_mapping = term["index_mapping"]
            order, n_interactions = index_mapping.shape
            neighbor_list[name] = {
                **term,
                # (order, M) -> (order, F*M), frame-major to match mapping_batch below.
                "index_mapping": (index_mapping.unsqueeze(1) + offsets).reshape(
                    order, n_frames * n_interactions
                ),
                "mapping_batch": frames.repeat_interleave(n_interactions),
            }
        data.neighbor_list = neighbor_list

        # Integrator-only, and its length would disagree with the batched fields.
        if getattr(data, "masses", None) is not None:
            del data.masses

        self._batch_cache[n_frames] = data
        return data

    def _prepare_template(self, template: Any) -> Any:
        """Move a configuration's static tensors to this potential's device.

        Integer fields keep their dtype -- they are indices and embedding ids, not
        floating-point state.

        :param template: An ``AtomicData``-shaped object.
        :return: ``template``, with its static tensors on the target device."""
        template.atom_types = template.atom_types.to(self._device)
        template.n_atoms = template.n_atoms.to(self._device)
        template.neighbor_list = {
            name: {
                key: value.to(self._device) if isinstance(value, Tensor) else value
                for key, value in term.items()
            }
            for name, term in template.neighbor_list.items()
        }
        return template


def _check_backbone_embeddings(
    bead_topology: "mdtraj.Topology",  # noqa: F821
    atom_types: Tensor,
) -> None:
    """Assert the backbone beads' embedding indices are residue-independent.

    In this model family ``N``/``C``/``O`` each map to a single embedding index
    regardless of residue. Checking that against the model's own bead topology is a
    cheap way to catch an ``input_pdb`` and ``configurations_file`` that describe
    different systems -- and it is independent of the permutation logic, so the two
    cannot fail together for the same reason.

    :param bead_topology: The model's CG topology, in the same order as ``atom_types``.
    :param atom_types: ``(N,)`` embedding indices from the configuration.
    :raises ValueError: If a backbone bead name maps to more than one index."""
    for name in _BACKBONE_BEADS:
        indices = [i for i, atom in enumerate(bead_topology.atoms) if atom.name == name]
        if not indices:
            continue
        distinct = sorted({int(atom_types[i]) for i in indices})
        if len(distinct) > 1:
            raise ValueError(
                f"bead {name!r} maps to several embedding indices {distinct} across residues; "
                "input_pdb and configurations_file most likely describe different systems."
            )
