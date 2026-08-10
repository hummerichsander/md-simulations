"""Differentiable PyTorch potentials from the configured MD force fields.

Build a :class:`Potential` from an mdtraj ``Topology`` (the system) and an ASE
``Calculator`` (the swappable engine -- OpenMM, LAMMPS, EMT, ...), then evaluate it
with torch tensors and differentiate to get forces. All numeric inputs and outputs
are torch tensors in nm / kJ.mol^-1; ASE is an internal implementation detail.

- :func:`build_potential` / :func:`build_potential_from_file` -- the usual entry
  point: a differentiable potential straight from a simulation config.
- :func:`build_calculator` / :func:`build_calculator_from_file` -- the ASE
  calculator alone, for callers that do not want torch.
- :class:`Potential` -- the differentiable, ASE-engine-backed potential.
- :class:`CGSchNetPotential` -- the native-torch potential for CGSchNet/mlcg
  coarse-grained models, which have no ASE calculator. Batched over frames and twice
  differentiable; needs the ``cgschnet-potentials`` extra (Python 3.12).
- :class:`PotentialLike` -- the contract both potentials satisfy.
- :func:`batched_potential_energy` -- evaluate several heterogeneous systems.

The engine-agnostic ASE bridge lives in
:mod:`md_simulations.torch_potentials.bridge` for advanced use and imports no
specific engine.

Requires the ``torch-potentials`` extra. The torch-backed names are resolved lazily
so that ``import md_simulations`` (which reaches this module eagerly) stays usable
on a core install; they raise :class:`ModuleNotFoundError` with install guidance on
first access instead."""

from importlib import import_module
from typing import Any

from md_simulations.torch_potentials.build import (
    EXTRA_HINT,
    build_calculator,
    build_calculator_from_file,
    build_potential,
    build_potential_from_file,
)
from md_simulations.torch_potentials.protocol import PotentialLike

__all__ = [
    "CGSchNetPotential",
    "Potential",
    "PotentialLike",
    "batched_potential_energy",
    "build_calculator",
    "build_calculator_from_file",
    "build_potential",
    "build_potential_from_file",
]

# Names whose modules import torch. Kept out of the eager import above so a
# core-only install can still `import md_simulations` (see the module docstring).
# `cgschnet` belongs here for torch alone: it imports no mlcg at module scope, so it
# resolves on a plain torch-potentials install and only needs mlcg once a checkpoint
# is actually unpickled.
_LAZY: dict[str, tuple[str, str]] = {
    "CGSchNetPotential": ("md_simulations.torch_potentials.cgschnet", "CGSchNetPotential"),
    "Potential": ("md_simulations.torch_potentials.potential", "Potential"),
    "batched_potential_energy": (
        "md_simulations.torch_potentials.batch",
        "batched_potential_energy",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve a torch-backed name on first access.

    :param name: Attribute name being looked up.
    :return: The resolved object.
    :raises AttributeError: If ``name`` is not an attribute of this module.
    :raises ModuleNotFoundError: If the ``torch-potentials`` extra is missing."""
    if (target := _LAZY.get(name)) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module, attr = target
    try:
        resolved = getattr(import_module(module), attr)
    except ModuleNotFoundError as exc:  # pragma: no cover - needs a core-only install
        raise ModuleNotFoundError(f"{name} {EXTRA_HINT}") from exc

    globals()[name] = resolved  # cache so __getattr__ runs once per name
    return resolved


def __dir__() -> list[str]:
    """List the module's public names, including the lazily-resolved ones.

    :return: Sorted public attribute names."""
    return sorted(__all__)
