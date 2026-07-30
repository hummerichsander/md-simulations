"""Batching over independent, heterogeneous systems.

The common case -- many frames of *one* system -- is handled directly by
:meth:`md_simulations.torch_potentials.potential.Potential.__call__` with a leading batch dimension
(``positions`` of shape ``(F, N, 3)``). This module covers the other case:
evaluating several systems with *different* topologies, each its own
:class:`~md_simulations.torch_potentials.potential.Potential`.

External MD engines are single-system, so this simply loops over the items and
stacks the results, optionally across threads. It is **not** fused on-GPU
batching. Threading helps only because engine calls release the GIL; a single
calculator is not safe for concurrent evaluation, so threaded batching requires
*distinct* ``Potential`` instances (each with its own calculator)."""

from collections.abc import Iterable

import torch
from torch import Tensor

from md_simulations.torch_potentials.potential import Potential

Item = tuple[Potential, Tensor] | tuple[Potential, Tensor, Tensor]


def batched_potential_energy(
    systems: Iterable[Item],
    num_workers: int | None = None,
) -> Tensor:
    """Evaluate several heterogeneous systems and stack the results.

    Each item is ``(potential, positions)`` or ``(potential, positions,
    unitcell_lengths)``; it is evaluated as ``potential(positions,
    unitcell_lengths=...)``. The returned tensor is differentiable, so a single
    ``.backward()`` flows the correct force into each item's ``positions`` (the
    caller owns those tensors and reads ``-positions.grad``).

    :param systems: Iterable (list or generator) of items. Each ``positions`` is
        an ``(N_i, 3)`` tensor in nanometre (systems may differ in size).
    :param num_workers: If given and > 1, evaluate items concurrently on a thread
        pool (engine calls release the GIL). Requires distinct ``Potential``
        instances. ``None`` or 1 runs them sequentially.
    :return: ``(n_items,)`` tensor of energies in kJ/mol, differentiable w.r.t.
        each input ``positions``."""
    items = list(systems)

    if num_workers is not None and num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            energies = list(pool.map(_evaluate, items))
    else:
        energies = [_evaluate(item) for item in items]

    return torch.stack(energies)


def _evaluate(item: Item) -> Tensor:
    """Evaluate one ``(potential, positions[, unitcell_lengths])`` item.

    :param item: A 2- or 3-tuple; the third element, if present, is
        ``unitcell_lengths`` (nm).
    :return: The scalar energy for this item."""
    pot, positions = item[0], item[1]
    lengths = item[2] if len(item) > 2 else None
    return pot(positions, unitcell_lengths=lengths)
