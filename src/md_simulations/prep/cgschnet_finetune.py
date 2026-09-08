#!/usr/bin/env python3
"""Fine-tuning helpers for the CGSchNet transferable checkpoint.

mlcg trains the *trainable* part of the model against ``cg_delta_forces`` and only later
merges the result back with the fixed priors (``mlcg-combine_model``). The shipped
checkpoint is a ``SumOut`` over 52 terms of which exactly one, ``SchNet``, carries
parameters — the other 51 are analytic priors whose tensors are buffers, not parameters.
Two things follow, and this module provides both:

* Training should start from the *pretrained* ``SchNet`` term rather than a fresh one, so
  :func:`pretrained_schnet` hands ``PLModel`` that exact submodule. Referencing it from a
  training YAML by ``class_path`` is what makes the run a fine-tune instead of a restart.
* The merge step afterwards needs the priors on their own, so :func:`write_prior_model`
  saves the checkpoint with ``SchNet`` dropped.

Splitting train from validation deserves a note. Frames 5 ps apart are far closer than the
~10 ns correlation time, so a random frame split leaves near-duplicates on both sides and
reports a validation loss that merely echoes the training loss. :func:`write_multi_h5`
therefore re-emits a dataset with one molecule group per source run, letting a whole
trajectory be held out — a split that actually measures generalisation.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("md_simulations.prep.cgschnet_finetune")

SCHNET_KEY = "SchNet"


def load_model(model_file: Path | str, device: str = "cpu"):
    """Load a re-exported CGSchNet checkpoint.

    :param model_file: Path to the re-exported ``.pt`` (see ``md-sim-cgschnet-reexport``).
    :param device: Device to map tensors onto.
    :return: The ``SumOut`` model."""

    return torch.load(str(model_file), map_location=device, weights_only=False)


def pretrained_schnet(model_file: str, device: str = "cpu") -> torch.nn.Module:
    """The pretrained ``SchNet`` term, ready to hand to ``mlcg.pl.PLModel`` as ``model``.

    Returns the term with its ``GradientsOut`` wrapper intact — that wrapper is what turns
    the energy into the ``forces`` prediction the force-matching loss compares against, so
    unlike the *evaluation* path (which strips it) training needs it kept.

    :param model_file: Path to the re-exported checkpoint.
    :param device: Device to map tensors onto.
    :return: The ``GradientsOut``-wrapped SchNet, with pretrained weights.
    :raises KeyError: If the checkpoint has no ``SchNet`` term."""

    model = load_model(model_file, device)
    if SCHNET_KEY not in model.models:
        raise KeyError(
            f"{model_file} has no '{SCHNET_KEY}' term; available: {list(model.models)}."
        )
    schnet = model.models[SCHNET_KEY]
    n_params = sum(p.numel() for p in schnet.parameters())
    logger.info(f"fine-tuning from pretrained {SCHNET_KEY} ({n_params:,} parameters)")
    return schnet


class PretrainedSchNet(torch.nn.Module):
    """YAML-referenceable handle on the pretrained ``SchNet`` term.

    ``mlcg.pl.PLModel`` declares ``model: torch.nn.Module``, so a training YAML can only
    supply it through a ``class_path`` naming an ``nn.Module`` subclass — a plain factory
    function is rejected. Constructing this class therefore *returns the checkpoint's own
    submodule* rather than a wrapper around it: ``PLModel.get_model()`` and
    ``merge_priors_and_checkpoint`` both expect the genuine ``GradientsOut``, and an extra
    level of nesting would prefix every state-dict key and break the merge.

    :param model_file: Path to the re-exported checkpoint.
    :param device: Device to map tensors onto."""

    def __new__(cls, model_file: str, device: str = "cpu") -> torch.nn.Module:  # type: ignore[misc]
        return pretrained_schnet(model_file, device)


def write_prior_model(model_file: Path | str, out_file: Path | str) -> int:
    """Save the checkpoint's fixed priors alone, for the post-training merge.

    :param model_file: Path to the re-exported checkpoint.
    :param out_file: Destination ``.pt``.
    :return: The number of prior terms written.
    :raises KeyError: If the checkpoint has no ``SchNet`` term to drop."""

    model = load_model(model_file)
    if SCHNET_KEY not in model.models:
        raise KeyError(f"{model_file} has no '{SCHNET_KEY}' term to drop.")
    del model.models[SCHNET_KEY]
    torch.save(model, str(out_file))
    logger.info(f"wrote {len(model.models)} prior terms to {out_file}")
    return len(model.models)


def write_multi_h5(
    path: Path | str,
    metaset: str,
    cg_coords: np.ndarray,
    cg_delta_forces: np.ndarray,
    cg_embeds: np.ndarray,
    segment_lengths: np.ndarray,
    names: list[str] | None = None,
) -> list[str]:
    """Write one molecule group per source run under a single metaset.

    Same per-molecule contract as
    :func:`~md_simulations.prep.cgschnet_dataset.write_h5` — ``cg_coords``,
    ``cg_delta_forces``, and the mandatory ``cg_embeds`` / ``N_frames`` attributes — but
    with the frames split back into their originating trajectories. That is what lets
    ``partition_options`` hold out an entire run for validation instead of interleaved
    frames, which at this frame spacing would leak almost completely.

    :param path: Output ``.h5`` path.
    :param metaset: Outer group name (the mlcg "metaset").
    :param cg_coords: CG coordinates (frames, n_beads, 3) in Angstrom.
    :param cg_delta_forces: Delta forces (frames, n_beads, 3) in kcal/mol/Angstrom.
    :param cg_embeds: Per-bead embedding types, shared by every molecule group.
    :param segment_lengths: Frames contributed by each source run, in order.
    :param names: Molecule group names; defaults to ``seg00``, ``seg01``, ….
    :return: The molecule names written, in order.
    :raises ValueError: If the segment lengths do not sum to the frame count."""

    import h5py

    lengths = np.asarray(segment_lengths, dtype=np.int64)
    if lengths.sum() != len(cg_coords):
        raise ValueError(
            f"segment_lengths sum to {lengths.sum()} but there are {len(cg_coords)} frames."
        )
    names = names or [f"seg{i:02d}" for i in range(len(lengths))]
    if len(names) != len(lengths):
        raise ValueError(f"got {len(names)} names for {len(lengths)} segments.")

    bounds = np.r_[0, np.cumsum(lengths)]
    with h5py.File(str(path), "w") as handle:
        parent = handle.create_group(metaset)
        for name, a, b in zip(names, bounds[:-1], bounds[1:]):
            group = parent.create_group(name)
            group.create_dataset("cg_coords", data=cg_coords[a:b].astype(np.float32))
            group.create_dataset(
                "cg_delta_forces", data=cg_delta_forces[a:b].astype(np.float32)
            )
            group.attrs["cg_embeds"] = np.asarray(cg_embeds, dtype=np.int64)
            group.attrs["N_frames"] = int(b - a)
            logger.info(f"{metaset}/{name}: {b - a} frames")
    return names


def main() -> None:
    """CLI entry point: split a built dataset by run and emit the prior-only model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, type=Path, help="Directory written by md-sim-cgschnet-dataset."
    )
    parser.add_argument(
        "--model-file", required=True, type=Path, help="Re-exported CGSchNet checkpoint."
    )
    parser.add_argument("--metaset", default="trpcage", help="Metaset (outer group) name.")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Molecule group names, one per source run (default seg00, seg01, ...).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = args.dataset
    coords = np.load(data / "cg_coords.npy", mmap_mode="r")
    delta = np.load(data / "cg_delta_forces.npy", mmap_mode="r")
    lengths = np.load(data / "segment_lengths.npy")

    import json

    meta = json.loads((data / "dataset.json").read_text())
    embeds = np.asarray(meta["cg_embeds"], dtype=np.int64)

    names = args.names
    if not names:
        names = [Path(r["run_dir"]).name for r in meta["runs"]]
    write_multi_h5(
        data / f"{args.metaset}_bysegment.h5", args.metaset, coords, delta, embeds, lengths, names
    )
    write_prior_model(args.model_file, data / "prior_only.pt")


if __name__ == "__main__":
    main()
