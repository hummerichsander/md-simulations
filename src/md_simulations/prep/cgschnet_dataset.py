#!/usr/bin/env python3
"""Build a CGSchNet force-matching dataset from an all-atom OpenMM run.

Projects an all-atom trajectory and its forces onto the CG bead set and writes the
``.npy`` arrays ``mlcg``'s training scripts consume. This is the piece
:mod:`~md_simulations.prep.cgschnet_map` deliberately leaves out — that tool handles
coordinates and embeddings only, and says force mapping is out of scope.

Two conventions matter and both are easy to get silently wrong:

**Units.** ``aggforce``/``mlcg`` work in Angstrom and kcal/mol; OpenMM writes nm and
kJ/mol/nm. ``guess_pairwise_constraints`` thresholds a *distance variance*, so feeding it
nm over-detects badly — on trp-cage it finds 602 "constraints" instead of the 136 OpenMM
actually applies. The detected count matching the run summary is the cheapest correctness
check available, so it is always logged.

**Force map.** Every map satisfying Noid's consistency condition is an unbiased estimator
of the same mean force, so the choice cannot bias the fit — only the variance of the
training signal. Measured on trp-cage against the transferable model, ``slice_optimize``
carries ~32% less variance than ``slice_aggregate`` (residual std 6.40 vs 7.75 kcal/mol/A,
both unbiased), so it is the default.

Usage::

    md-sim-cgschnet-dataset --run data/trp-cage/AMBER14_implicit_scan_360K \\
        --bead-pdb data/pdbs/2JOF_5B-CG.pdb --out data/trp-cage/fm_360K \\
        --cgschnet-config configs/trpcage-cgschnet-300K.yaml --name trpcage

With ``--cgschnet-config`` the fixed prior's forces are subtracted and an mlcg-ready
``.h5`` is written in the layout of ``examples/h5_pl/single_molecule/1L2Y_prior_tag.h5``:
``<name>/<name>/{cg_coords,cg_delta_forces}``. SchNet is the only trainable term, so it
trains on that residual and the prior is added back at simulation time.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import mdtraj as md

logger = logging.getLogger("md_simulations.prep.cgschnet_dataset")

ForceMap = Literal["slice_optimize", "slice_aggregate"]

# OpenMM nm / kJ mol^-1 nm^-1  ->  aggforce Angstrom / kcal mol^-1 A^-1
NM_TO_ANGSTROM = 10.0
KJNM_TO_KCALA = 1.0 / 4.184 / 10.0
CONSTRAINT_THRESHOLD = 5e-3


def bead_indices(aa_topology: md.Topology, cg_topology: md.Topology) -> np.ndarray:
    """Map each CG bead to its all-atom index, in the CG topology's own order.

    The CG topology fixes the order, so the caller controls whether the output is in
    mlcg's bead order (``N, CA, CB, C, O``) or AMBER's (``N, CA, C, O, CB``). These differ
    and the mismatch is silent — same bead count, permuted coordinates.

    :param aa_topology: All-atom topology of the source trajectory.
    :param cg_topology: CG topology defining the bead set and their order.
    :return: Array of all-atom indices, one per bead.
    :raises ValueError: If a bead has no matching all-atom atom."""

    indices = []
    for atom in cg_topology.atoms:
        hits = aa_topology.select(f"resSeq {atom.residue.resSeq} and name {atom.name}")
        if len(hits) != 1:
            raise ValueError(
                f"bead {atom.residue.name}{atom.residue.resSeq}-{atom.name} matched "
                f"{len(hits)} all-atom atoms, expected exactly 1."
            )
        indices.append(int(hits[0]))

    return np.asarray(indices, dtype=np.int64)


def iter_force_frames(path: Path, n_atoms: int, chunk: int) -> Iterator[np.ndarray]:
    """Stream ``forces.txt`` in chunks of frames.

    The file is whitespace text with one line per frame (341 MB for 50k frames of 284
    atoms), so it is never loaded whole.

    :param path: Path to the force file written by ``ForceReporter``.
    :param n_atoms: Number of all-atom sites per frame.
    :param chunk: Frames per yielded block.
    :return: Iterator over arrays of shape (frames, n_atoms, 3) in kJ/mol/nm."""

    with path.open() as handle:
        while True:
            rows = [np.fromstring(line, sep=" ") for line in _take(handle, chunk)]
            if not rows:
                return
            yield np.asarray(rows).reshape(len(rows), n_atoms, 3)


def _take(handle, count: int) -> list[str]:
    """Read up to ``count`` lines from an open handle.

    :param handle: Open text handle.
    :param count: Maximum number of lines.
    :return: The lines read (fewer than ``count`` at end of file)."""

    out = []
    for line in handle:
        out.append(line)
        if len(out) == count:
            break

    return out


def frame_selection(
    n_frames: int,
    stride: int = 1,
    start: int = 0,
    stop: int | None = None,
    frame_indices: Path | None = None,
) -> np.ndarray:
    """Resolve which frames to keep, as absolute indices into the source trajectory.

    Selection is resolved up front and applied by masking inside each already-paired
    block — never by striding the readers. ``mdtraj.iterload(chunk=c, stride=s)`` yields
    blocks of ``c`` frames spanning ``c * s`` *file* frames, so striding the trajectory
    reader while taking every ``s``-th line of ``forces.txt`` silently misaligns
    coordinates and forces by a factor of ``s``.

    ``--start`` is an equilibration correction and is legitimate. Selecting frames by
    *collective-variable value* would not be: it reshapes the sampling distribution in a
    way that is not a time-translation, and while force matching tolerates a reweighted
    configuration distribution, it must remain one the dynamics actually produced.

    :param n_frames: Frames available in the source trajectory.
    :param stride: Keep every ``stride``-th frame of the ``[start, stop)`` window.
    :param start: First frame to consider.
    :param stop: One past the last frame to consider; ``None`` means to the end.
    :param frame_indices: ``.npy`` file of explicit indices; excludes the other options.
    :return: Sorted array of unique frame indices.
    :raises ValueError: On an empty selection, or invalid or conflicting arguments."""

    if frame_indices is not None:
        if stride != 1 or start != 0 or stop is not None:
            raise ValueError(
                "--frame-indices cannot be combined with --stride/--start/--stop."
            )
        selection = np.asarray(np.load(frame_indices)).ravel()
        if not np.issubdtype(selection.dtype, np.integer):
            raise ValueError(f"{frame_indices} must hold integers, got {selection.dtype}.")
        if len(np.unique(selection)) != len(selection):
            raise ValueError(f"{frame_indices} contains duplicate indices.")
        if len(selection) and (selection.min() < 0 or selection.max() >= n_frames):
            raise ValueError(
                f"{frame_indices} has indices outside [0, {n_frames}): "
                f"[{selection.min()}, {selection.max()}]."
            )
        selection = np.sort(selection)
    else:
        if stride < 1:
            raise ValueError(f"--stride must be at least 1, got {stride}.")
        if start < 0:
            raise ValueError(f"--start must be non-negative, got {start}.")
        selection = np.arange(start, min(stop or n_frames, n_frames), stride)

    if len(selection) == 0:
        raise ValueError("The frame selection is empty; nothing to project.")
    return selection.astype(np.int64)


def count_lines(path: Path) -> int:
    """Count lines in a text file without holding it in memory.

    :param path: Path to the file.
    :return: The number of lines."""

    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))


def cg_trajectory(cg_coords_angstrom: np.ndarray, bead_pdb: Path) -> md.Trajectory:
    """Wrap CG coordinates as an ``mdtraj.Trajectory`` on the bead topology.

    The unit conversion is the point: this module stores Angstrom for aggforce and mlcg,
    while mdtraj is nm throughout. Skipping the division inflates every distance tenfold,
    which drives ``Q`` to zero and the radius of gyration to nonsense — with nothing
    raising.

    :param cg_coords_angstrom: CG coordinates (frames, n_beads, 3) in Angstrom.
    :param bead_pdb: CG topology matching the bead order of the coordinates.
    :return: A trajectory in nm."""

    topology = md.load(str(bead_pdb)).topology
    return md.Trajectory(
        np.asarray(cg_coords_angstrom, dtype=np.float32) / NM_TO_ANGSTROM, topology
    )


def cg_native(aa_native_pdb: Path, indices: np.ndarray, bead_pdb: Path) -> md.Trajectory:
    """Project an all-atom reference structure onto the bead set.

    CG-space collective variables need a CG-space reference: the all-atom native's heavy
    atoms are indexed against a 284-atom topology, and those indices mean nothing on a
    97-bead one. Contact sets must be rebuilt here, not carried over.

    :param aa_native_pdb: All-atom reference structure.
    :param indices: All-atom index per bead, from :func:`bead_indices`.
    :param bead_pdb: CG topology defining the bead order.
    :return: A single-frame CG trajectory in nm."""

    native = md.load(str(aa_native_pdb))[0]
    topology = md.load(str(bead_pdb)).topology
    return md.Trajectory(native.xyz[:, indices, :], topology)


def build_force_map(
    coords: np.ndarray,
    forces: np.ndarray,
    cg_matrix: np.ndarray,
    kind: ForceMap,
    expected_constraints: int | None = None,
) -> tuple[np.ndarray, int]:
    """Fit the all-atom -> CG force projection matrix.

    :param coords: All-atom coordinates (frames, n_atoms, 3) in Angstrom.
    :param forces: All-atom forces (frames, n_atoms, 3) in kcal/mol/Angstrom.
    :param cg_matrix: Configurational slice map (n_beads, n_atoms).
    :param kind: ``slice_optimize`` (lower variance) or ``slice_aggregate``.
    :param expected_constraints: Constraint count from the run summary, warned against.
    :return: The force map (n_beads, n_atoms) and the detected constraint count."""

    from aggforce import (
        LinearMap,
        constraint_aware_uni_map,
        guess_pairwise_constraints,
        project_forces,
        qp_linear_map,
    )

    constraints = guess_pairwise_constraints(coords, threshold=CONSTRAINT_THRESHOLD)
    logger.info(f"detected {len(constraints)} pairwise constraints")
    if expected_constraints is not None and len(constraints) != expected_constraints:
        logger.warning(
            f"detected {len(constraints)} constraints but the run applied "
            f"{expected_constraints} — check that coordinates are in Angstrom."
        )

    method, kwargs = (
        (qp_linear_map, {"l2_regularization": 1e3})
        if kind == "slice_optimize"
        else (constraint_aware_uni_map, {})
    )
    result = project_forces(
        coords=coords,
        forces=forces,
        coord_map=LinearMap(cg_matrix),
        constrained_inds=constraints,
        method=method,
        **kwargs,
    )

    return result["tmap"].force_map.standard_matrix, len(constraints)


def iter_paired_blocks(
    run_dir: Path, aa_pdb: Path, n_atoms: int, chunk: int, selection: np.ndarray
):
    """Yield selected all-atom coordinate/force blocks, in Angstrom and kcal/mol/Angstrom.

    Both readers advance unstrided and in lockstep; the selection is applied as a mask
    *within* each paired block, so coordinates and forces can never drift apart.

    :param run_dir: Simulation output directory.
    :param aa_pdb: All-atom topology.
    :param n_atoms: All-atom site count per frame.
    :param chunk: Frames per block.
    :param selection: Absolute frame indices to keep, sorted.
    :return: Iterator of ``(positions, forces)`` float64 blocks."""

    dcd = run_dir / "trajectory.dcd"
    force_blocks = iter_force_frames(run_dir / "forces.txt", n_atoms, chunk)
    offset = 0
    for block in md.iterload(str(dcd), top=str(aa_pdb), chunk=chunk):
        try:
            f_block = next(force_blocks)
        except StopIteration:
            logger.warning(f"{run_dir}: force file ended before the trajectory; truncating.")
            return
        take = min(len(block), len(f_block))
        local = selection[(selection >= offset) & (selection < offset + take)] - offset
        offset += take
        if len(local) == 0:
            continue
        yield (
            (block.xyz[local] * NM_TO_ANGSTROM).astype(np.float64),
            (f_block[local] * KJNM_TO_KCALA).astype(np.float64),
        )


def read_all_atom(
    run_dir: Path, aa_pdb: Path, n_atoms: int, chunk: int, selection: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Read selected all-atom coordinates and forces, unprojected, in float64.

    Used for the force-map fit, which runs a quadratic program and wants full precision.

    :param run_dir: Simulation output directory.
    :param aa_pdb: All-atom topology.
    :param n_atoms: All-atom site count per frame.
    :param chunk: Frames per block.
    :param selection: Absolute frame indices to keep, sorted.
    :return: ``(coords, forces)`` in Angstrom and kcal/mol/Angstrom.
    :raises ValueError: If no frame survived the selection."""

    coords, forces = [], []
    for positions, f_atoms in iter_paired_blocks(run_dir, aa_pdb, n_atoms, chunk, selection):
        coords.append(positions)
        forces.append(f_atoms)
    if not coords:
        raise ValueError(f"{run_dir}: no frames survived the selection.")
    return np.concatenate(coords), np.concatenate(forces)


def project_run(
    run_dir: Path,
    aa_pdb: Path,
    cg_matrix: np.ndarray,
    projection: np.ndarray,
    n_atoms: int,
    chunk: int,
    selection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one run's selected frames onto the CG beads.

    :param run_dir: Simulation output directory.
    :param aa_pdb: All-atom topology.
    :param cg_matrix: One-hot slicing map (n_beads, n_atoms) for coordinates.
    :param projection: aggforce force map (n_beads, n_atoms).
    :param n_atoms: All-atom site count per frame.
    :param chunk: Frames per block.
    :param selection: Absolute frame indices to keep, sorted.
    :return: ``(cg_coords, cg_forces)`` in Angstrom and kcal/mol/Angstrom.
    :raises ValueError: If no frame survived the selection."""

    coords, forces = [], []
    for positions, f_atoms in iter_paired_blocks(run_dir, aa_pdb, n_atoms, chunk, selection):
        coords.append(np.einsum("mn,ind->imd", cg_matrix, positions).astype(np.float32))
        forces.append(np.einsum("mn,ind->imd", projection, f_atoms).astype(np.float32))

    if not coords:
        raise ValueError(f"{run_dir}: no frames survived the selection.")
    return np.concatenate(coords), np.concatenate(forces)


def build_dataset(
    run_dirs: Path | list[Path],
    bead_pdb: Path,
    out_dir: Path,
    force_map: ForceMap = "slice_optimize",
    fit_frames: int = 500,
    chunk: int = 500,
    topology: Path | None = None,
    cgschnet_config: Path | None = None,
    data_root: Path | None = None,
    name: str | None = None,
    stride: int = 1,
    start: int = 0,
    stop: int | None = None,
    frame_indices: Path | None = None,
    prior_device: str = "cpu",
    write_cg_dcd: bool = False,
    native_pdb: Path | None = None,
) -> dict[str, object]:
    """Project one or more all-atom runs onto their CG beads and write the training arrays.

    Several runs pool into a single dataset so independent replicas at the *same*
    temperature can be combined. They must share a temperature: force matching targets
    the conditional mean force, which is itself temperature-dependent, so pooling two
    temperatures averages two different targets at the same configuration.

    :param run_dirs: One or more simulation output directories
        (``trajectory.dcd``, ``forces.txt``).
    :param bead_pdb: CG topology fixing the bead set and order.
    :param out_dir: Directory the ``.npy`` arrays and metadata are written to.
    :param force_map: Which aggforce map to fit.
    :param fit_frames: Frames used to fit the map and detect constraints.
    :param chunk: Frames held in memory per block.
    :param topology: All-atom topology; defaults to the first run's ``initial_structure.pdb``.
    :param cgschnet_config: cgschnet config; when given, prior forces are subtracted and an
        mlcg-ready ``.h5`` is written alongside the arrays.
    :param data_root: Data root the cgschnet config's paths resolve against.
    :param name: Molecule name for the h5 groups; defaults to the first run's name.
    :param stride: Keep every ``stride``-th selected frame of each run.
    :param start: First frame of each run to consider.
    :param stop: One past the last frame of each run to consider.
    :param frame_indices: ``.npy`` of explicit indices, applied to every run.
    :param prior_device: Device for prior-force evaluation (``"cpu"`` or ``"cuda"``).
    :param write_cg_dcd: Also write ``cg_trajectory.dcd`` for CG-space analysis.
    :param native_pdb: All-atom reference for the CG-space coverage check.
    :return: Metadata describing the written dataset."""

    runs = [run_dirs] if isinstance(run_dirs, Path) else list(run_dirs)
    if not runs:
        raise ValueError("At least one run directory is required.")
    name = name or runs[0].name

    aa_pdb = topology or runs[0] / "initial_structure.pdb"
    aa_topology = md.load(str(aa_pdb)).topology
    cg_topology = md.load(str(bead_pdb)).topology
    indices = bead_indices(aa_topology, cg_topology)
    logger.info(f"{len(indices)} beads from {aa_topology.n_atoms} atoms, {len(runs)} run(s)")

    cg_matrix = np.zeros((len(indices), aa_topology.n_atoms))
    cg_matrix[np.arange(len(indices)), indices] = 1.0

    # Pairing is positional: line i of forces.txt is frame i of the DCD. Check it rather
    # than discovering a truncated force file as a quietly shorter dataset.
    selections, source_frames = [], []
    for run in runs:
        with md.open(str(run / "trajectory.dcd")) as handle:
            n_dcd = len(handle)
        n_force = count_lines(run / "forces.txt")
        if n_force != n_dcd:
            raise ValueError(
                f"{run}: forces.txt has {n_force} lines but the DCD has {n_dcd} frames. "
                "Frame pairing is positional, so a mismatch means the run is incomplete."
            )
        source_frames.append(n_dcd)
        selections.append(frame_selection(n_dcd, stride, start, stop, frame_indices))

    # The map depends on the topology and constraints, not on the frames, so it is fit
    # once on the first run's selected prefix and reused everywhere. Reusing it is also
    # required: a per-run map would make the pooled forces inconsistent.
    first = selections[0][:fit_frames]
    fit_coords, fit_forces = read_all_atom(
        runs[0], aa_pdb, aa_topology.n_atoms, chunk, first
    )
    projection, n_constraints = build_force_map(
        fit_coords,
        fit_forces,
        cg_matrix,
        force_map,
        expected_constraints=_summary_constraints(runs[0]),
    )

    coords_blocks, forces_blocks, per_run = [], [], []
    for run, selection in zip(runs, selections):
        c, f = project_run(
            run, aa_pdb, cg_matrix, projection, aa_topology.n_atoms, chunk, selection
        )
        coords_blocks.append(c)
        forces_blocks.append(f)
        per_run.append({"run_dir": str(run), "n_frames": int(len(c)),
                        "temperature": _summary_temperature(run)})
        logger.info(f"{run.name}: projected {len(c)} frames")

    out_dir.mkdir(parents=True, exist_ok=True)
    coords_out = np.concatenate(coords_blocks)
    forces_out = np.concatenate(forces_blocks)
    # Release the per-run blocks: they double peak memory once concatenated, which at
    # ~half a million frames is the difference between 2 GB and 4 GB.
    coords_blocks.clear()
    forces_blocks.clear()
    n_frames = int(len(coords_out))
    np.save(out_dir / "cg_coords.npy", coords_out)
    np.save(out_dir / "cg_forces.npy", forces_out)
    np.save(out_dir / "force_map.npy", projection)
    np.save(out_dir / "bead_indices.npy", indices)
    np.save(out_dir / "segment_lengths.npy",
            np.array([r["n_frames"] for r in per_run], dtype=np.int64))

    temperatures = {r["temperature"] for r in per_run if r["temperature"] is not None}
    if len(temperatures) > 1:
        logger.warning(
            f"pooling runs at different temperatures {sorted(temperatures)}: the mean "
            "force is temperature-dependent, so the fitted potential targets neither."
        )

    delta_written = False
    embeds = None
    if cgschnet_config is not None:
        priors = prior_forces(
            coords_out, cgschnet_config, data_root or Path("data"), bead_pdb,
            device=prior_device,
        )
        delta = forces_out - priors
        np.save(out_dir / "cg_delta_forces.npy", delta)
        embeds = cg_embeddings(cgschnet_config, data_root or Path("data"), bead_pdb)
        write_h5(out_dir / f"{name}.h5", name, coords_out, delta, embeds)
        delta_written = True
        logger.info(
            f"delta forces: RMS {np.sqrt((delta ** 2).mean()):.2f} vs total "
            f"{np.sqrt((forces_out ** 2).mean()):.2f} kcal/mol/A "
            f"(prior explains {100 * (1 - (delta ** 2).mean() / (forces_out ** 2).mean()):.1f}%)"
        )

    coverage = None
    if write_cg_dcd or native_pdb is not None:
        cg_traj = cg_trajectory(coords_out, bead_pdb)
        if write_cg_dcd:
            cg_traj.save_dcd(str(out_dir / "cg_trajectory.dcd"))
        if native_pdb is not None:
            coverage = cg_coverage(cg_traj, native_pdb, indices, bead_pdb)
            logger.info(
                f"CG-space coverage: Q {coverage['q_mean']:.3f} "
                f"[{coverage['q_min']:.3f}, {coverage['q_max']:.3f}], "
                f"Rg {coverage['rg_mean']:.3f} nm, "
                f"frames below Q=0.45: {100 * coverage['fraction_low_q']:.1f}%"
            )

    meta = {
        "run_dirs": [str(r) for r in runs],
        "runs": per_run,
        "bead_pdb": str(bead_pdb),
        "aa_topology": str(aa_pdb),
        "n_source_frames": [int(n) for n in source_frames],
        "n_frames": n_frames,
        "stride": int(stride),
        "start": int(start),
        "stop": None if stop is None else int(stop),
        "frame_indices": None if frame_indices is None else str(frame_indices),
        "n_beads": int(len(indices)),
        "n_atoms": int(aa_topology.n_atoms),
        "force_map": force_map,
        "fit_frames": int(len(first)),
        "n_constraints": int(n_constraints),
        "name": name,
        "delta_forces": delta_written,
        "cg_embeds": None if embeds is None else embeds.tolist(),
        "temperature": sorted(temperatures)[0] if len(temperatures) == 1 else sorted(temperatures),
        "cg_coverage": coverage,
        "units": {"positions": "angstrom", "forces": "kcal/mol/angstrom"},
    }
    (out_dir / "dataset.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"wrote {n_frames} frames to {out_dir}")

    return meta


def cg_coverage(
    cg_traj: md.Trajectory, native_pdb: Path, indices: np.ndarray, bead_pdb: Path
) -> dict[str, float]:
    """Basin coverage of a CG dataset, measured in CG space.

    Recorded in ``dataset.json`` so "does the training set contain both basins" is a
    property of the dataset rather than something a human remembers to check. ``Q`` is
    recomputed against a bead-projected reference; the all-atom contact set does not
    transfer, and CG ``Rg`` is mass-weighted over bead masses so it is a bimodality
    probe rather than a number comparable to the all-atom value.

    :param cg_traj: CG trajectory in nm.
    :param native_pdb: All-atom reference structure.
    :param indices: All-atom index per bead, from :func:`bead_indices`.
    :param bead_pdb: CG topology defining the bead order.
    :return: Summary statistics of ``Q`` and the radius of gyration."""

    from md_simulations.analysis.cvs import fraction_native_contacts, radius_of_gyration

    native = cg_native(native_pdb, indices, bead_pdb)
    q = fraction_native_contacts(cg_traj, native, min_seq_sep=6)
    rg = radius_of_gyration(cg_traj)
    return {
        "q_mean": float(q.mean()),
        "q_std": float(q.std()),
        "q_min": float(q.min()),
        "q_max": float(q.max()),
        "fraction_low_q": float(np.mean(q < 0.45)),
        "rg_mean": float(rg.mean()),
        "rg_std": float(rg.std()),
        "q_min_seq_sep": 6,
    }


def _summary_temperature(run_dir: Path) -> float | None:
    """Read the temperature out of ``simulation_summary.txt`` if present.

    :param run_dir: Simulation output directory.
    :return: The temperature in K, or None if unavailable."""

    summary = run_dir / "simulation_summary.txt"
    if not summary.exists():
        return None
    for line in summary.read_text().splitlines():
        if line.strip().startswith("Temperature:"):
            try:
                return float(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                return None
    return None


def prior_forces(
    cg_coords_angstrom: np.ndarray,
    cgschnet_config: Path,
    data_root: Path,
    bead_pdb: Path,
    device: str = "cpu",
    chunk: int = 200,
) -> np.ndarray:
    """Evaluate the fixed prior's forces on CG coordinates.

    SchNet is trained on *delta* forces — the CG force minus the prior's — because at
    simulation time the prior is added back. mlcg's own trp-cage example stores exactly
    ``cg_delta_forces``, so the subtraction has to happen here. The prior-only potential is
    the shipped model with its single trainable term (``SchNet``) dropped; every other term
    carries zero parameters and is a fixed analytic potential.

    :param cg_coords_angstrom: CG coordinates (frames, n_beads, 3) in Angstrom.
    :param cgschnet_config: Path to the cgschnet simulation config.
    :param data_root: Data root the config's relative paths resolve against.
    :param bead_pdb: CG topology in the same bead order as ``cg_coords_angstrom``.
    :param device: Device to evaluate on.
    :param chunk: Frames per forward pass.
    :return: Prior forces (frames, n_beads, 3) in kcal/mol/Angstrom."""

    import torch

    from md_simulations.config.load import load_config
    from md_simulations.torch_potentials.cgschnet import (
        CGSchNetPotential,
        bead_permutation,
        prune_empty_terms,
        unwrap_energy_models,
    )

    config = load_config(str(cgschnet_config))
    topology = md.load(str(bead_pdb)).topology
    model = torch.load(
        str(config.resolve(data_root, config.model_file)), map_location=device, weights_only=False
    )
    configurations = torch.load(
        str(config.resolve(data_root, config.configurations_file)),
        map_location="cpu",
        weights_only=False,
    )
    template = configurations[0] if isinstance(configurations, (list, tuple)) else configurations

    terms = unwrap_energy_models(model)
    dropped = terms.pop("SchNet", None)
    if dropped is None:
        raise ValueError("model has no 'SchNet' term; cannot separate prior from learned part.")
    logger.info(f"prior-only potential keeps {len(terms)} terms, dropped SchNet")

    import copy

    template = copy.deepcopy(template)
    kept_models, kept_neighbor_list, _ = prune_empty_terms(terms, template.neighbor_list)
    template.neighbor_list = kept_neighbor_list

    model_top = md.load(str(config.resolve(data_root, config.input_pdb))).topology
    potential = CGSchNetPotential(
        kept_models,
        template,
        topology,
        permutation=bead_permutation(model_top, topology),
        device=device,
    )

    out = []
    for start in range(0, len(cg_coords_angstrom), chunk):
        block = cg_coords_angstrom[start : start + chunk] / NM_TO_ANGSTROM
        positions = torch.tensor(block, dtype=torch.float32, device=device).requires_grad_(True)
        potential(positions).sum().backward()
        out.append((-positions.grad).detach().cpu().numpy())

    return (np.concatenate(out) * KJNM_TO_KCALA).astype(np.float32)


def write_h5(
    path: Path,
    name: str,
    cg_coords: np.ndarray,
    cg_delta_forces: np.ndarray,
    cg_embeds: np.ndarray,
) -> None:
    """Write a single-molecule H5 dataset in the layout mlcg's training scripts expect.

    Layout is ``<partition>/<molecule>/`` — here both named ``name`` — with the datasets
    on the molecule group and two **attributes** alongside them. Both attributes are
    mandatory: ``mlcg.datasets.h5_dataset`` reads ``attrs:N_frames`` unconditionally in
    ``MetaSet.grab_n_frames``, and ``attrs:cg_embeds`` is the default
    ``hdf_key_mapping["embeds"]``. A file without them raises on load rather than
    training, so omitting them makes the dataset silently unusable.

    :param path: Output ``.h5`` path.
    :param name: Molecule name used for both group levels.
    :param cg_coords: CG coordinates (frames, n_beads, 3) in Angstrom.
    :param cg_delta_forces: Delta forces (frames, n_beads, 3) in kcal/mol/Angstrom.
    :param cg_embeds: Per-bead embedding types (n_beads,), in ``cg_coords`` bead order.
    :return: None.
    :raises ValueError: If the embedding length does not match the bead count."""

    import h5py

    if len(cg_embeds) != cg_coords.shape[1]:
        raise ValueError(
            f"cg_embeds has {len(cg_embeds)} entries but the coordinates have "
            f"{cg_coords.shape[1]} beads."
        )

    with h5py.File(path, "w") as handle:
        group = handle.create_group(f"{name}/{name}")
        group.create_dataset("cg_coords", data=cg_coords.astype(np.float32))
        group.create_dataset("cg_delta_forces", data=cg_delta_forces.astype(np.float32))
        group.attrs["cg_embeds"] = np.asarray(cg_embeds, dtype=np.int64)
        group.attrs["N_frames"] = int(cg_coords.shape[0])


def cg_embeddings(
    cgschnet_config: Path, data_root: Path, bead_pdb: Path
) -> np.ndarray:
    """Per-bead embedding types for a CG topology, taken from the model's configurations.

    The types live in ``configurations_file`` in the *model's* bead order, while the
    dataset is written in ``bead_pdb`` order. ``bead_permutation`` gives ``perm`` with
    ``perm[j]`` the caller-bead index at model slot ``j``, so recovering caller order is
    a scatter (``embeds[perm] = types``) and *not* a gather. On trp-cage the two
    topologies are the same file and ``perm`` is the identity, so the wrong direction
    would go unnoticed here and silently corrupt the next system.

    :param cgschnet_config: Path to the cgschnet simulation config.
    :param data_root: Data root the config's relative paths resolve against.
    :param bead_pdb: CG topology defining the output bead order.
    :return: 1-D ``np.int64`` array of length ``n_beads``."""

    import torch

    from md_simulations.config.load import load_config
    from md_simulations.prep.cgschnet_embeddings import extract_atom_types
    from md_simulations.torch_potentials.cgschnet import bead_permutation

    config = load_config(str(cgschnet_config))
    configurations = torch.load(
        str(config.resolve(data_root, config.configurations_file)),
        map_location="cpu",
        weights_only=False,
    )
    types = extract_atom_types(configurations)

    model_top = md.load(str(config.resolve(data_root, config.input_pdb))).topology
    bead_top = md.load(str(bead_pdb)).topology
    permutation = bead_permutation(model_top, bead_top)
    if len(types) != len(permutation):
        raise ValueError(
            f"configurations carry {len(types)} bead types but the topology has "
            f"{len(permutation)} beads."
        )

    embeds = np.empty(len(permutation), dtype=np.int64)
    embeds[np.asarray(permutation, dtype=np.int64)] = types
    return embeds


def _summary_constraints(run_dir: Path) -> int | None:
    """Read the constraint count out of ``simulation_summary.txt`` if present.

    :param run_dir: Simulation output directory.
    :return: The applied constraint count, or None if unavailable."""

    summary = run_dir / "simulation_summary.txt"
    if not summary.exists():
        return None
    for line in summary.read_text().splitlines():
        if line.strip().startswith("Constraints:"):
            try:
                # The summary writes this with a thousands separator.
                return int(line.split(":", 1)[1].strip().replace(",", ""))
            except ValueError:
                return None

    return None


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        required=True,
        nargs="+",
        type=Path,
        help="One or more simulation output directories, pooled into one dataset. "
        "They must share a temperature.",
    )
    parser.add_argument("--bead-pdb", required=True, type=Path, help="CG topology (bead set and order).")
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--topology", type=Path, default=None, help="All-atom topology PDB.")
    parser.add_argument(
        "--force-map",
        default="slice_optimize",
        choices=["slice_optimize", "slice_aggregate"],
        help="aggforce map; slice_optimize carries ~32%% less variance.",
    )
    parser.add_argument("--fit-frames", type=int, default=500, help="Frames used to fit the map.")
    parser.add_argument("--chunk", type=int, default=500, help="Frames per in-memory block.")
    parser.add_argument(
        "--cgschnet-config",
        type=Path,
        default=None,
        help="cgschnet config; subtracts prior forces and writes an mlcg-ready .h5.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="cgschnet data root.")
    parser.add_argument("--name", default=None, help="Molecule name for the h5 groups.")
    parser.add_argument(
        "--stride", type=int, default=1, help="Keep every Nth frame (default 1: all frames)."
    )
    parser.add_argument(
        "--start", type=int, default=0, help="First frame to keep, to drop an equilibration."
    )
    parser.add_argument("--stop", type=int, default=None, help="One past the last frame to keep.")
    parser.add_argument(
        "--frame-indices",
        type=Path,
        default=None,
        help=".npy of explicit frame indices; excludes --stride/--start/--stop.",
    )
    parser.add_argument(
        "--prior-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for prior-force evaluation (default cpu; use cuda for large sets).",
    )
    parser.add_argument(
        "--write-cg-dcd",
        action="store_true",
        help="Also write cg_trajectory.dcd so md-sim-cvs/-tica/-basins can read the CG set.",
    )
    parser.add_argument(
        "--native",
        type=Path,
        default=None,
        help="All-atom reference for the CG-space basin-coverage check in dataset.json.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_dataset(
        args.run,
        args.bead_pdb,
        args.out,
        force_map=args.force_map,
        fit_frames=args.fit_frames,
        chunk=args.chunk,
        topology=args.topology,
        cgschnet_config=args.cgschnet_config,
        data_root=args.data_root,
        name=args.name,
        stride=args.stride,
        start=args.start,
        stop=args.stop,
        frame_indices=args.frame_indices,
        prior_device=args.prior_device,
        write_cg_dcd=args.write_cg_dcd,
        native_pdb=args.native,
    )


if __name__ == "__main__":
    main()
