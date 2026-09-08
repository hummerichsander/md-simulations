"""Shared trajectory loading and free-energy-surface helpers.

Engine-agnostic building blocks reused by the analysis scripts (TICA and
collective variables): trajectory loading, periodic-image sanitising, and the
smoothed 2-D free-energy histogram. Depends only on ``mdtraj``/``numpy``/``scipy``.
"""

from pathlib import Path

import numpy as np


def drop_spurious_box(traj):
    """Drop an unphysical unit cell smaller than the molecule it contains.

    Implicit-solvent runs still write a placeholder unit cell to the trajectory,
    which would wrap the whole molecule onto a point. A real periodic box must be
    able to contain the molecule, so any box smaller than the molecule's span is
    dropped (leaving the trajectory non-periodic).

    :param traj: An ``mdtraj.Trajectory``.
    :return: The trajectory, modified in place (box cleared if spurious)."""
    if traj.unitcell_lengths is None:
        return traj
    span = np.linalg.norm(traj.xyz.max(axis=1) - traj.xyz.min(axis=1), axis=1).max()
    if traj.unitcell_lengths.min() < span:
        traj.unitcell_vectors = None
    return traj


def make_ca_whole(traj):
    """Make a sliced Cα chain whole across periodic boundaries.

    Bonds consecutive Cα atoms so mdtraj can walk the chain and apply the
    minimum-image convention bond-by-bond (consecutive Cαs are ~0.38 nm apart,
    well within L/2, so this is robust even for extended states). Spurious
    implicit-solvent boxes are dropped first via :func:`drop_spurious_box`.

    :param traj: Cα-only ``mdtraj.Trajectory``.
    :return: The trajectory made whole, modified in place."""
    drop_spurious_box(traj)
    if traj.unitcell_lengths is None:
        return traj
    cas = list(traj.top.atoms)
    for a, b in zip(cas[:-1], cas[1:]):
        traj.top.add_bond(a, b)
    traj.make_molecules_whole(inplace=True)
    return traj


def load_segments(pdb: str | Path, frames: list[str | Path], stride: int = 1) -> list:
    """Load trajectory files as separate segments, one per file.

    Keeping the files apart matters wherever consecutive frames are assumed to be
    consecutive in *time*: independent replicas concatenated into one series would
    contribute a spurious jump at every seam (a fake transition when counting basin
    crossings, a bogus lagged pair when fitting TICA).

    :param pdb: Path to a PDB providing the topology.
    :param frames: One or more trajectory files (any mdtraj-readable format).
    :param stride: Keep every ``stride``-th frame.
    :return: One ``mdtraj.Trajectory`` per input file, in order."""
    import mdtraj as md

    return [md.load(str(f), top=str(pdb), stride=stride) for f in frames]


def load_trajectory(pdb: str | Path, frames: list[str | Path], stride: int = 1):
    """Load one or more trajectory files against a PDB topology.

    :param pdb: Path to a PDB providing the topology.
    :param frames: One or more trajectory files (any mdtraj-readable format).
    :param stride: Keep every ``stride``-th frame.
    :return: A single concatenated ``mdtraj.Trajectory``."""
    import mdtraj as md

    parts = load_segments(pdb, frames, stride)
    return parts[0] if len(parts) == 1 else md.join(parts)


def free_energy_1d(
    x: np.ndarray,
    bins: int = 90,
    sigma: float = 2.0,
    f_max: float = 7.0,
    x_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed 1-D free energy ``-kT ln p`` (kT = 1) on a histogram grid.

    :param x: The coordinate.
    :param bins: Number of histogram bins.
    :param sigma: Gaussian smoothing width in bins.
    :param f_max: Free-energy ceiling; higher values are set to NaN. Pass ``np.inf``
        to keep the whole surface, which basin location needs — a rarely visited
        well can sit far above any plotting ceiling.
    :param x_range: Optional ``(min, max)`` histogram span; defaults to the data extent.
    :return: ``(centres, F)``, both of shape ``(bins,)``."""
    from scipy.ndimage import gaussian_filter

    H, edges = np.histogram(x, bins=bins, range=x_range)
    p = gaussian_filter(H.astype(float), sigma)
    p /= p.sum()
    F = np.full(p.shape, np.inf)
    np.log(p, out=F, where=p > 0)
    np.negative(F, out=F, where=p > 0)
    F -= F[np.isfinite(F)].min()
    F[F > f_max] = np.nan
    return 0.5 * (edges[:-1] + edges[1:]), F


def free_energy_2d(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    sigma: float,
    f_max: float,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed 2-D free energy ``-kT ln p`` (kT = 1) on a histogram grid.

    :param x: First coordinate.
    :param y: Second coordinate.
    :param bins: Histogram bins per axis.
    :param sigma: Gaussian smoothing width in bins.
    :param f_max: Free-energy ceiling; higher values are set to NaN.
    :param x_range: Optional ``(min, max)`` histogram span for x; defaults to the
        data extent. Set it to the plot limits so the surface fills the axes.
    :param y_range: Optional ``(min, max)`` histogram span for y.
    :return: ``(x_centres, y_centres, F)`` with ``F`` of shape ``(bins, bins)``."""
    from scipy.ndimage import gaussian_filter

    hist_range = None if x_range is None and y_range is None else [x_range, y_range]
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=hist_range)
    p = gaussian_filter(H, sigma)
    p /= p.sum()
    F = np.full(p.shape, np.inf)
    np.log(p, out=F, where=p > 0)
    np.negative(F, out=F, where=p > 0)
    F -= F[np.isfinite(F)].min()
    F[F > f_max] = np.nan
    return 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), F
