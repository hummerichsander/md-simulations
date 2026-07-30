#!/usr/bin/env python3
"""Time-lagged independent component analysis (TICA) of an MD trajectory.

Featurises a trajectory (Cα distances and/or backbone torsions), then either
fits a fresh TICA model or projects onto a preexisting one. Fitting pickles the
model so a later run can reproduce the same projection for another trajectory
(e.g. to compare a coarse-grained run against a reference) by passing it back
via ``--tica-model``. Passing several ``--lags`` fits one model per lag and
writes a multi-panel free-energy comparison.

Featurisation depends only on ``mdtraj``/``numpy`` (core deps); fitting a new
model additionally needs ``deeptime`` (the ``analysis`` extra).
"""

import argparse
from pathlib import Path

import numpy as np

from md_simulations.analysis.fes import (
    free_energy_2d,
    load_trajectory,
    make_ca_whole,
)


def ca_distance_features(traj, exclude_neighbors: int = 2) -> np.ndarray:
    """Compute pairwise Cα–Cα distances after making the chain whole.

    :param traj: An ``mdtraj.Trajectory`` (any atom selection; Cαs are sliced
        out internally).
    :param exclude_neighbors: Skip Cα pairs closer than this in sequence
        (``|i - j| <= exclude_neighbors``); the default of 2 matches PyEMMA's
        ``add_distances_ca``.
    :return: Array of shape ``(n_frames, n_pairs)`` of distances in nm.
    :raises ValueError: If the topology has fewer than two Cα atoms."""
    ca = traj.atom_slice(traj.top.select("name CA"))
    n = ca.n_atoms
    if n < 2:
        raise ValueError(f"Need at least two Cα atoms for distances, found {n}.")
    ca = make_ca_whole(ca)
    pairs = np.array(
        [(i, j) for i in range(n) for j in range(i + 1, n) if j - i > exclude_neighbors]
    )
    if len(pairs) == 0:
        raise ValueError(
            f"No Cα pairs left after excluding {exclude_neighbors} neighbors "
            f"({n} Cα atoms). Lower --exclude-neighbors."
        )
    import mdtraj as md

    return md.compute_distances(ca, pairs, periodic=False)


def backbone_torsion_features(traj) -> np.ndarray:
    """Compute backbone φ/ψ torsions encoded as ``(cos, sin)`` pairs.

    The sine/cosine encoding avoids the ±π discontinuity of raw angles.

    :param traj: An ``mdtraj.Trajectory`` with a protein backbone.
    :return: Array of shape ``(n_frames, 2 * (n_phi + n_psi))``.
    :raises ValueError: If no backbone torsions are found."""
    import mdtraj as md

    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    angles = np.concatenate([phi, psi], axis=1)
    if angles.shape[1] == 0:
        raise ValueError("No backbone φ/ψ torsions found in the topology.")
    return np.concatenate([np.cos(angles), np.sin(angles)], axis=1)


FEATURES = {
    "ca_distances": ca_distance_features,
    "backbone_torsions": backbone_torsion_features,
}


def compute_features(
    traj, features: list[str], exclude_neighbors: int = 2
) -> np.ndarray:
    """Compute and concatenate the requested feature sets for a trajectory.

    :param traj: An ``mdtraj.Trajectory``.
    :param features: Feature names, each a key of :data:`FEATURES`.
    :param exclude_neighbors: Passed to :func:`ca_distance_features`.
    :return: Array of shape ``(n_frames, n_features)``.
    :raises KeyError: If a feature name is unknown."""
    blocks = []
    for name in features:
        if name not in FEATURES:
            raise KeyError(f"Unknown feature '{name}'. Choose from {list(FEATURES)}.")
        if name == "ca_distances":
            blocks.append(ca_distance_features(traj, exclude_neighbors))
        else:
            blocks.append(FEATURES[name](traj))
    return np.concatenate(blocks, axis=1)


def fit_tica(features: np.ndarray, lagtime: int, dim: int, scaling: str | None):
    """Fit a TICA model to feature trajectories.

    :param features: Array of shape ``(n_frames, n_features)``.
    :param lagtime: TICA lag time in frames.
    :param dim: Number of independent components to keep.
    :param scaling: deeptime scaling mode (``"kinetic_map"``, ``"commute_map"``
        or ``None``).
    :return: A fitted deeptime ``CovarianceKoopmanModel``.
    :raises ImportError: If deeptime is not installed."""
    try:
        from deeptime.decomposition import TICA
    except ImportError as e:
        raise ImportError(
            "Fitting a TICA model requires deeptime. Install the analysis extra: "
            "`uv sync --extra analysis`."
        ) from e
    tica = TICA(lagtime=lagtime, dim=dim, scaling=scaling)
    return tica.fit([features]).fetch_model()


def save_tica_model(model, path: str | Path) -> None:
    """Pickle a fitted TICA model for later reuse via :func:`load_tica_model`.

    :param model: A fitted deeptime ``CovarianceKoopmanModel``.
    :param path: Destination ``.pkl`` path."""
    import pickle

    Path(path).write_bytes(pickle.dumps(model))


def load_tica_model(path: str | Path):
    """Load a pickled deeptime TICA model saved by :func:`_save_model`.

    :param path: Path to a ``.pkl`` file holding a ``CovarianceKoopmanModel``.
    :return: The unpickled model, ready for ``.transform``."""
    import pickle

    return pickle.loads(Path(path).read_bytes())


def plot_free_energy(
    projections: list[np.ndarray],
    out_file: Path,
    labels: list[str | None] | None = None,
    f_max: float = 7.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot one or more TICA free-energy surfaces into a single PDF.

    With two or more components each projection becomes a contour panel (a shared
    colour bar across panels); with one component they are overlaid 1-D profiles.
    Free energy is ``-kT ln p`` in units of kT (kT = 1).

    :param projections: One projection array per panel, shape ``(n_frames, n_comp)``.
    :param out_file: Output path (``.pdf`` per the project's figure convention).
    :param labels: Optional per-projection labels (e.g. ``"lag = 1000"``).
    :param f_max: Free-energy ceiling in kT for the colour scale.
    :param xlim: Optional ``(min, max)`` limits for the x-axis (TIC 1).
    :param ylim: Optional ``(min, max)`` limits for the y-axis (TIC 2 for a
        contour panel, free energy for a 1-D profile)."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature"])
    except (ImportError, OSError):
        pass

    labels = labels or [None] * len(projections)
    n = len(projections)

    if projections[0].shape[1] >= 2:
        fig, axes = plt.subplots(
            1, n, figsize=(3.3 * n, 2.5), constrained_layout=True, squeeze=False
        )
        levels = np.linspace(0, f_max, 50)
        cf = None
        for ax, proj, label in zip(axes[0], projections, labels):
            xc, yc, F = free_energy_2d(
                proj[:, 0], proj[:, 1], 90, 2.0, f_max, x_range=xlim, y_range=ylim
            )
            cf = ax.contourf(xc, yc, F.T, cmap="jet", levels=levels)
            cf.set_rasterized(True)  # keep the vector PDF small
            ax.set_xlabel("TIC 1")
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            if label:
                ax.set_title(label)
        axes[0][0].set_ylabel("TIC 2")
        fig.colorbar(cf, ax=axes, label=r"Free energy $[k_{\mathrm{B}}T]$")
    else:
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        for proj, label in zip(projections, labels):
            H, edges = np.histogram(proj[:, 0], bins=90, density=True, range=xlim)
            F = np.where(H > 0, -np.log(H), np.nan)
            F -= np.nanmin(F)
            ax.plot(0.5 * (edges[:-1] + edges[1:]), F, label=label)
        ax.set_xlabel("TIC 1")
        ax.set_ylabel(r"Free energy $[k_{\mathrm{B}}T]$")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if any(labels):
            ax.legend()

    fig.savefig(out_file, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Entry point: featurise a trajectory and fit or project a TICA model."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdb", required=True, help="PDB providing the topology.")
    parser.add_argument(
        "--frames",
        required=True,
        nargs="+",
        help="One or more trajectory files (dcd/xtc/…) to analyse.",
    )
    parser.add_argument(
        "--features",
        required=True,
        nargs="+",
        choices=list(FEATURES),
        help="TICA features to compute and concatenate.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride on load.")
    parser.add_argument(
        "--exclude-neighbors",
        type=int,
        default=2,
        help="Sequence separation to skip for Cα distances (default 2).",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1000],
        help="TICA lag time(s) in frames (fitting). Passing several fits one model "
        "per lag and writes a multi-panel FES comparison.",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Number of independent components."
    )
    parser.add_argument(
        "--scaling",
        default="kinetic_map",
        choices=["kinetic_map", "commute_map", "none"],
        help="deeptime component scaling (default kinetic_map).",
    )
    parser.add_argument(
        "--tica-model",
        default=None,
        help="Preexisting pickled TICA model (.pkl); project via its .transform "
        "instead of fitting.",
    )
    parser.add_argument(
        "-o", "--out-dir", default=".", help="Directory for outputs (default cwd)."
    )
    parser.add_argument(
        "--prefix", default="tica", help="Filename prefix for outputs (default 'tica')."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Also save a free-energy surface figure."
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="x-axis limits (TIC 1) for the free-energy plot.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="y-axis limits (TIC 2 for a contour panel, free energy for a 1-D "
        "profile) for the free-energy plot.",
    )
    args = parser.parse_args()

    scaling = None if args.scaling == "none" else args.scaling
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = load_trajectory(args.pdb, args.frames, args.stride)
    features = compute_features(traj, args.features, args.exclude_neighbors)
    print(f"Featurised {features.shape[0]} frames × {features.shape[1]} features.")

    projections: list[np.ndarray] = []
    labels: list[str | None] = []

    if args.tica_model:
        model = load_tica_model(args.tica_model)
        projection = model.transform(features)[:, : args.dim]
        np.save(out_dir / f"{args.prefix}_projection.npy", projection)
        print(f"Projected via preexisting model → {projection.shape[1]} TICs.")
        projections, labels = [projection], [None]
    else:
        n_frames = features.shape[0]
        lags = [lag for lag in args.lags if lag < n_frames]
        for lag in args.lags:
            if lag >= n_frames:
                print(f"Skipping lag={lag}: trajectory has only {n_frames} frames.")
        if not lags:
            parser.error(
                f"No lag is smaller than the {n_frames} available frames; "
                "lower --lags or --stride."
            )
        multi = len(lags) > 1
        for lag in lags:
            model = fit_tica(features, lag, args.dim, scaling)
            projection = model.transform(features)
            prefix = f"{args.prefix}_lag{lag}" if multi else args.prefix
            save_tica_model(model, out_dir / f"{prefix}_model.pkl")
            np.save(out_dir / f"{prefix}_projection.npy", projection)
            print(f"Fitted TICA (lag={lag}, dim={args.dim}); saved model & projection.")
            projections.append(projection)
            labels.append(f"lag = {lag}" if multi else None)

    if args.plot:
        fig_file = out_dir / f"{args.prefix}_fes.pdf"
        plot_free_energy(projections, fig_file, labels, xlim=args.xlim, ylim=args.ylim)
        print(f"Wrote free-energy surface → {fig_file}")


if __name__ == "__main__":
    main()
