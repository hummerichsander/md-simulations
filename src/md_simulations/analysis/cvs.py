#!/usr/bin/env python3
"""Collective-variable free-energy surfaces of an MD trajectory.

Computes structural collective variables (CVs) frame-by-frame — RMSD to a
native structure, fraction of native contacts (Best–Hummer ``Q``), radius of
gyration, end-to-end distance — and renders them as smoothed 2-D free-energy
surfaces. With more than two CVs every pairwise projection is drawn as its own
panel in one figure.

Featurisation depends only on ``mdtraj``/``numpy`` (core deps); the free-energy
histogram additionally needs ``scipy`` and plotting needs ``matplotlib`` (the
``analysis``/``viz`` extras).
"""

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np

from md_simulations.analysis.fes import (
    drop_spurious_box,
    free_energy_2d,
    load_trajectory,
)


def rmsd_to_native(
    traj, native, *, selection: str = "name CA", periodic: bool = False
) -> np.ndarray:
    """Root-mean-square deviation to a native structure after superposition.

    :param traj: An ``mdtraj.Trajectory``.
    :param native: Single-frame reference ``mdtraj.Trajectory`` (matching topology).
    :param selection: Atom selection to fit and measure on (default Cα).
    :param periodic: Unused; accepted for a uniform CV signature.
    :return: Array of shape ``(n_frames,)`` of RMSD values in nm."""
    import mdtraj as md

    idx = traj.top.select(selection)
    if len(idx) == 0:
        raise ValueError(f"Selection '{selection}' matched no atoms for RMSD.")
    return md.rmsd(traj, native, atom_indices=idx)


def fraction_native_contacts(
    traj,
    native,
    *,
    native_cutoff: float = 0.45,
    min_seq_sep: int = 3,
    beta: float = 50.0,
    lam: float = 1.8,
    periodic: bool = False,
) -> np.ndarray:
    """Best–Hummer fraction of native contacts ``Q``.

    Native contacts are heavy-atom pairs whose residues are at least
    ``min_seq_sep`` apart in sequence and closer than ``native_cutoff`` in the
    reference. Each contact contributes a smooth switching function
    ``1 / (1 + exp[beta (r - lam r0)])`` (Best, Hummer & Eaton, PNAS 2013).

    :param traj: An ``mdtraj.Trajectory``.
    :param native: Single-frame reference ``mdtraj.Trajectory`` (matching topology).
    :param native_cutoff: Contact distance cutoff in the reference, in nm.
    :param min_seq_sep: Minimum residue-index separation for a contact.
    :param beta: Switching-function sharpness in nm⁻¹.
    :param lam: Tolerance factor applied to each native distance.
    :param periodic: Apply the minimum-image convention to distances.
    :return: Array of shape ``(n_frames,)`` of ``Q`` in ``[0, 1]``.
    :raises ValueError: If the reference has no native contacts."""
    import mdtraj as md

    heavy = native.top.select_atom_indices("heavy")
    resid = np.array([native.top.atom(i).residue.index for i in heavy])
    pairs = np.array(
        [
            (heavy[a], heavy[b])
            for a in range(len(heavy))
            for b in range(a + 1, len(heavy))
            if abs(resid[a] - resid[b]) >= min_seq_sep
        ]
    )
    if len(pairs) == 0:
        raise ValueError(
            f"No heavy-atom pairs with residue separation ≥ {min_seq_sep}; lower --q-min-seq-sep."
        )
    r0 = md.compute_distances(native, pairs, periodic=False)[0]
    contacts = pairs[r0 < native_cutoff]
    if len(contacts) == 0:
        raise ValueError(
            f"No native contacts within {native_cutoff} nm in the reference; "
            "raise --q-cutoff or check the native structure."
        )
    r = md.compute_distances(traj, contacts, periodic=periodic)
    r0_contacts = r0[r0 < native_cutoff]
    q = 1.0 / (1.0 + np.exp(beta * (r - lam * r0_contacts)))
    return q.mean(axis=1)


def radius_of_gyration(traj, native=None, *, periodic: bool = False) -> np.ndarray:
    """Mass-weighted radius of gyration.

    :param traj: An ``mdtraj.Trajectory``.
    :param native: Unused; accepted for a uniform CV signature.
    :param periodic: Unused; accepted for a uniform CV signature.
    :return: Array of shape ``(n_frames,)`` in nm."""
    import mdtraj as md

    return md.compute_rg(traj)


def end_to_end_distance(traj, native=None, *, periodic: bool = False) -> np.ndarray:
    """Distance between the first and last Cα atom.

    :param traj: An ``mdtraj.Trajectory``.
    :param native: Unused; accepted for a uniform CV signature.
    :param periodic: Apply the minimum-image convention to the distance.
    :return: Array of shape ``(n_frames,)`` in nm.
    :raises ValueError: If fewer than two Cα atoms are present."""
    import mdtraj as md

    ca = traj.top.select("name CA")
    if len(ca) < 2:
        raise ValueError(f"Need at least two Cα atoms for end-to-end, found {len(ca)}.")
    pair = np.array([[ca[0], ca[-1]]])
    return md.compute_distances(traj, pair, periodic=periodic)[:, 0]


@dataclass(frozen=True)
class CV:
    """Metadata for a collective variable."""

    func: Callable
    label: str
    needs_native: bool


CVS: dict[str, CV] = {
    "rmsd": CV(rmsd_to_native, "RMSD to native [nm]", needs_native=True),
    "native_contacts": CV(
        fraction_native_contacts, r"Fraction of native contacts $Q$", needs_native=True
    ),
    "rg": CV(radius_of_gyration, "Radius of gyration [nm]", needs_native=False),
    "end_to_end": CV(end_to_end_distance, "End-to-end distance [nm]", needs_native=False),
}


def compute_cvs(
    traj,
    native,
    cvs: list[str],
    *,
    periodic: bool = False,
    q_native_cutoff: float = 0.45,
    q_min_seq_sep: int = 3,
) -> dict[str, np.ndarray]:
    """Compute the requested collective variables for a trajectory.

    :param traj: An ``mdtraj.Trajectory``.
    :param native: Single-frame reference trajectory, or ``None`` if no CV needs it.
    :param cvs: CV names, each a key of :data:`CVS`.
    :param periodic: Apply the minimum-image convention to distance-based CVs.
    :param q_native_cutoff: Native-contact cutoff (nm) for ``native_contacts``.
    :param q_min_seq_sep: Minimum residue separation for ``native_contacts``.
    :return: Mapping of CV name to an array of shape ``(n_frames,)``.
    :raises KeyError: If a CV name is unknown.
    :raises ValueError: If a native-requiring CV is asked for without a reference."""
    out: dict[str, np.ndarray] = {}
    for name in cvs:
        if name not in CVS:
            raise KeyError(f"Unknown CV '{name}'. Choose from {list(CVS)}.")
        spec = CVS[name]
        if spec.needs_native and native is None:
            raise ValueError(f"CV '{name}' needs a native structure (--native).")
        match name:
            case "native_contacts":
                out[name] = fraction_native_contacts(
                    traj,
                    native,
                    native_cutoff=q_native_cutoff,
                    min_seq_sep=q_min_seq_sep,
                    periodic=periodic,
                )
            case _:
                out[name] = spec.func(traj, native, periodic=periodic)
    return out


def plot_cv_free_energy(
    cvs: dict[str, np.ndarray],
    out_file: Path,
    f_max: float = 7.0,
    bins: int = 90,
    sigma: float = 2.0,
) -> None:
    """Plot every pairwise 2-D free-energy surface of the given CVs into one PDF.

    A single panel is drawn for two CVs; with more, every ``C(n, 2)`` pair gets
    its own contour panel arranged in a grid, sharing one colour bar. Free energy
    is ``-kT ln p`` in units of kT (kT = 1).

    :param cvs: Mapping of CV name to an array of shape ``(n_frames,)``; labels
        come from :data:`CVS` when the name is known, else the name itself.
    :param out_file: Output path (``.pdf`` per the project's figure convention).
    :param f_max: Free-energy ceiling in kT for the colour scale.
    :param bins: Histogram bins per axis.
    :param sigma: Gaussian smoothing width in bins.
    :raises ValueError: If fewer than two CVs are supplied."""
    import matplotlib.pyplot as plt

    if len(cvs) < 2:
        raise ValueError("Need at least two CVs for a 2-D free-energy surface.")

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature"])
    except (ImportError, OSError):
        pass

    names = list(cvs)
    labels = {n: (CVS[n].label if n in CVS else n) for n in names}
    pairs = list(combinations(names, 2))
    ncols = min(3, len(pairs))
    nrows = -(-len(pairs) // ncols)  # ceil division

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.3 * ncols, 2.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    flat = axes.ravel()
    levels = np.linspace(0, f_max, 50)
    cf = None
    for ax, (xn, yn) in zip(flat, pairs):
        xc, yc, F = free_energy_2d(cvs[xn], cvs[yn], bins, sigma, f_max)
        cf = ax.contourf(xc, yc, F.T, cmap="jet", levels=levels)
        cf.set_rasterized(True)  # keep the vector PDF small
        ax.set_xlabel(labels[xn])
        ax.set_ylabel(labels[yn])
    for ax in flat[len(pairs) :]:
        ax.set_visible(False)
    fig.colorbar(cf, ax=axes, label=r"Free energy $[k_{\mathrm{B}}T]$")

    fig.savefig(out_file, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Entry point: compute collective variables and plot their 2-D FES."""
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
        "--native",
        default=None,
        help="Reference structure for RMSD / native contacts (defaults to --pdb).",
    )
    parser.add_argument(
        "--cvs",
        required=True,
        nargs="+",
        choices=list(CVS),
        help="Collective variables to compute (need ≥ 2 for a 2-D surface).",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride on load.")
    parser.add_argument(
        "--q-cutoff",
        type=float,
        default=0.45,
        help="Native-contact distance cutoff in nm (default 0.45).",
    )
    parser.add_argument(
        "--q-min-seq-sep",
        type=int,
        default=3,
        help="Minimum residue separation for native contacts (default 3).",
    )
    parser.add_argument("-o", "--out-dir", default=".", help="Directory for outputs (default cwd).")
    parser.add_argument(
        "--prefix", default="cv", help="Filename prefix for outputs (default 'cv')."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Also save the free-energy surface figure."
    )
    parser.add_argument(
        "--f-max", type=float, default=7.0, help="Free-energy ceiling in kT (default 7)."
    )
    args = parser.parse_args()

    if len(args.cvs) < 2:
        parser.error("Need at least two --cvs for a 2-D free-energy surface.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = load_trajectory(args.pdb, args.frames, args.stride)
    drop_spurious_box(traj)
    periodic = traj.unitcell_lengths is not None

    native = None
    if any(CVS[name].needs_native for name in args.cvs):
        import mdtraj as md

        native = md.load(str(args.native or args.pdb), top=str(args.pdb))[0]

    cvs = compute_cvs(
        traj,
        native,
        args.cvs,
        periodic=periodic,
        q_native_cutoff=args.q_cutoff,
        q_min_seq_sep=args.q_min_seq_sep,
    )
    print(f"Computed {len(cvs)} CV(s) over {traj.n_frames} frames: {', '.join(cvs)}.")

    np.savez(out_dir / f"{args.prefix}_cvs.npz", **cvs)

    if args.plot:
        fig_file = out_dir / f"{args.prefix}_fes.pdf"
        plot_cv_free_energy(cvs, fig_file, f_max=args.f_max)
        print(f"Wrote free-energy surface(s) → {fig_file}")


if __name__ == "__main__":
    main()
