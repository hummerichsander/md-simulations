"""Give an existing trajectory's constrained bond lengths their thermal width, analytically.

A trajectory produced with ``constraints: hbonds`` has every bond to hydrogen frozen at exactly its
equilibrium length, so those coordinates carry no thermal variance at all -- measured std 0.40 pm
against an XTC quantization floor of 0.41. That is fatal for anything that reweights a generative
model against the all-atom potential: the model still generates them, while the constrained
Hamiltonian OpenMM builds has no bond term on them and so scores them not at all. A subspace where
the proposal is narrow and the target flat contributes ``d / 2`` nats squared to the variance of the
log-weights -- 8.1 nats for trp-cage's 136 bonds -- and no amount of training removes it.

``sampling/relax.py`` repairs this by integrating each frame briefly in the unconstrained
potential, and works: the per-bond width converges to the analytic one. But the distribution it is
converging *to* is known in closed form. The bond term OpenMM deleted is separable, and the
residual coupling of a bond length to everything else measures 650 against the bond's own
285000 kJ/mol/nm^2 -- 0.2% -- so

    d_i ~ N(r0_i, sqrt(kT / k_i))    independently per bond

to that accuracy. Drawing it directly is exact by construction rather than by convergence, costs no
dynamics at all, and moves nothing except the one coordinate that was broken: the hydrogen slides
along its existing bond axis, so every bond angle and torsion it participates in is preserved
exactly.

``r0`` and ``k`` are read off the ``HarmonicBondForce`` of the system built with
``constraints: none``, which is the only build that has them -- under ``hbonds`` those terms are
deleted. Which bonds to resample is taken from the *constrained* twin of the same config rather than
assumed to be "the hydrogens", so the set is exactly what the reference simulation froze.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import openmm
import openmm.unit as unit

from md_simulations.config import SimulationConfig, load_config, resolve_data_root
from md_simulations.core.logging import setup_logger
from md_simulations.engines import build_engine
from md_simulations.engines.base import OpenMMEngine

NM = unit.nanometer
KJ_PER_NM2 = unit.kilojoule_per_mole / unit.nanometer**2
BOLTZMANN = 8.314462618e-3  # kJ/mol/K


def _build(config: SimulationConfig, data_root: Path, logger: logging.Logger) -> openmm.System:
    """Build the OpenMM system for a configuration.

    :param config: The simulation configuration.
    :param data_root: Root the configuration's paths resolve against.
    :param logger: Logger handed to the engine.
    :return: The built OpenMM system."""
    engine = build_engine(config, data_root, logger)
    if not isinstance(engine, OpenMMEngine):
        raise NotImplementedError(f"resampling needs an OpenMM engine, got {config.engine!r}")

    return engine.build_system().system


def constrained_bond_parameters(
    config: SimulationConfig,
    data_root: Path,
    constrained_as: str = "hbonds",
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the bonds a reference simulation constrained, and their harmonic parameters.

    The set is read from the *constrained* twin of ``config`` rather than inferred from element
    names, so it is exactly what the reference simulation froze -- including under
    ``allbonds``. The force constants come from the unconstrained build, the only one that has
    them.

    :param config: An unconstrained configuration (``constraints: none``).
    :param data_root: Root the configuration's paths resolve against.
    :param constrained_as: The constraint scheme the reference trajectory was produced with.
    :param logger: Logger for progress messages.
    :return: The moving atom indices, their anchor atom indices, the equilibrium lengths in nm,
        and the Gaussian widths ``sqrt(kT / k)`` in nm."""
    if getattr(config, "constraints", None) != "none":
        raise ValueError(
            f"config has constraints={getattr(config, 'constraints', 'n/a')!r}; the equilibrium "
            "lengths and force constants live in the HarmonicBondForce, which OpenMM deletes for "
            "constrained bonds. Pass the `constraints: none` twin of the potential."
        )

    quiet = logging.getLogger("md_simulations.resample") if logger is None else logger
    flexible = _build(config, data_root, quiet)
    constrained = _build(
        config.model_copy(update={"constraints": constrained_as}), data_root, quiet
    )

    if flexible.getNumConstraints() != 0:
        # rigidWater is a separate createSystem argument, is not exposed by AmberConfig, and
        # defaults to True, so a solvated system keeps rigid water even under `constraints: none`.
        raise ValueError(
            f"the unconstrained build still carries {flexible.getNumConstraints()} constraints, so "
            "some bonds have no force constant to draw from. This is rigid water: the system needs "
            "rigidWater=False, which no config exposes yet."
        )

    harmonic = [f for f in flexible.getForces() if isinstance(f, openmm.HarmonicBondForce)]
    if not harmonic:
        raise ValueError("the unconstrained build has no HarmonicBondForce to read")
    springs = {}
    for i in range(harmonic[0].getNumBonds()):
        a, b, length, k = harmonic[0].getBondParameters(i)
        springs[frozenset((a, b))] = (length.value_in_unit(NM), k.value_in_unit(KJ_PER_NM2))

    kT = BOLTZMANN * config.temperature
    movers, anchors, r0, sigma = [], [], [], []
    for i in range(constrained.getNumConstraints()):
        a, b, _distance = constrained.getConstraintParameters(i)
        if (spring := springs.get(frozenset((a, b)))) is None:
            raise ValueError(
                f"constrained bond {a}-{b} has no HarmonicBondForce term in the unconstrained "
                "build, so there is nothing to draw from"
            )
        # the lighter atom moves; for a bond to hydrogen that is the hydrogen
        mass_a = constrained.getParticleMass(a).value_in_unit(unit.dalton)
        mass_b = constrained.getParticleMass(b).value_in_unit(unit.dalton)
        mover, anchor = (a, b) if mass_a < mass_b else (b, a)

        movers.append(mover)
        anchors.append(anchor)
        r0.append(spring[0])
        sigma.append(np.sqrt(kT / spring[1]))

    movers = np.array(movers, dtype=int)
    if len(np.unique(movers)) != len(movers):
        raise ValueError(
            "an atom moves in more than one constrained bond, so the draws would overwrite each "
            "other; resampling assumes each constrained bond has its own free end"
        )

    return movers, np.array(anchors, dtype=int), np.array(r0), np.array(sigma)


def resample_bond_lengths(
    positions: np.ndarray,
    movers: np.ndarray,
    anchors: np.ndarray,
    r0: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Redraw each named bond length from its Boltzmann distribution, keeping its direction.

    A pure radial move: only the moving atom's distance from its anchor changes, so every bond
    angle and torsion the pair takes part in survives exactly, and no other atom is touched.

    :param positions: Input coordinates, ``(F, N, 3)`` in nanometre.
    :param movers: Index of the atom to move, per bond.
    :param anchors: Index of the atom it is measured from, per bond.
    :param r0: Equilibrium length per bond, in nanometre.
    :param sigma: Gaussian width per bond, in nanometre.
    :param rng: Random generator for the draws.
    :return: The resampled coordinates, ``(F, N, 3)`` in nanometre."""
    out = positions.copy()

    axis = positions[:, movers] - positions[:, anchors]
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    lengths = r0 + rng.normal(0.0, sigma, size=(len(positions), len(movers)))
    out[:, movers] = positions[:, anchors] + lengths[..., None] * axis

    return out


def resample_trajectory(
    trajectory: str,
    topology: str,
    out: Path,
    movers: np.ndarray,
    anchors: np.ndarray,
    r0: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    chunk: int = 100_000,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Resample a trajectory file chunk by chunk, so memory stays flat in the trajectory length.

    Frames are drawn independently and a generator's draws do not depend on how they are split, so
    the output is bit-identical to resampling the whole trajectory at once.

    :param trajectory: Input trajectory path.
    :param topology: Topology of the input trajectory.
    :param out: Output trajectory path; the format follows its extension.
    :param movers: Index of the atom to move, per bond.
    :param anchors: Index of the atom it is measured from, per bond.
    :param r0: Equilibrium length per bond, in nanometre.
    :param sigma: Gaussian width per bond, in nanometre.
    :param rng: Random generator for the draws.
    :param start: First input frame to take.
    :param stop: Stop before this input frame, or ``None`` for the end.
    :param stride: Take every ``stride``-th input frame.
    :param chunk: Number of frames held in memory at once.
    :return: The number of frames written, and the per-bond length std before and after, in nm."""
    import mdtraj as md
    from mdtraj.formats import TRRTrajectoryFile, XTCTrajectoryFile
    from mdtraj.utils import in_units_of

    top = md.load_topology(topology)
    if (highest := max(movers.max(), anchors.max())) >= top.n_atoms:
        raise ValueError(
            f"the config indexes atom {highest} but the trajectory has only {top.n_atoms}; the "
            "config and the trajectory are not the same system"
        )

    remaining = None if stop is None else len(range(start, stop, stride))
    n = 0
    # moments of d - r0 rather than d, so the frozen input's ~1e-6 nm width survives float64
    moments = np.zeros((4, len(movers)))
    with md.open(str(out), "w") as f:
        for traj in md.iterload(trajectory, chunk=chunk, top=top, skip=start, stride=stride):
            if remaining is not None:
                traj = traj[:remaining]
                remaining -= len(traj)

            positions = traj.xyz.astype(np.float64)
            resampled = resample_bond_lengths(positions, movers, anchors, r0, sigma, rng)

            before = np.linalg.norm(positions[:, movers] - positions[:, anchors], axis=-1) - r0
            after = np.linalg.norm(resampled[:, movers] - resampled[:, anchors], axis=-1) - r0
            moments += [before.sum(0), (before**2).sum(0), after.sum(0), (after**2).sum(0)]
            n += len(traj)

            xyz = in_units_of(resampled.astype(np.float32), "nanometers", f.distance_unit)
            if isinstance(f, XTCTrajectoryFile | TRRTrajectoryFile):
                f.write(xyz, time=traj.time)
            else:
                f.write(xyz)

            if remaining == 0:
                break

    if n == 0:
        raise ValueError(f"no frames selected from {trajectory}")
    mean = moments / n
    return n, np.sqrt(mean[1] - mean[0] ** 2), np.sqrt(mean[3] - mean[2] ** 2)


def main() -> None:
    """Entry point for the ``md-sim-resample-hbonds`` command."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Unconstrained config defining the potential to draw from.")
    parser.add_argument("--trajectory", required=True, help="Input trajectory to resample.")
    parser.add_argument("--topology", required=True, help="Topology of the input trajectory.")
    parser.add_argument("-o", "--out", required=True, type=Path, help="Output trajectory path.")
    parser.add_argument(
        "--constrained-as",
        default="hbonds",
        choices=("hbonds", "allbonds"),
        help="The constraint scheme the input trajectory was produced with.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Take every stride-th input frame.")
    parser.add_argument("--start", type=int, default=0, help="First input frame to take.")
    parser.add_argument("--stop", type=int, default=None, help="Stop before this input frame.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the bond-length draws.")
    parser.add_argument("--chunk", type=int, default=100_000, help="Frames held in memory at once.")
    parser.add_argument(
        "--data-root", default=None, help="Root for the config's paths (overrides MD_DATA_ROOT)."
    )
    args = parser.parse_args()

    # Checked here rather than left to mdtraj, whose complaint about an unsupported "" topology
    # format names neither the offending argument nor the empty string that produced it.
    for name, label in (
        ("config", "config"),
        ("trajectory", "--trajectory"),
        ("topology", "--topology"),
    ):
        path = getattr(args, name)
        if not path:
            parser.error(f"{label} is empty; check for an unset variable in the command line")
        if not Path(path).is_file():
            parser.error(f"{label} does not exist: {path}")

    config = load_config(args.config)
    data_root = resolve_data_root(config, args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("md_simulations", args.out.parent, getattr(logging, config.log_level))
    logger.info(f"Config:     {args.config} (constraints={getattr(config, 'constraints', 'n/a')})")
    logger.info(f"Trajectory: {args.trajectory}")
    logger.info(f"Output:     {args.out}")

    movers, anchors, r0, sigma = constrained_bond_parameters(
        config, data_root, args.constrained_as, logger
    )
    logger.info(
        f"Resampling {len(movers)} bonds constrained as {args.constrained_as!r} at "
        f"{config.temperature} K: width {sigma.min() * 1000:.2f}-{sigma.max() * 1000:.2f} pm."
    )

    n, before, achieved = resample_trajectory(
        args.trajectory,
        args.topology,
        args.out,
        movers,
        anchors,
        r0,
        sigma,
        np.random.default_rng(args.seed),
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        chunk=args.chunk,
    )

    # Exact by construction, so a ratio away from 1 means a bug -- the wrong atom moved, or the
    # wrong width -- rather than a run that needs to be longer.
    logger.info(
        f"Bond width {before.mean() * 1000:.3f} -> {achieved.mean() * 1000:.3f} pm "
        f"(target {sigma.mean() * 1000:.3f}, ratio {(achieved / sigma).mean():.3f})."
    )
    logger.info(f"Wrote {n} resampled frames to {args.out}.")


if __name__ == "__main__":
    main()
