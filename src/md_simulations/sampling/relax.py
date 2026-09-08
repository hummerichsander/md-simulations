"""Per-frame relaxation of an existing trajectory in an unconstrained potential.

A trajectory produced with ``constraints: hbonds`` has its bonds to hydrogen frozen at exactly
their equilibrium length, so those coordinates carry no thermal variance at all. That is fatal for
anything that reweights a generative model against the all-atom potential: the model still has to
generate them, while the constrained Hamiltonian OpenMM builds has no bond term on them and so
scores them not at all. The mismatch adds ``d / 2`` nats squared to the variance of the log-weights
(~8 nats for trp-cage's 136 bonds) and no amount of training removes it.

This tool re-thermalises those degrees of freedom without re-running the simulation. Each input
frame is taken as a starting point, given fresh Maxwell-Boltzmann velocities, and integrated for a
short time in the *unconstrained* potential. The slow degrees of freedom are already correctly
distributed, so they need not move; only the stiff bonds have to find their thermal width.

Two things decide whether the output is usable, and both must be checked rather than assumed:

- **The relaxation has to be long enough.** A harmonic mode started at its minimum with a thermal
  velocity has width ``sigma * |sin(omega t)|``, so a short burst gives a phase-dependent width
  averaging ``sigma / sqrt(2)`` -- half the problem, still there. An X-H stretch period is ~11 fs,
  while thermalisation runs off the Langevin bath (~1 ps at ``friction = 1.0``). Scan
  ``--relax-time`` and compare the resulting per-bond width against ``sqrt(kT / k)``.
- **The timestep has to resolve the now-unconstrained stretch.** 2 fs does not; use ~0.5 fs.

The energy is deliberately never minimised: minimising would collapse the ensemble onto the
potential's minimum, which is the same delta distribution the constraint produced, only in a
different place.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import openmm
import openmm.unit as unit
from openmm import app
from tqdm import tqdm

from md_simulations.config import load_config, resolve_data_root
from md_simulations.core.integrators import create_integrator
from md_simulations.core.logging import setup_logger
from md_simulations.engines import build_engine
from md_simulations.engines.base import OpenMMEngine


def frame_indices(n_frames: int, start: int, stop: int | None, stride: int) -> np.ndarray:
    """Resolve the selected frames of a trajectory to absolute indices.

    :param n_frames: Total number of frames in the trajectory.
    :param start: First frame to take.
    :param stop: One past the last frame to take, or ``None`` for the end.
    :param stride: Take every ``stride``-th frame.
    :return: The selected absolute frame indices."""
    return np.arange(n_frames)[start : n_frames if stop is None else stop : stride]


def relax_frames(
    positions: np.ndarray,
    system: openmm.System,
    topology: app.Topology,
    temperature: float,
    friction: float,
    timestep: float,
    relax_time: float,
    platform: str | None = None,
    seed: int | None = None,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Relax each frame independently by a short burst of unconstrained dynamics.

    One :class:`openmm.app.Simulation` is built and reused across frames -- the context is the
    expensive object, and nothing about the system changes between frames.

    :param positions: Input coordinates, ``(F, N, 3)`` in nanometre.
    :param system: The OpenMM system to integrate in; must carry no constraints.
    :param topology: The matching topology.
    :param temperature: Temperature in Kelvin, for both the bath and the initial velocities.
    :param friction: Langevin friction coefficient in ps^-1.
    :param timestep: Integration timestep in picoseconds.
    :param relax_time: Time to integrate each frame for, in picoseconds.
    :param platform: OpenMM platform name, or ``None`` for the default.
    :param seed: Seed for the velocity draw, for reproducibility.
    :param logger: Logger for progress messages.
    :return: The relaxed coordinates, ``(F, N, 3)`` in nanometre."""
    if system.getNumConstraints() != 0:
        raise ValueError(
            f"the system carries {system.getNumConstraints()} constraints; relaxing under them "
            "leaves the constrained bond lengths exactly where they were. Build the potential "
            "with `constraints: none`."
        )

    n_steps = int(round(relax_time / timestep))
    if n_steps < 1:
        raise ValueError(f"relax_time {relax_time} ps is shorter than one {timestep} ps step")

    integrator = create_integrator(temperature, friction, timestep, kind="langevin_middle")
    if seed is not None:
        integrator.setRandomNumberSeed(seed)

    simulation = (
        app.Simulation(topology, system, integrator, openmm.Platform.getPlatformByName(platform))
        if platform is not None
        else app.Simulation(topology, system, integrator)
    )
    if logger is not None:
        logger.info(
            f"Relaxing {len(positions)} frames for {relax_time} ps each "
            f"({n_steps} steps of {timestep * 1000:.2f} fs) at {temperature} K on "
            f"{simulation.context.getPlatform().getName()}."
        )

    out = np.empty_like(positions)
    for i, x in enumerate(tqdm(positions, desc="Relaxing", unit="frame")):
        simulation.context.setPositions(x * unit.nanometer)
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
        integrator.step(n_steps)
        state = simulation.context.getState(getPositions=True)
        out[i] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    return out


def main() -> None:
    """Entry point for the ``md-sim-relax`` command."""
    import mdtraj as md

    from md_simulations.analysis.fes import drop_spurious_box

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Simulation config defining the unconstrained potential.")
    parser.add_argument("--trajectory", required=True, help="Input trajectory to relax.")
    parser.add_argument("--topology", required=True, help="Topology of the input trajectory.")
    parser.add_argument("-o", "--out", required=True, type=Path, help="Output trajectory path.")
    parser.add_argument(
        "--relax-time", type=float, default=2.0, help="Relaxation time per frame (ps)."
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.0005,
        help="Integration timestep (ps); must resolve the unconstrained X-H stretch.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Take every stride-th input frame.")
    parser.add_argument("--start", type=int, default=0, help="First input frame to take.")
    parser.add_argument("--stop", type=int, default=None, help="Stop before this input frame.")
    parser.add_argument("--platform", default=None, help="OpenMM platform (CPU, CUDA, ...).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the velocity draws.")
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

    engine = build_engine(config, data_root, logger)
    if not isinstance(engine, OpenMMEngine):
        raise NotImplementedError(f"relaxation needs an OpenMM engine, got {config.engine!r}")
    built = engine.build_system()

    traj = drop_spurious_box(md.load(args.trajectory, top=args.topology))
    selection = frame_indices(traj.n_frames, args.start, args.stop, args.stride)
    logger.info(f"Selected {len(selection)} of {traj.n_frames} frames.")

    if built.system.getNumParticles() != traj.n_atoms:
        raise ValueError(
            f"the config builds {built.system.getNumParticles()} particles but the trajectory has "
            f"{traj.n_atoms} atoms"
        )

    relaxed = relax_frames(
        traj.xyz[selection].astype(np.float64),
        built.system,
        built.topology,
        temperature=config.temperature,
        friction=config.friction,
        timestep=args.timestep,
        relax_time=args.relax_time,
        platform=args.platform,
        seed=args.seed,
        logger=logger,
    )

    out_traj = md.Trajectory(relaxed.astype(np.float32), traj.topology)
    out_traj.time = traj.time[selection]
    out_traj.save(str(args.out))
    logger.info(f"Wrote {out_traj.n_frames} relaxed frames to {args.out}.")


if __name__ == "__main__":
    main()
