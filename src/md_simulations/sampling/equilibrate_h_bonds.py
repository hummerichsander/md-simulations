"""Per-frame equilibration of constrained bonds, in the unconstrained potential at fixed heavy atoms.

A trajectory produced with ``constraints: hbonds`` has every bond to hydrogen frozen at exactly its
equilibrium length, so those coordinates carry no thermal variance at all. That is fatal for
anything that reweights a generative model against the all-atom potential: the model still has to
generate them, while the constrained Hamiltonian OpenMM builds has no bond term on them and so
scores them not at all.

``sampling/resample_h_bonds.py`` repairs this analytically, by drawing each length from
``N(r0, sqrt(kT / k))`` and sliding the hydrogen along its own bond axis. That draw treats the bond
as an isolated harmonic oscillator -- exact only to the extent that the length decouples from
everything else. It does not: the measured residual coupling of 650 against the bond's own
285000 kJ/mol/nm^2 is small but not zero, and the chignolin fiber pilot measured an X-H width 3 %
narrower under flexible dynamics than the independent-harmonic draw predicts. The *conditional*
distribution of the hydrogens at a given heavy-atom configuration,

    p(x_H | x_heavy)  ~  exp(-U(x_H, x_heavy) / kT),

is what the all-atom potential actually scores, and it knows about the angles, the torsions and
the nonbonded terms that the per-bond draw drops.

This module samples that conditional directly. Each input frame is taken as a starting point, all
atoms except the constrained bonds' free ends are given zero mass, and the free ends are integrated
in the *unconstrained* potential until their distribution has converged. Zero mass takes an atom
out of the equations of motion exactly, so the heavy atoms come back bit-identical and the output
frame is the input frame with its hydrogens repaired -- same frame order, same coarse-grained
beads, same slow coordinates. Only the hydrogens, and anything derived from them (energies,
forces, scores) change.

Which atoms are free is taken from the *constrained twin* of the same config -- the free end of
every bond the reference simulation actually froze -- rather than assumed to be "the hydrogens",
so the set is exactly the broken one, under ``allbonds`` as much as under ``hbonds``. This is the
same selection :mod:`md_simulations.sampling.resample_h_bonds` redraws.

Three things decide whether the output is usable, and all three must be checked rather than
assumed:

- **The equilibration has to be long enough.** A harmonic mode started at its minimum with a
  thermal velocity has width ``sigma * |sin(omega t)|``, so a short burst gives a phase-dependent
  width averaging ``sigma / sqrt(2)`` -- half the problem, still there. An X-H stretch period is
  ~11 fs, while thermalisation runs off the Langevin bath (~1 ps at ``friction = 1.0``), and with
  the heavy atoms frozen it runs *only* off the bath: the earlier free-atom variant reached 0.986
  of the analytic width at 2 ps where the frozen one reached 0.930, because heavy-atom motion is
  itself a channel to the bath. Scan ``--time`` and stop where the per-bond width stops moving.
  Convergence is a plateau in ``--time``, not a ratio to ``sqrt(kT / k)``: that analytic width is
  the thing being corrected, so matching it would mean the run had achieved nothing.
- **Raising the friction is the cheap way to get there.** The stationary distribution of Langevin
  dynamics does not depend on the friction, only the speed of approach to it does, so
  ``--friction`` above the config's own value buys convergence per step and biases nothing. Too
  high turns the hydrogens overdamped and slows the hydroxyl and methyl rotations instead, so this
  too is scanned rather than assumed.
- **The timestep has to resolve the now-unconstrained stretch.** 2 fs does not; use ~0.5 fs.

The energy is deliberately never minimised: minimising would collapse the hydrogens onto the
potential's minimum, which is the same delta distribution the constraint produced, only in a
different place.

The fully flexible variant this module replaces (``sampling/relax.py``, which moved every atom)
is gone, and the measurement that retired it is worth keeping: on trp-cage, heavy-atom RMSD from
the input frame reached 69 pm after 0.5 ps, 91 pm after 1 ps and 120 pm after 2 ps -- the duration
the bond widths need. Its output was therefore a *nearby* configuration rather than the input frame
with its hydrogens repaired, which broke the frame-level correspondence with everything derived
from the same trajectory. Freezing costs a longer equilibration and buys that correspondence back.
"""

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import openmm
import openmm.unit as unit
from openmm import app
from tqdm import tqdm

from md_simulations.config import (
    FrozenAtomsConfig,
    SimulationConfig,
    load_config,
    resolve_data_root,
)
from md_simulations.core.freezing import freeze_atoms
from md_simulations.core.integrators import create_integrator
from md_simulations.core.logging import setup_logger
from md_simulations.core.tracking import record_outputs
from md_simulations.engines import build_engine
from md_simulations.engines.base import BuiltSystem, OpenMMEngine
from md_simulations.sampling.resample_h_bonds import constrained_bond_parameters


def build_frozen_system(
    config: SimulationConfig,
    data_root: Path,
    mobile: np.ndarray,
    logger: logging.Logger,
) -> BuiltSystem:
    """Build the unconstrained system with every atom outside *mobile* frozen.

    :param config: An unconstrained configuration (``constraints: none``).
    :param data_root: Root the configuration's paths resolve against.
    :param mobile: Indices of the atoms left free.
    :param logger: Logger handed to the engine.
    :return: The build, its frozen atoms at zero mass. Its ``positions`` are the config's input
        structure, not the trajectory's -- useful as a reference frame, not as a starting point.
    :raises NotImplementedError: If the config names a non-OpenMM engine."""
    engine = build_engine(config, data_root, logger)
    if not isinstance(engine, OpenMMEngine):
        raise NotImplementedError(f"equilibration needs an OpenMM engine, got {config.engine!r}")
    built = engine.build_system()

    free = set(int(i) for i in mobile)
    frozen = [i for i in range(built.system.getNumParticles()) if i not in free]
    if not frozen:
        raise ValueError("every atom is a constrained-bond end; there is nothing to condition on")

    # freeze_atoms also drops the centre-of-mass motion remover, and refuses up front any
    # constraint touching a frozen atom -- which `constraints: none` has already ruled out, but
    # rigid water would not.
    freeze_atoms(built.system, built.topology, FrozenAtomsConfig(indices=frozen), logger)

    return built


def equilibrate_frames(
    positions: np.ndarray,
    system: openmm.System,
    topology: app.Topology,
    mobile: np.ndarray,
    temperature: float,
    timestep: float,
    equilibration_time: float,
    friction: float | None = None,
    platform: str | None = None,
    seed: int = 42,
    frame_seeds: np.ndarray | None = None,
    return_energies: bool = False,
    simulation: app.Simulation | None = None,
    logger: logging.Logger | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Equilibrate the mobile atoms of each frame independently, at fixed frozen atoms.

    One :class:`openmm.app.Simulation` is built and reused across frames -- the context is the
    expensive object, and nothing about the system changes between frames. Both the integrator and
    the per-frame velocity draw are seeded, so a given ``seed`` reproduces the output exactly; the
    velocity seed is offset per frame so that two frames do not share a draw.

    :param positions: Input coordinates, ``(F, N, 3)`` in nanometre.
    :param system: The OpenMM system to integrate in; must carry no constraints, and must already
        have its frozen atoms at zero mass.
    :param topology: The matching topology.
    :param mobile: Indices of the atoms left free, i.e. the ones that may change.
    :param temperature: Temperature in Kelvin, for both the bath and the initial velocities.
    :param timestep: Integration timestep in picoseconds.
    :param equilibration_time: Time to integrate each frame for, in picoseconds.
    :param friction: Langevin friction coefficient in ps^-1; needed only when *simulation* is
        ``None`` and one has to be built here.
    :param platform: OpenMM platform name, or ``None`` for the default.
    :param seed: Seed for the integrator and, unless *frame_seeds* is given, the velocity draws.
    :param frame_seeds: One velocity seed per frame, overriding the default ``seed + 1 + i``. A
        sharded run passes the *absolute* frame index here, so that two shards -- which both start
        counting at zero -- cannot draw the same velocities for different frames.
    :param return_energies: Also return the potential energy of every frame before and after,
        which costs one extra energy evaluation per frame and nothing else.
    :param simulation: A simulation to reuse across calls; built here when ``None``.
    :param logger: Logger for progress messages.
    :return: The equilibrated coordinates, ``(F, N, 3)`` in nanometre -- and, with
        *return_energies*, the ``(F, 2)`` potential energies in kJ/mol, input then output. Frozen
        atoms are copied from the input rather than read back, so they are bit-identical to it."""
    if system.getNumConstraints() != 0:
        raise ValueError(
            f"the system carries {system.getNumConstraints()} constraints; equilibrating under "
            "them leaves the constrained bond lengths exactly where they were. Build the "
            "potential with `constraints: none`."
        )

    n_steps = int(round(equilibration_time / timestep))
    if n_steps < 1:
        raise ValueError(
            f"equilibration time {equilibration_time} ps is shorter than one {timestep} ps step"
        )

    if simulation is None:
        if friction is None:
            raise ValueError("give either a `simulation` to reuse or a `friction` to build one")
        simulation = build_simulation(
            system, topology, temperature, friction, timestep, platform, seed
        )
    if logger is not None:
        logger.info(
            f"Equilibrating {len(positions)} frames for {equilibration_time} ps each "
            f"({n_steps} steps of {timestep * 1000:.2f} fs) at {temperature} K on "
            f"{simulation.context.getPlatform().getName()}."
        )

    # The frozen atoms never move, so the output starts as a copy of the input and only the mobile
    # rows are overwritten. Reading them back would be equivalent but for float32 round-tripping.
    if frame_seeds is None:
        frame_seeds = seed + 1 + np.arange(len(positions))
    elif len(frame_seeds) != len(positions):
        raise ValueError(f"got {len(frame_seeds)} frame seeds for {len(positions)} frames")

    out = positions.copy()
    energies = np.empty((len(positions), 2))
    for i, x in enumerate(tqdm(positions, desc="Equilibrating", unit="frame")):
        simulation.context.setPositions(x * unit.nanometer)
        # One extra energy evaluation per frame, against the thousands of steps that follow, for
        # the one number that says what the equilibration bought: how far up the potential the
        # input's hydrogens sat.
        energies[i, 0] = (
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        simulation.context.setVelocitiesToTemperature(
            temperature * unit.kelvin, int(frame_seeds[i])
        )
        simulation.integrator.step(n_steps)
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        out[i, mobile] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[mobile]
        energies[i, 1] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    return (out, energies) if return_energies else out


def build_simulation(
    system: openmm.System,
    topology: app.Topology,
    temperature: float,
    friction: float,
    timestep: float,
    platform: str | None = None,
    seed: int = 42,
) -> app.Simulation:
    """Build the reusable simulation the frames are equilibrated in.

    :param system: The OpenMM system, frozen atoms already at zero mass.
    :param topology: The matching topology.
    :param temperature: Temperature in Kelvin.
    :param friction: Langevin friction coefficient in ps^-1.
    :param timestep: Integration timestep in picoseconds.
    :param platform: OpenMM platform name, or ``None`` for the default.
    :param seed: Seed for the integrator.
    :return: The simulation."""
    integrator = create_integrator(temperature, friction, timestep, kind="langevin_middle")
    integrator.setRandomNumberSeed(seed)
    if platform is not None:
        return app.Simulation(
            topology, system, integrator, openmm.Platform.getPlatformByName(platform)
        )
    return app.Simulation(topology, system, integrator)


def bond_groups(
    topology: app.Topology, movers: np.ndarray, anchors: np.ndarray
) -> dict[str, np.ndarray]:
    """Group the constrained bonds by element pair, e.g. ``N-H``, ``C-H``, ``O-H``.

    The mean over all 61 of chignolin's X-H bonds hides the part that matters: a C-H stretch sees
    almost nothing but its own bond, while an O-H or N-H also sees the hydrogen bond it donates,
    and those are the bonds whose width the independent-harmonic draw gets wrong. Reporting them
    apart makes that visible while the job runs.

    :param topology: The built topology, whose atom order the indices refer to.
    :param movers: Index of the free end, per bond.
    :param anchors: Index of the atom it is measured from, per bond.
    :return: Bond indices per ``<anchor element>-<mover element>`` label, label-sorted."""
    atoms = list(topology.atoms())
    groups: dict[str, list[int]] = {}
    for i, (mover, anchor) in enumerate(zip(movers, anchors)):
        label = f"{atoms[anchor].element.symbol}-{atoms[mover].element.symbol}"
        groups.setdefault(label, []).append(i)
    return {label: np.array(idx) for label, idx in sorted(groups.items())}


def diagnostics(
    moments: np.ndarray,
    n: int,
    energies: np.ndarray,
    sigma: np.ndarray | None = None,
    groups: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Turn the accumulated moments into the metrics a run logs.

    :param moments: Running sums of ``d - r0`` and its square, input then output, per bond.
    :param n: Number of frames they were accumulated over.
    :param energies: Running sums of the potential energy, input then output, in kJ/mol.
    :param sigma: Analytic width ``sqrt(kT / k)`` per bond, for the ratios.
    :param groups: Bond indices per element-pair label, for the per-type breakdown.
    :return: The metrics, widths in pm and energies in kJ/mol."""
    mean = moments / n
    width_in = np.sqrt(np.maximum(mean[1] - mean[0] ** 2, 0.0))
    width_out = np.sqrt(np.maximum(mean[3] - mean[2] ** 2, 0.0))
    e_in, e_out = energies / n

    values = {
        "width/in_pm": float(width_in.mean() * 1000),
        "width/out_pm": float(width_out.mean() * 1000),
        "width/gain": float((width_out.mean() / width_in.mean()) if width_in.mean() else 0.0),
        "shift/out_pm": float(mean[2].mean() * 1000),
        "energy/in_kJ_mol": float(e_in),
        "energy/out_kJ_mol": float(e_out),
        "energy/drop_kJ_mol": float(e_in - e_out),
    }
    if sigma is not None:
        ratio = width_out / sigma
        values |= {
            "width/analytic_pm": float(sigma.mean() * 1000),
            "width/out_over_analytic": float(ratio.mean()),
            "width/out_over_analytic_min": float(ratio.min()),
            "width/out_over_analytic_max": float(ratio.max()),
        }
    for label, idx in (groups or {}).items():
        values[f"width/{label}_out_pm"] = float(width_out[idx].mean() * 1000)
        if sigma is not None:
            values[f"width/{label}_out_over_analytic"] = float((width_out[idx] / sigma[idx]).mean())
    return values


def shard_bounds(
    n_frames: int, shard: int, n_shards: int, start: int = 0, stop: int | None = None
) -> tuple[int, int]:
    """Split a frame selection into contiguous blocks and return this shard's bounds.

    Contiguous, not strided: the shards of this tool are concatenated back into one trajectory,
    and only contiguous blocks concatenate in shard order into the *input's* frame order. That
    order is what every per-frame array derived from the same trajectory -- energies, scores,
    held-out frame indices -- is indexed by, so a strided split would silently permute the dataset
    against all of them. (The lab repo's ``slurm/resample-h-bonds.sh`` has a sibling for
    *relaxation* that sharded by stride, which was safe only because its output was never
    re-indexed against the input.)

    The blocks differ by at most one frame, and the last one absorbs the remainder.

    :param n_frames: Number of frames in the trajectory.
    :param shard: Zero-based index of this shard.
    :param n_shards: Total number of shards.
    :param start: First frame of the overall selection.
    :param stop: Stop before this frame, or ``None`` for the end of the trajectory.
    :return: This shard's ``(start, stop)`` as absolute frame indices.
    :raises ValueError: If the shard index is out of range or the selection is empty."""
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} is out of range for {n_shards} shards")
    end = n_frames if stop is None else min(stop, n_frames)
    if end <= start:
        raise ValueError(f"empty selection: frames [{start}, {end}) of {n_frames}")

    edges = np.linspace(start, end, n_shards + 1).round().astype(int)
    return int(edges[shard]), int(edges[shard + 1])


def count_frames(trajectory: str, topology: str) -> int:
    """Count a trajectory's frames without holding it in memory.

    :param trajectory: Trajectory path.
    :param topology: Topology path, needed by the formats that carry no atom count.
    :return: The number of frames."""
    import mdtraj as md

    try:
        with md.open(trajectory) as f:
            return len(f)
    except (TypeError, NotImplementedError):  # formats whose reader has no length
        return sum(len(chunk) for chunk in md.iterload(trajectory, chunk=10_000, top=topology))


def equilibrate_trajectory(
    trajectory: str,
    topology: str,
    out: Path,
    simulation: app.Simulation,
    movers: np.ndarray,
    anchors: np.ndarray,
    r0: np.ndarray,
    temperature: float,
    timestep: float,
    equilibration_time: float,
    seed: int = 42,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    chunk: int = 2_000,
    sigma: np.ndarray | None = None,
    groups: dict[str, np.ndarray] | None = None,
    progress: Callable[[int, dict[str, float]], None] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[int, np.ndarray, np.ndarray, dict[str, float]]:
    """Equilibrate a trajectory file chunk by chunk, so memory stays flat in its length.

    Frames are equilibrated independently, so the chunking changes nothing but the memory profile.

    :param trajectory: Input trajectory path.
    :param topology: Topology of the input trajectory.
    :param out: Output trajectory path; the format follows its extension.
    :param simulation: The simulation to integrate in, frozen atoms already at zero mass.
    :param movers: Index of the free end, per constrained bond; these are the mobile atoms.
    :param anchors: Index of the atom it is measured from, per constrained bond.
    :param r0: Equilibrium length per bond, in nanometre; the widths are accumulated around it.
    :param temperature: Temperature in Kelvin.
    :param timestep: Integration timestep in picoseconds.
    :param equilibration_time: Time to integrate each frame for, in picoseconds.
    :param seed: Base for the velocity draws; each frame uses ``seed + 1 + <absolute frame
        index>``, so neither the chunking nor a split into shards changes a frame's draw.
    :param start: First input frame to take.
    :param stop: Stop before this input frame, or ``None`` for the end.
    :param stride: Take every ``stride``-th input frame.
    :param chunk: Number of frames held in memory at once.
    :param sigma: Analytic width ``sqrt(kT / k)`` per bond, for the logged ratios.
    :param groups: Bond indices per element-pair label, for the per-type breakdown.
    :param progress: Called after each chunk with the number of frames done so far and the
        running diagnostics, for a tracker to log.
    :param logger: Logger for progress messages.
    :return: The number of frames written, the per-bond length std before and after in nm, and
        the final diagnostics."""
    import mdtraj as md
    from mdtraj.formats import TRRTrajectoryFile, XTCTrajectoryFile
    from mdtraj.utils import in_units_of

    top = md.load_topology(topology)
    if (highest := max(movers.max(), anchors.max())) >= top.n_atoms:
        raise ValueError(
            f"the config indexes atom {highest} but the trajectory has only {top.n_atoms}; the "
            "config and the trajectory are not the same system"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    chunk_started = time.monotonic()
    energies = np.zeros(2)
    remaining = None if stop is None else len(range(start, stop, stride))
    n = 0
    # moments of d - r0 rather than d, so the frozen input's ~1e-6 nm width survives float64
    moments = np.zeros((4, len(movers)))
    with md.open(str(out), "w") as f:
        for traj in md.iterload(trajectory, chunk=chunk, top=top, skip=start, stride=stride):
            if remaining is not None:
                traj = traj[:remaining]
                remaining -= len(traj)
            if not len(traj):
                break

            positions = traj.xyz.astype(np.float64)
            equilibrated, frame_energies = equilibrate_frames(
                positions,
                simulation.system,
                simulation.topology,
                movers,
                temperature=temperature,
                timestep=timestep,
                equilibration_time=equilibration_time,
                frame_seeds=seed + 1 + start + (n + np.arange(len(positions))) * stride,
                return_energies=True,
                simulation=simulation,
                logger=None,
            )

            before = np.linalg.norm(positions[:, movers] - positions[:, anchors], axis=-1) - r0
            after = np.linalg.norm(equilibrated[:, movers] - equilibrated[:, anchors], axis=-1) - r0
            moments += [before.sum(0), (before**2).sum(0), after.sum(0), (after**2).sum(0)]
            energies += frame_energies.sum(0)
            n += len(traj)

            now = time.monotonic()
            values = diagnostics(moments, n, energies, sigma, groups) | {
                "speed/frames_per_s": len(traj) / max(now - chunk_started, 1e-9)
            }
            chunk_started = now
            if logger is not None:
                logger.info(
                    f"Equilibrated {n} frames: X-H width {values['width/in_pm']:.3f} -> "
                    f"{values['width/out_pm']:.3f} pm, potential "
                    f"{values['energy/drop_kJ_mol']:+.2f} kJ/mol."
                )
            if progress is not None:
                progress(n, values)

            xyz = in_units_of(equilibrated.astype(np.float32), "nanometers", f.distance_unit)
            if isinstance(f, XTCTrajectoryFile | TRRTrajectoryFile):
                f.write(xyz, time=traj.time)
            else:
                f.write(xyz)

            if remaining == 0:
                break

    if n == 0:
        raise ValueError(f"no frames selected from {trajectory}")
    mean = moments / n
    return (
        n,
        np.sqrt(np.maximum(mean[1] - mean[0] ** 2, 0.0)),
        np.sqrt(np.maximum(mean[3] - mean[2] ** 2, 0.0)),
        diagnostics(moments, n, energies, sigma, groups),
    )


def start_equilibration_run(
    args: argparse.Namespace, config: SimulationConfig, friction: float, sigma: np.ndarray
):
    """Open the Beamtime run this job logs to.

    One run per shard, so a 16-task array is 16 runs in one group and a requeued task continues
    only its own. The config records what the output cannot be re-derived from: which frames of
    which trajectory, under which potential, for how long and at what friction.

    :param args: The parsed command line.
    :param config: The simulation config defining the potential.
    :param friction: The friction actually used, the config's unless overridden.
    :param sigma: The analytic per-bond widths, recorded as the reference the ratios are against.
    :return: The started run."""
    import beamtime

    return beamtime.init(
        dir=args.beamtime_group or config.beamtime_group,
        name=args.beamtime_name or args.out.stem,
        tags=[config.engine, "equilibrate-h-bonds"],
        config={
            "tool": "equilibrate_h_bonds",
            "potential_config": str(Path(args.config).resolve()),
            "trajectory": str(Path(args.trajectory).resolve()),
            "topology": str(Path(args.topology).resolve()),
            "out": str(args.out.resolve()),
            "constrained_as": args.constrained_as,
            "equilibration_time_ps": args.time,
            "timestep_ps": args.timestep,
            "friction_per_ps": friction,
            "temperature": config.temperature,
            "platform": args.platform,
            "seed": args.seed,
            "shard": args.shard,
            "n_shards": args.n_shards,
            "frames": {"start": args.start, "stop": args.stop, "stride": args.stride},
            "n_bonds": int(len(sigma)),
            "analytic_width_pm": float(sigma.mean() * 1000),
        },
    )


def main() -> None:
    """Entry point for the ``md-sim-equilibrate-hbonds`` command."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Simulation config defining the unconstrained potential.")
    parser.add_argument("--trajectory", required=True, help="Input trajectory to equilibrate.")
    parser.add_argument("--topology", required=True, help="Topology of the input trajectory.")
    parser.add_argument("-o", "--out", required=True, type=Path, help="Output trajectory path.")
    parser.add_argument(
        "--constrained-as",
        default="hbonds",
        help="Constraint scheme the input trajectory was produced with; names the bonds to free.",
    )
    parser.add_argument(
        "--time", type=float, default=2.0, help="Equilibration time per frame (ps)."
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.0005,
        help="Integration timestep (ps); must resolve the unconstrained X-H stretch.",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="Langevin friction (1/ps); defaults to the config's. Higher converges faster and "
        "biases nothing, since the stationary distribution does not depend on it.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Take every stride-th input frame.")
    parser.add_argument("--start", type=int, default=0, help="First input frame to take.")
    parser.add_argument("--stop", type=int, default=None, help="Stop before this input frame.")
    parser.add_argument("--chunk", type=int, default=2000, help="Frames held in memory at once.")
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Zero-based index of this shard; narrows --start/--stop to a contiguous block so "
        "that the shards concatenate, in shard order, back into the input's frame order.",
    )
    parser.add_argument("--n-shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--platform", default=None, help="OpenMM platform (CPU, CUDA, ...).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the velocity draws.")
    parser.add_argument(
        "--data-root", default=None, help="Root for the config's paths (overrides MD_DATA_ROOT)."
    )
    parser.add_argument(
        "--beamtime-group",
        default=None,
        help="Beamtime group for this job; defaults to the config's (i.e. the system).",
    )
    parser.add_argument(
        "--beamtime-name",
        default=None,
        help="Beamtime run name; defaults to the output file's stem, so a sharded array lands as "
        "one run per shard in the group.",
    )
    parser.add_argument(
        "--no-beamtime", action="store_true", help="Do not register this job with Beamtime."
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
    logger.info(f"Trajectory: {args.trajectory} (constrained as {args.constrained_as})")
    logger.info(f"Output:     {args.out}")

    if args.shard is not None:
        n_frames = count_frames(args.trajectory, args.topology)
        args.start, args.stop = shard_bounds(
            n_frames, args.shard, args.n_shards, args.start, args.stop
        )
        logger.info(
            f"Shard {args.shard} of {args.n_shards}: frames [{args.start}, {args.stop}) "
            f"of {n_frames}."
        )

    movers, anchors, r0, sigma = constrained_bond_parameters(
        config, data_root, args.constrained_as, logger
    )
    logger.info(
        f"{len(movers)} constrained bonds; their free ends are the only mobile atoms, "
        f"analytic width {sigma.mean() * 1000:.3f} pm on average."
    )

    built = build_frozen_system(config, data_root, movers, logger)
    friction = config.friction if args.friction is None else args.friction
    simulation = build_simulation(
        built.system,
        built.topology,
        config.temperature,
        friction,
        args.timestep,
        args.platform,
        args.seed,
    )
    logger.info(
        f"Equilibrating for {args.time} ps per frame "
        f"({int(round(args.time / args.timestep))} steps of {args.timestep * 1000:.2f} fs) at "
        f"{config.temperature} K, friction {friction} /ps, on "
        f"{simulation.context.getPlatform().getName()}."
    )

    groups = bond_groups(built.topology, movers, anchors)
    logger.info(
        "Constrained bonds by type: "
        + ", ".join(f"{label} x{len(idx)}" for label, idx in groups.items())
    )

    run = None if args.no_beamtime else start_equilibration_run(args, config, friction, sigma)
    if run is not None:
        logger.info(f"Beamtime run: {run.name} ({run.id})")
    try:
        n, before, after, final = equilibrate_trajectory(
            args.trajectory,
            args.topology,
            args.out,
            simulation,
            movers,
            anchors,
            r0,
            temperature=config.temperature,
            timestep=args.timestep,
            equilibration_time=args.time,
            seed=args.seed,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
            chunk=args.chunk,
            sigma=sigma,
            groups=groups,
            progress=None if run is None else (lambda step, values: run.log(values, step=step)),
            logger=logger,
        )
    except BaseException:
        if run is not None:
            run.finish(status="failed")
        raise

    logger.info(f"Wrote {n} equilibrated frames to {args.out}.")
    logger.info(
        "Per-bond length std, averaged over bonds: "
        f"{before.mean() * 1000:.3f} pm in -> {after.mean() * 1000:.3f} pm out "
        f"(independent-harmonic sqrt(kT/k): {sigma.mean() * 1000:.3f} pm). "
        "The output is the conditional, not the analytic draw; a residual difference from "
        "sqrt(kT/k) is the coupling this tool exists to capture, but check it against a longer "
        "--time before believing it."
    )
    for label, idx in groups.items():
        logger.info(
            f"  {label:>4} x{len(idx):<3} {after[idx].mean() * 1000:6.3f} pm out, "
            f"{(after[idx] / sigma[idx]).mean():.3f} of analytic"
        )

    if run is not None:
        # Logged as an event as well as a final metric row: the per-bond extremes say whether a
        # single bond failed to converge, which the mean over 61 of them hides.
        run.log(final | {"frames": n}, step=n)
        run.event("equilibration finished", frames=n, **final)
        record_outputs(run, [args.out])
        run.finish()


if __name__ == "__main__":
    main()
