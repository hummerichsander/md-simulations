"""Temperature replica exchange MD (T-REMD) for OpenMM systems.

One :class:`openmm.app.Simulation` per replica runs at a rung of a temperature ladder; every
``exchange_interval`` steps neighbouring temperatures attempt to swap, with the Metropolis
criterion of :mod:`md_simulations.core.replica_exchange`. The point is equilibrium sampling at
the lowest (target) temperature, where barriers are too high for plain MD: configurations
diffuse up the ladder, cross the barriers hot, and come back down.

Swapping temperatures, not coordinates
--------------------------------------
Each context is one continuous replica: its positions never jump. A swap changes the
temperature of two replicas (``integrator.setTemperature``) and rescales their velocities by
``sqrt(T_new / T_old)`` so the kinetic energy matches the new temperature. This is equivalent to
exchanging the configurations, but moves no coordinates, and ``replica_at[k]`` (the replica
currently at temperature ``k``) is all the bookkeeping there is. Frames are written *before* the
swap of the same step, so a frame belongs to the temperature it was sampled at. The system must
be NVT: the criterion uses the potential energy only, and a barostat would need the enthalpy.

Replicas run one after another in one process; :meth:`ReplicaExchange._advance` is the only place
that steps them, so a thread pool can be added there.

Outputs
-------
At the target temperature, in the output directory, in the plain-MD file formats (frames every
``output_freq`` steps): ``trajectory.dcd``, ``energies.csv``, ``forces.npy``,
``initial_structure.pdb``, ``final_structure.pdb`` (the last sampled frame at the target) and
``simulation_summary.txt`` with a replica-exchange section. The other temperatures write into
``T{k:02d}_{T:.0f}K/`` as ``other_output`` says: ``energies.csv`` only, or all three files. Step
numbers are global, and identical across files. ``exchanges.csv`` has one row per exchange
attempt: ``step``, ``parity``, ``replica_at_T00, ...`` (who sits at each temperature *after* the
swap) and ``accepted_T00_T01, ...`` (``1``/``0``, empty for pairs not attempted at that parity).

Restart
-------
Every ``checkpoint_interval`` steps, after the exchange, ``checkpoint/`` receives each replica's
``context.createCheckpoint()`` bytes and ``state.json`` (step, ``replica_at``, parity, exchange
RNG state, statistics, up/down labels, Beamtime run ids). Files are written to a temporary name
and renamed, and ``state.json`` goes last and names the step of the ``.chk`` files it belongs to,
so a kill never leaves a mixed checkpoint. Every open output is fsynced before the checkpoint is
written, so even a node crash cannot leave the checkpoint ahead of its files. Resuming (``md-sim run --resume``) restores the
contexts, re-applies each integrator's temperature from ``replica_at`` (a checkpoint does not
carry it), cuts every text output and the DCDs back to the checkpointed step and appends from
there. Every file must hold exactly the rows or frames the checkpointed step implies, and the output
intervals must be unchanged (``num_steps`` may grow); otherwise the resume stops. A checkpoint
without ``--resume`` is an error; nothing is overwritten. OpenMM checkpoints
are platform- and device-specific: resume on the same platform (CUDA, CPU, ...) and OpenMM
version the run started on. The integrators' random streams are restored by the platform where it
supports that, so a resumed run is a valid continuation, not bitwise identical.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any
import json
import logging
import math
import os
import struct
import time

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from beamtime.sdk.run import Run
from tqdm import tqdm

from md_simulations.core.integrators import create_integrator
from md_simulations.core.replica_exchange import (
    ExchangeStats,
    attempt_swaps,
    attempted_pairs,
    geometric_ladder,
)
from md_simulations.core.forces import FORCES_FILE, ForceWriter
from md_simulations.core.runner import save_final_structure, save_initial_structure
from md_simulations.core.summary import write_simulation_summary
from md_simulations.core.tracking import degrees_of_freedom, kinetic_temperature

CHECKPOINT_DIR = "checkpoint"
CHECKPOINT_STATE = "state.json"
EXCHANGES_FILE = "exchanges.csv"
ENERGIES_HEADER = '#"Step","Potential Energy (kJ/mole)"\n'

_KJ = unit.kilojoules_per_mole
_FORCE = unit.kilojoules_per_mole / unit.nanometer
_R = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
_BAROSTATS = (
    openmm.MonteCarloBarostat,
    openmm.MonteCarloAnisotropicBarostat,
    openmm.MonteCarloMembraneBarostat,
    openmm.MonteCarloFlexibleBarostat,
)


def _complete_rows(path: Path, step: int, rows: int) -> None:
    """Cut a CSV with a header and a ``step`` first column back to the rows up to *step*.

    A trailing line a kill cut short is dropped as well.

    :param path: The file, replaced in place.
    :param step: The last step to keep.
    :param rows: The number of rows the checkpoint implies; fewer is an error, since appending
        after missing rows would shift this file against the others."""
    kept = _rewrite_lines(
        path, lambda i, line: i == 0 or (line.endswith("\n") and int(line.split(",")[0]) <= step)
    )
    if kept - 1 != rows:
        raise ValueError(
            f"{path} holds {kept - 1} rows up to step {step}, the checkpoint needs {rows}"
        )


def resolve_ladder(config: Any) -> list[float]:
    """Resolve the temperature ladder of a simulation config.

    :param config: A simulation config with a ``replica_exchange`` block.
    :return: The temperatures in Kelvin, ascending, starting at ``config.temperature``."""
    rex = config.replica_exchange
    if rex.temperatures is not None:
        return list(rex.temperatures)
    return geometric_ladder(config.temperature, rex.t_max, rex.n_replicas)


def temperature_names(temperatures: list[float]) -> list[str]:
    """Name every temperature, for its output folder and Beamtime run.

    :param temperatures: The ladder.
    :return: ``T{k:02d}_{T:.0f}K`` for each rung."""
    return [f"T{k:02d}_{t:.0f}K" for k, t in enumerate(temperatures)]


def rescale_velocities(simulation: app.Simulation, factor: float) -> None:
    """Scale the velocities of a simulation, which scales its kinetic energy by ``factor**2``.

    :param simulation: The simulation.
    :param factor: The velocity factor."""
    velocities = simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)
    simulation.context.setVelocities(velocities * factor)


def read_checkpoint(output_dir: Path) -> dict | None:
    """Read the state of the checkpoint in an output directory.

    :param output_dir: The output directory of a replica-exchange run.
    :return: The content of ``checkpoint/state.json``, or ``None`` without a checkpoint."""
    path = output_dir / CHECKPOINT_DIR / CHECKPOINT_STATE
    return json.loads(path.read_text()) if path.exists() else None


def _atomic_write(path: Path, data: bytes) -> None:
    """Write a file so that readers see the old or the new content, never a part.

    :param path: Destination.
    :param data: The content."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _rewrite_lines(path: Path, keep: Callable[[int, str], bool]) -> int:
    """Keep only some lines of a text file, streaming.

    :param path: The file, replaced in place.
    :param keep: Called with the line index and the line; the line is kept if it returns true.
    :return: The number of lines kept."""
    tmp = path.with_name(path.name + ".tmp")
    kept = 0
    with open(path) as src, open(tmp, "w") as dst:
        for i, line in enumerate(src):
            if keep(i, line):
                dst.write(line)
                kept += 1
    os.replace(tmp, path)
    return kept


def truncate_dcd(path: Path, frames: int) -> None:
    """Cut a DCD back to its first frames and fix its header, so it can be appended to.

    :param path: The DCD file written by :class:`openmm.app.DCDFile`.
    :param frames: The number of frames to keep."""
    with open(path, "r+b") as f:
        header = f.read(276)
        first_step, interval = struct.unpack_from("<ii", header, 12)
        box = struct.unpack_from("<i", header, 48)[0]
        n_atoms = struct.unpack_from("<i", header, 268)[0]
        frame_size = (56 if box else 0) + 3 * (8 + 4 * n_atoms)
        end = len(header) + frames * frame_size
        if os.fstat(f.fileno()).st_size < end:
            raise ValueError(f"{path} has fewer than the {frames} frames its checkpoint expects")
        f.truncate(end)
        f.seek(8)
        f.write(struct.pack("<i", frames))
        f.seek(20)
        f.write(struct.pack("<i", first_step + (frames - 1) * interval if frames else 0))


class FrameWriter:
    """The files of one temperature: ``energies.csv``, plus trajectory and forces when ``full``.

    Writes the formats of :func:`md_simulations.core.add_standard_reporters`.

    :param directory: Where the files go.
    :param topology: Topology for the DCD.
    :param timestep: Integrator step size, for the DCD header.
    :param interval: Steps between frames.
    :param full: Also write ``trajectory.dcd`` and ``forces.npy``.
    :param resume_step: When given, reopen the files of an earlier run, cut back to this step."""

    def __init__(
        self,
        directory: Path,
        topology: app.Topology,
        timestep: Any,
        interval: int,
        full: bool,
        resume_step: int | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.full = full
        self._dcd: app.DCDFile | None = None
        self._dcd_file = None
        self._forces = None
        energies = directory / "energies.csv"
        dcd = directory / "trajectory.dcd"
        forces = directory / FORCES_FILE
        frames = 0 if resume_step is None else resume_step // interval
        if resume_step is None:
            self._energies = open(energies, "w")
            self._energies.write(ENERGIES_HEADER)
        else:
            _complete_rows(energies, resume_step, frames)
            self._energies = open(energies, "a")
        if full:
            if resume_step is None:
                self._dcd_file = open(dcd, "wb")
                self._forces = ForceWriter(forces, topology.getNumAtoms())
            else:
                truncate_dcd(dcd, frames)
                self._forces = ForceWriter(forces, topology.getNumAtoms(), append=True)
                self._forces.truncate(frames)
                self._dcd_file = open(dcd, "r+b")
            self._dcd = app.DCDFile(
                self._dcd_file, topology, timestep, interval, interval, resume_step is not None
            )

    def due(self, step: int) -> bool:
        """Whether a frame is written at *step*.

        :param step: The global step.
        :return: ``True`` on a multiple of the interval."""
        return step % self.interval == 0

    def write(self, step: int, state: openmm.State) -> None:
        """Write one frame.

        :param step: The global step.
        :param state: A state with the energy, and positions and forces if ``full``."""
        self._energies.write(f"{step},{state.getPotentialEnergy().value_in_unit(_KJ)}\n")
        self._energies.flush()
        if self._dcd is not None and self._forces is not None:
            self._dcd.writeModel(
                state.getPositions(asNumpy=True), periodicBoxVectors=state.getPeriodicBoxVectors()
            )
            self._forces.write(state.getForces(asNumpy=True).value_in_unit(_FORCE))

    def sync(self) -> None:
        """Force everything written so far to disk."""
        for f in (self._energies, self._forces, self._dcd_file):
            if f is not None:
                f.flush()
                os.fsync(f.fileno())

    def close(self) -> None:
        """Close the files."""
        self._energies.close()
        if self._forces is not None and self._dcd_file is not None:
            self._forces.close()
            self._dcd_file.close()


class ReplicaExchange:
    """Run T-REMD on one system, see the module docstring.

    :param topology: Topology of the system.
    :param system: The OpenMM system, shared by all replicas.
    :param positions: Initial positions, the same for all replicas.
    :param box_vectors: Periodic box vectors to set on every context, or ``None``.
    :param config: The simulation config, with a ``replica_exchange`` block.
    :param output_dir: The directory to write into.
    :param runs: One Beamtime run per temperature, in ladder order.
    :param logger: Logger for progress messages.
    :param initialize_velocities: Draw velocities at each replica's temperature at the start."""

    def __init__(
        self,
        *,
        topology: app.Topology,
        system: openmm.System,
        positions: Any,
        box_vectors: Any | None,
        config: Any,
        output_dir: Path,
        runs: list[Run],
        logger: logging.Logger,
        initialize_velocities: bool = True,
    ) -> None:
        if any(isinstance(force, _BAROSTATS) for force in system.getForces()):
            raise NotImplementedError("replica exchange needs an NVT system, but it has a barostat")
        self.config = config
        self.rex = config.replica_exchange
        self.output_dir = output_dir
        self.runs = runs
        self.logger = logger
        self.initialize_velocities = initialize_velocities
        self.topology = topology
        self.system = system

        self.temperatures = resolve_ladder(config)
        self.names = temperature_names(self.temperatures)
        self.betas = [1.0 / (_R * t) for t in self.temperatures]
        if len(runs) != len(self.temperatures):
            raise ValueError(f"need {len(self.temperatures)} runs, got {len(runs)}")

        other_freq = self.rex.other_output_freq or config.output_freq
        freqs = [config.output_freq, self.rex.exchange_interval]
        if self.rex.other_output != "none":
            freqs.append(other_freq)
        self.chunk = min(freqs)
        self.log_every = self.chunk * max(1, round(config.report_interval / self.chunk))
        self.other_freq = other_freq

        self.simulations = [
            self._create_simulation(r, positions, box_vectors) for r in self.indices
        ]
        self.periodic = system.usesPeriodicBoundaryConditions()
        self.dof = degrees_of_freedom(system)

        self.replica_at = list(self.indices)
        self.parity = 0
        self.step = 0
        self.rng = np.random.default_rng(self.rex.seed)
        self.stats = ExchangeStats(len(self.temperatures))
        self.writers: list[FrameWriter | None] = []
        self._exchanges = None
        self._last_log: tuple[float, float] | None = None

    @property
    def indices(self) -> range:
        """The replica indices, which are also the starting temperature indices.

        :return: ``range(n_temperatures)``."""
        return range(len(self.temperatures))

    def _create_simulation(self, r: int, positions: Any, box_vectors: Any | None) -> app.Simulation:
        """Build the simulation of replica *r*, starting at temperature ``r``.

        :param r: The replica index.
        :param positions: Initial positions.
        :param box_vectors: Periodic box vectors, or ``None``.
        :return: The simulation."""
        cfg = self.config
        integrator = create_integrator(
            self.temperatures[r], cfg.friction, cfg.timestep, kind=cfg.integrator
        )
        if self.rex.seed is not None:
            integrator.setRandomNumberSeed(self.rex.seed + 1 + r)
        simulation = app.Simulation(self.topology, self.system, integrator)
        simulation.context.setPositions(positions)
        if box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*box_vectors)
        return simulation

    def run(self, resume: dict | None = None) -> None:
        """Run to ``num_steps``, from the start or from a checkpoint.

        :param resume: The state of a checkpoint (:func:`read_checkpoint`), or ``None`` to start."""
        cfg = self.config
        if resume is None:
            self._initialise()
        else:
            self._restore(resume)
        self._open_outputs(resume)
        try:
            self.logger.info(
                f"Running T-REMD with {len(self.temperatures)} replicas at "
                f"{', '.join(f'{t:.1f}' for t in self.temperatures)} K for {cfg.num_steps:,} steps "
                f"(from step {self.step:,}, exchange every {self.rex.exchange_interval} steps)..."
            )
            n_chunks = math.ceil((cfg.num_steps - self.step) / self.chunk)
            for _ in tqdm(range(n_chunks), desc="T-REMD Progress"):
                self._advance(self.chunk)
                self.step += self.chunk
                self._boundary()
        finally:
            self._close_outputs()
        self._write_summary()
        self._log_statistics()
        self.logger.info("Simulation completed successfully!")
        self.logger.info(f"Output files: {self.output_dir}")

    def _initialise(self) -> None:
        """Save the initial structure and summary, minimise and thermalise every replica."""
        save_initial_structure(self.simulations[0], self.output_dir, self.logger)
        self._write_summary()
        for r, simulation in enumerate(self.simulations):
            if self.config.minimize_energy:
                self.logger.info(f"Minimizing energy of replica {r}...")
                simulation.minimizeEnergy()
            if self.initialize_velocities:
                seed = [] if self.rex.seed is None else [self.rex.seed + 1 + r]
                simulation.context.setVelocitiesToTemperature(
                    self.temperatures[r] * unit.kelvin,  # type: ignore[arg-type]
                    *seed,
                )
        self.stats.touch_ends(self.replica_at)

    def _restore(self, state: dict) -> None:
        """Restore the contexts and the exchange state of a checkpoint.

        :param state: The content of ``state.json``."""
        saved = state["temperatures"]
        if len(saved) != len(self.temperatures) or not np.allclose(saved, self.temperatures):
            raise ValueError(f"the checkpoint ladder {saved} differs from {self.temperatures}")
        platform = self.simulations[0].context.getPlatform().getName()
        if state["platform"] != platform:
            raise ValueError(
                f"the checkpoint was written on the {state['platform']} platform, "
                f"this is {platform}; OpenMM checkpoints do not cross platforms"
            )
        if state["grid"] != self._grid():
            raise ValueError(
                f"the output grid changed since the checkpoint: {state['grid']} became "
                f"{self._grid()}; only num_steps may differ on resume"
            )
        self.step = state["step"]
        self.replica_at = list(state["replica_at"])
        self.parity = state["parity"]
        self.rng.bit_generator.state = state["rng_state"]
        self.stats = ExchangeStats.from_dict(state["stats"])
        directory = self.output_dir / CHECKPOINT_DIR
        for r, simulation in enumerate(self.simulations):
            simulation.context.loadCheckpoint((directory / state["files"][r]).read_bytes())
            simulation.context.setStepCount(self.step)
        for k, r in enumerate(self.replica_at):
            self.simulations[r].integrator.setTemperature(self.temperatures[k] * unit.kelvin)
        self.logger.info(f"Restored the checkpoint at step {self.step:,}.")

    def _open_outputs(self, resume: dict | None) -> None:
        """Open every output file, fresh or cut back to the checkpoint.

        :param resume: The checkpoint state, or ``None`` for a fresh run."""
        cfg, rex = self.config, self.rex
        resume_step = None if resume is None else resume["step"]
        timestep = self.simulations[0].integrator.getStepSize()
        self.writers = [
            FrameWriter(
                self.output_dir, self.topology, timestep, cfg.output_freq, True, resume_step
            )
        ]
        for k in range(1, len(self.temperatures)):
            if rex.other_output == "none":
                self.writers.append(None)
                continue
            self.writers.append(
                FrameWriter(
                    self.output_dir / self.names[k],
                    self.topology,
                    timestep,
                    self.other_freq,
                    rex.other_output == "full",
                    resume_step,
                )
            )

        path = self.output_dir / EXCHANGES_FILE
        if resume_step is None:
            self._exchanges = open(path, "w")
            n = len(self.temperatures)
            header = ["step", "parity"]
            header += [f"replica_at_T{k:02d}" for k in range(n)]
            header += [f"accepted_T{k:02d}_T{k + 1:02d}" for k in range(n - 1)]
            self._exchanges.write(",".join(header) + "\n")
        else:
            _complete_rows(path, resume_step, resume_step // self.rex.exchange_interval)
            self._exchanges = open(path, "a")

    def _close_outputs(self) -> None:
        """Close every output file."""
        for writer in self.writers:
            if writer is not None:
                writer.close()
        if self._exchanges is not None:
            self._exchanges.close()

    def _advance(self, n_steps: int) -> None:
        """Step every replica, the only place that does.

        :param n_steps: Steps to take."""
        for simulation in self.simulations:
            simulation.step(n_steps)

    def _boundary(self) -> None:
        """Write the outputs, exchange and checkpoint that are due at the current step."""
        step, rex = self.step, self.rex
        states = self._sample()
        for writer, state in zip(self.writers, states):
            if writer is not None and writer.due(step):
                writer.write(step, state)
        if step == self.config.num_steps:
            # the last sampled frame, before the final swap can move another replica to the target.
            save_final_structure(self.simulations[self.replica_at[0]], self.output_dir, self.logger)
        if step % rex.exchange_interval == 0:
            self._exchange(states)
        if step % self.log_every == 0:
            self._log_runs(states)
        if rex.checkpoint_interval and step % rex.checkpoint_interval == 0:
            self._checkpoint()

    def _sample(self) -> list[openmm.State]:
        """Read the state of the replica at every temperature.

        :return: The states in ladder order, with positions and forces where a writer wants them."""
        states = []
        for k, r in enumerate(self.replica_at):
            writer = self.writers[k]
            full = writer is not None and writer.full and writer.due(self.step)
            state = self.simulations[r].context.getState(
                getEnergy=True, getPositions=full, getForces=full, enforcePeriodicBox=self.periodic
            )
            if not math.isfinite(state.getPotentialEnergy().value_in_unit(_KJ)):
                raise ValueError(
                    f"non-finite potential energy at step {self.step}, replica {r} "
                    f"({self.temperatures[k]:.1f} K)"
                )
            states.append(state)
        return states

    def _exchange(self, states: list[openmm.State]) -> None:
        """Attempt the swaps of the current parity and move the temperatures.

        :param states: The states at every temperature, from before the swap."""
        energies = [s.getPotentialEnergy().value_in_unit(_KJ) for s in states]
        new_replica_at, accepted = attempt_swaps(
            energies, self.betas, self.replica_at, self.parity, self.rng
        )
        old_k = {r: k for k, r in enumerate(self.replica_at)}
        for k, r in enumerate(new_replica_at):
            if old_k[r] != k:
                self.simulations[r].integrator.setTemperature(self.temperatures[k] * unit.kelvin)
                rescale_velocities(
                    self.simulations[r],
                    math.sqrt(self.temperatures[k] / self.temperatures[old_k[r]]),
                )
        self.replica_at = new_replica_at
        self.stats.record(self.parity, accepted, new_replica_at)

        n = len(self.temperatures)
        attempted = attempted_pairs(self.parity, n)
        flags = ["" if k not in attempted else str(int(k in accepted)) for k in range(n - 1)]
        row = [self.step, self.parity, *new_replica_at, *flags]
        assert self._exchanges is not None
        self._exchanges.write(",".join(str(v) for v in row) + "\n")
        self._exchanges.flush()
        self.parity = 1 - self.parity

    def _log_runs(self, states: list[openmm.State]) -> None:
        """Log every temperature's state, and the exchange statistics, to its Beamtime run.

        :param states: The states at every temperature, from before the last swap."""
        now = time.time()
        ps = states[0].getTime().value_in_unit(unit.picosecond)
        speed = None
        if self._last_log is not None and now > self._last_log[0]:
            speed = (ps - self._last_log[1]) * 86.4 / (now - self._last_log[0])
        self._last_log = (now, ps)
        for k, (run, state) in enumerate(zip(self.runs, states)):
            potential = state.getPotentialEnergy().value_in_unit(_KJ)
            kinetic = state.getKineticEnergy().value_in_unit(_KJ)
            values = {
                "energy/potential": potential,
                "energy/kinetic": kinetic,
                "energy/total": potential + kinetic,
                "temperature": kinetic_temperature(kinetic, self.dof),
                "exchange/round_trips": self.stats.round_trips,
            }
            if speed is not None:
                values["speed/ns_per_day"] = speed
            if k < len(self.temperatures) - 1:
                values["exchange/acceptance"] = self.stats.acceptance(k)
            run.log(values, step=self.step)

    def _grid(self) -> dict:
        """Describe what the output files are cut by, which a resume must not change.

        :return: The intervals, and the atom count."""
        other = None if self.rex.other_output == "none" else self.other_freq
        return {
            "output_freq": self.config.output_freq,
            "other_output": self.rex.other_output,
            "other_output_freq": other,
            "exchange_interval": self.rex.exchange_interval,
            "n_atoms": self.system.getNumParticles(),
        }

    def _checkpoint(self) -> None:
        """Write the checkpoint of the current step and drop the previous one.

        The outputs are synced first: a checkpoint ahead of its files after a node crash would
        shift them against each other on resume."""
        for writer in self.writers:
            if writer is not None:
                writer.sync()
        assert self._exchanges is not None
        self._exchanges.flush()
        os.fsync(self._exchanges.fileno())
        directory = self.output_dir / CHECKPOINT_DIR
        directory.mkdir(exist_ok=True)
        files = [f"replica_{r:02d}_{self.step}.chk" for r in self.indices]
        for simulation, name in zip(self.simulations, files):
            _atomic_write(directory / name, simulation.context.createCheckpoint())
        state = {
            "step": self.step,
            "replica_at": self.replica_at,
            "parity": self.parity,
            "rng_state": self.rng.bit_generator.state,
            "stats": self.stats.to_dict(),
            "temperatures": self.temperatures,
            "platform": self.simulations[0].context.getPlatform().getName(),
            "openmm_version": openmm.version.version,
            "files": files,
            "grid": self._grid(),
            "run_ids": [run.id for run in self.runs],
        }
        _atomic_write(directory / CHECKPOINT_STATE, json.dumps(state).encode())
        for old in directory.glob("replica_*.chk"):
            if old.name not in files:
                old.unlink()
        self.logger.info(f"Wrote checkpoint at step {self.step:,}.")

    def _write_summary(self) -> None:
        """Write ``simulation_summary.txt`` with the replica-exchange section."""
        rex, stats = self.rex, self.stats
        ps = rex.exchange_interval * self.config.timestep
        rows = [
            ("Temperatures", ", ".join(f"{t:.2f} K" for t in self.temperatures)),
            ("Exchange interval", f"{rex.exchange_interval:,} steps ({ps:.4g} ps)"),
            ("Exchange seed", "none (random)" if rex.seed is None else str(rex.seed)),
            (
                "Other temperatures",
                "none"
                if rex.other_output == "none"
                else f"{rex.other_output} every {self.other_freq:,} steps",
            ),
        ]
        if any(stats.attempts):
            for k, (low, high) in enumerate(zip(self.temperatures, self.temperatures[1:])):
                rows.append(
                    (
                        f"Acceptance {low:.1f}-{high:.1f} K",
                        f"{stats.acceptance(k):.3f} ({stats.accepted[k]}/{stats.attempts[k]})",
                    )
                )
            rows.append(("Round trips", str(stats.round_trips)))
        write_simulation_summary(
            self.output_dir,
            self.config,
            self.simulations[self.replica_at[0]],
            self.logger,
            extra_sections={"Replica exchange": rows},
        )

    def _log_statistics(self) -> None:
        """Log the final acceptance per pair and the round trips."""
        for k, (low, high) in enumerate(zip(self.temperatures, self.temperatures[1:])):
            self.logger.info(
                f"Swap acceptance {low:.1f}-{high:.1f} K: {self.stats.acceptance(k):.3f} "
                f"({self.stats.accepted[k]}/{self.stats.attempts[k]})"
            )
        self.logger.info(f"Replica round trips: {self.stats.round_trips}")
