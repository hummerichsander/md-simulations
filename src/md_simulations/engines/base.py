import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import openmm
import openmm.app as app
import openmm.unit as unit
from beamtime.sdk.run import Run

from md_simulations.core import (
    BeamtimeReporter,
    ReplicaExchange,
    add_standard_reporters,
    create_integrator,
    freeze_atoms,
    record_outputs,
    read_checkpoint,
    resume_replica_runs,
    run_simulation_loop,
    save_final_structure,
    save_initial_structure,
    start_replica_runs,
    start_run,
    write_simulation_summary,
)
from md_simulations.core.remd import (
    CHECKPOINT_DIR,
    EXCHANGES_FILE,
    resolve_ladder,
    temperature_names,
)

# The files every OpenMM engine writes, referenced from its Beamtime run.
OUTPUT_FILES = (
    "trajectory.dcd",
    "energies.csv",
    "forces.npy",
    "initial_structure.pdb",
    "final_structure.pdb",
    "simulation_summary.txt",
)


@dataclass
class BuiltSystem:
    """The pieces an OpenMM engine produces before simulation starts.

    :param topology: OpenMM topology describing the (possibly solvated) system.
    :param system: The parameterised OpenMM system (forces added).
    :param positions: Initial atomic/bead positions.
    :param box_vectors: Optional periodic box vectors to set on the context
        (needed for systems loaded from GRO files)."""

    topology: app.Topology
    system: openmm.System
    positions: object
    box_vectors: object | None = None


class Engine(ABC):
    """Base class for all simulation engines.

    :param config: The engine-specific configuration model.
    :param data_root: Root directory for resolving input/output paths.
    :param logger: Logger for progress messages.
    :param resume: Continue from the checkpoint of an earlier run (replica exchange only)."""

    def __init__(
        self, config, data_root: Path, logger: logging.Logger, resume: bool = False
    ) -> None:
        self.config = config
        self.data_root = data_root
        self.logger = logger
        self.resume = resume

    @abstractmethod
    def run(self) -> Path:
        """Run the simulation end-to-end.

        :return: The output directory containing the results."""
        raise NotImplementedError


class OpenMMEngine(Engine):
    """Shared OpenMM flow: build → integrate → minimise/equilibrate → run.

    Concrete engines implement :meth:`build_system`; velocity initialisation
    can be toggled via :attr:`initialize_velocities`."""

    initialize_velocities: bool = True

    @abstractmethod
    def build_system(self) -> BuiltSystem:
        """Build the OpenMM system for this engine.

        :return: A :class:`BuiltSystem` with topology, system, and positions."""
        raise NotImplementedError

    def run(self) -> Path:
        """Run the simulation as one Beamtime run, or as a group of runs for replica exchange.

        :return: The output directory containing the results."""
        output_dir = self.config.output_dir(self.data_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.replica_exchange is not None:
            self.run_replica_exchange(output_dir)
            return output_dir
        if self.resume:
            raise ValueError("resume is only supported for replica-exchange runs")
        with start_run(self.config, output_dir) as run:
            self.simulate(output_dir, run)
            record_outputs(run, [output_dir / name for name in OUTPUT_FILES])
        return output_dir

    def simulate(self, output_dir: Path, run: Run) -> None:
        """Build, minimise, equilibrate and run the simulation, logging to *run*.

        :param output_dir: The directory to write into.
        :param run: The Beamtime run."""
        cfg = self.config
        built = self.build_system()
        if cfg.frozen_atoms is not None:
            freeze_atoms(built.system, built.topology, cfg.frozen_atoms, self.logger)

        integrator = create_integrator(
            cfg.temperature, cfg.friction, cfg.timestep, kind=cfg.integrator
        )

        self.logger.info("Creating simulation...")
        simulation = app.Simulation(built.topology, built.system, integrator)
        simulation.context.setPositions(built.positions)
        if built.box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*built.box_vectors)

        # Save the starting frame and parameter summary up front, before the run.
        save_initial_structure(simulation, output_dir, self.logger)
        write_simulation_summary(output_dir, cfg, simulation, self.logger)

        if cfg.minimize_energy:
            self.logger.info("Minimizing energy...")
            simulation.minimizeEnergy()
        if self.initialize_velocities:
            self.logger.info(f"Initializing velocities at {cfg.temperature} K...")
            simulation.context.setVelocitiesToTemperature(
                cfg.temperature * unit.kelvin  # type: ignore[arg-type]
            )

        self.logger.info("Setting up reporters...")
        add_standard_reporters(
            simulation,
            output_dir,
            report_interval=cfg.report_interval,
            output_freq=cfg.output_freq,
            num_steps=cfg.num_steps,
            logger=self.logger,
            write_forces=cfg.write_forces,
        )
        simulation.reporters.append(BeamtimeReporter(run=run, report_interval=cfg.report_interval))

        self.logger.info(
            f"Running {cfg.engine} MD for {cfg.num_steps:,} steps "
            f"(dt={cfg.timestep} ps, T={cfg.temperature} K)..."
        )
        run_simulation_loop(simulation, cfg.num_steps, cfg.output_freq)

        save_final_structure(simulation, output_dir, self.logger)

        self.logger.info("Simulation completed successfully!")
        self.logger.info(f"Output files: {output_dir}")

    def run_replica_exchange(self, output_dir: Path) -> None:
        """Run T-REMD with one Beamtime run per temperature, resuming from a checkpoint if asked.

        On resume the runs of the checkpointed attempt are continued as new attempts, so the logs
        of one simulation stay in one run per temperature; see
        :func:`md_simulations.core.resume_replica_runs`.

        :param output_dir: The directory to write into."""
        cfg = self.config
        state = read_checkpoint(output_dir)
        if state is not None and not self.resume:
            raise FileExistsError(
                f"{output_dir / CHECKPOINT_DIR} holds a checkpoint: pass --resume to continue "
                "it, or remove it to start over."
            )
        if state is None and self.resume:
            raise FileNotFoundError(
                f"no checkpoint to resume from in {output_dir / CHECKPOINT_DIR}"
            )

        temperatures = resolve_ladder(cfg)
        names = temperature_names(temperatures)
        with ExitStack() as stack:
            if state is None:
                runs = start_replica_runs(cfg, output_dir, names)
            else:
                runs = resume_replica_runs(cfg, output_dir, names, state["run_ids"], state["step"])
            runs = [stack.enter_context(run) for run in runs]

            built = self.build_system()
            if cfg.frozen_atoms is not None:
                freeze_atoms(built.system, built.topology, cfg.frozen_atoms, self.logger)
            ReplicaExchange(
                topology=built.topology,
                system=built.system,
                positions=built.positions,
                box_vectors=built.box_vectors,
                config=cfg,
                output_dir=output_dir,
                runs=runs,
                logger=self.logger,
                initialize_velocities=self.initialize_velocities,
            ).run(state)

            shared = [EXCHANGES_FILE, "simulation_summary.txt"]
            for k, run in enumerate(runs):
                folder = output_dir if k == 0 else output_dir / names[k]
                files = [folder / name for name in ("energies.csv", "trajectory.dcd", "forces.npy")]
                if k == 0:
                    files += [output_dir / name for name in OUTPUT_FILES]
                record_outputs(run, files + [output_dir / name for name in shared])
