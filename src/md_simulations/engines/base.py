import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import openmm
import openmm.app as app
import openmm.unit as unit

from md_simulations.core import (
    add_standard_reporters,
    create_integrator,
    run_simulation_loop,
    save_final_structure,
    save_initial_structure,
    write_simulation_summary,
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
    :param logger: Logger for progress messages."""

    def __init__(self, config, data_root: Path, logger: logging.Logger) -> None:
        self.config = config
        self.data_root = data_root
        self.logger = logger

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
        cfg = self.config
        output_dir = cfg.output_dir(self.data_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        built = self.build_system()

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
        )

        self.logger.info(
            f"Running {cfg.engine} MD for {cfg.num_steps:,} steps "
            f"(dt={cfg.timestep} ps, T={cfg.temperature} K)..."
        )
        run_simulation_loop(simulation, cfg.num_steps, cfg.output_freq)

        save_final_structure(simulation, output_dir, self.logger)

        self.logger.info("Simulation completed successfully!")
        self.logger.info(f"Output files: {output_dir}")
        return output_dir
