import logging

import openmm.app as app
import openmm.unit as unit


def minimize_and_equilibrate(
    simulation: app.Simulation,
    temperature: float,
    logger: logging.Logger | None = None,
) -> None:
    """Energy-minimise the system and assign Maxwell–Boltzmann velocities.

    :param simulation: Fully initialised :class:`openmm.app.Simulation` with
        positions already set on the context.
    :param temperature: Temperature in Kelvin for velocity initialisation.
    :param logger: Optional logger for progress messages.
    :return: None."""
    _log = logger or logging.getLogger(__name__)
    _log.info("Minimizing energy...")
    simulation.minimizeEnergy()
    _log.info(f"Initializing velocities at {temperature} K...")
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)  # type: ignore[arg-type]
