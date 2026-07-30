from typing import Literal

import openmm
import openmm.unit as unit


def create_integrator(
    temperature: float,
    friction: float,
    timestep: float,
    kind: Literal["langevin", "langevin_middle"] = "langevin",
) -> openmm.Integrator:
    """Create a Langevin (NVT) integrator.

    Used by all engines. No barostat is attached here; callers that need NPT
    should add a :class:`openmm.MonteCarloBarostat` to the system before
    creating the :class:`openmm.app.Simulation`.

    :param temperature: Target temperature in Kelvin.
    :param friction: Langevin friction coefficient in ps⁻¹.
    :param timestep: Integration timestep in picoseconds.
    :param kind: ``"langevin"`` for :class:`openmm.LangevinIntegrator` (default,
        matches the historical all-atom/CG setup) or ``"langevin_middle"`` for
        the more accurate :class:`openmm.LangevinMiddleIntegrator` (used for the
        GROMACS-input path to mimic GROMACS ``sd``).
    :return: Configured OpenMM integrator."""
    T = temperature * unit.kelvin
    gamma = friction / unit.picosecond
    dt = timestep * unit.picoseconds
    match kind:
        case "langevin":
            return openmm.LangevinIntegrator(T, gamma, dt)  # type: ignore[arg-type]
        case "langevin_middle":
            return openmm.LangevinMiddleIntegrator(T, gamma, dt)  # type: ignore[arg-type]
