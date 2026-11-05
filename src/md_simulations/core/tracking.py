"""Beamtime logging: a simulation is a run, a replica set is a group of runs."""

from pathlib import Path
from typing import Any
import time

import beamtime
import openmm
import openmm.app as app
import openmm.unit as unit
from beamtime.sdk.paths import locate
from beamtime.sdk.run import Run, new_run, to_proposal

_R = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)


def run_config(config: Any, output_dir: Path) -> dict[str, Any]:
    """Build the config a run records: the simulation config plus where its files are.

    :param config: The simulation config.
    :param output_dir: The directory the simulation writes into.
    :return: The run config."""
    return config.model_dump(mode="json") | {"output_dir": str(output_dir.resolve())}


def start_run(config: Any, output_dir: Path) -> Run:
    """Start the Beamtime run of one simulation in ``config.beamtime_group``.

    :param config: The simulation config.
    :param output_dir: The directory the simulation writes into.
    :return: The started run; use it as a context manager to record failures."""
    return beamtime.init(
        dir=config.beamtime_group,
        name=config.beamtime_name,
        tags=[config.engine],
        config=run_config(config, output_dir),
    )


def start_replica_runs(config: Any, output_dir: Path, names: list[str]) -> list[Run]:
    """Start one Beamtime run per replica, grouped below ``<beamtime_group>/<beamtime_name>``.

    Beamtime's ``init`` keeps one live run per process, so batched replicas are made with its
    internal API: they capture no output, and a killed job leaves them ``lost``, not ``killed``.

    :param config: The simulation config.
    :param output_dir: The directory the simulation writes into.
    :param names: One run name per replica.
    :return: The started runs; use each as a context manager to record failures."""
    location = locate(f"{config.beamtime_group}/{config.beamtime_name}")
    runs = []
    for i, name in enumerate(names):
        cfg = run_config(config, output_dir) | {"replica_index": i}
        run = new_run(location, name, [config.engine], cfg, to_proposal(None))
        run.start()
        runs.append(run)
    return runs


def record_outputs(run: Run, paths: list[Path]) -> None:
    """Reference the output files of a simulation from its run, leaving them where they are.

    :param run: The run.
    :param paths: The output files; missing ones are skipped."""
    for path in paths:
        if path.exists():
            run.artifact(path, kind=path.suffix.lstrip("."), copy=False)


def degrees_of_freedom(system: openmm.System) -> int:
    """Count the degrees of freedom of a system, as OpenMM's ``StateDataReporter`` does.

    :param system: The system.
    :return: Three per massive particle, less constraints and a centre-of-mass motion remover."""
    massive = [
        system.getParticleMass(i).value_in_unit(unit.dalton) > 0
        for i in range(system.getNumParticles())
    ]
    dof = 3 * sum(massive)
    for i in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(i)
        if massive[p1] or massive[p2]:
            dof -= 1
    if any(isinstance(force, openmm.CMMotionRemover) for force in system.getForces()):
        dof -= 3
    return dof


class BeamtimeReporter:
    """OpenMM reporter that logs energies, temperature and speed to a Beamtime run."""

    def __init__(self, *, run: Run, report_interval: int) -> None:
        """Create the reporter.

        :param run: The run to log to.
        :param report_interval: Steps between logged values."""
        self.run = run
        self.report_interval = report_interval
        self.dof: int | None = None
        self.last: tuple[float, float] | None = None

    def describeNextReport(self, simulation: app.Simulation) -> dict[str, Any]:
        """Ask for energies at the next multiple of the report interval.

        :param simulation: The simulation.
        :return: OpenMM's report request."""
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return {"steps": steps, "periodic": None, "include": ["energy"]}

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """Log the state at the current step.

        :param simulation: The simulation.
        :param state: The state with energies."""
        if self.dof is None:
            self.dof = degrees_of_freedom(simulation.system)
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        values = {
            "energy/potential": potential,
            "energy/kinetic": kinetic,
            "energy/total": potential + kinetic,
            "temperature": 2.0 * kinetic / (self.dof * _R) if self.dof > 0 else 0.0,
        }

        now, ps = time.time(), state.getTime().value_in_unit(unit.picosecond)
        if self.last is not None and now > self.last[0]:
            values["speed/ns_per_day"] = (ps - self.last[1]) * 86.4 / (now - self.last[0])
        self.last = (now, ps)
        self.run.log(values, step=simulation.currentStep)
