import sys
import logging
from pathlib import Path

import openmm
import openmm.app as app
import openmm.unit as unit


class ForceReporter:
    """Write per-atom (or per-bead) forces to a whitespace-delimited text file.

    Each line in the output file corresponds to one report interval and
    contains the x, y, z force components (kJ mol⁻¹ nm⁻¹) for every atom
    in *atomSubset* (or all atoms when *atomSubset* is ``None``).

    Compatible with both all-atom OpenMM topologies and the 3-bead CG
    topologies produced by OpenAWSEM.

    :param file: Path to the output force file.
    :param reportInterval: Steps between successive force writes.
    :param atomSubset: Optional list of atom indices to write. When ``None``
        all atoms are written.
    """

    def __init__(
        self,
        file: str,
        reportInterval: int,
        atomSubset: list[int] | None = None,
    ) -> None:
        self._out = open(file, "w")
        self._reportInterval = reportInterval
        self._atomSubset = atomSubset

    def __del__(self) -> None:
        try:
            self._out.close()
        except Exception:
            pass

    def describeNextReport(self, simulation: app.Simulation) -> dict:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {"steps": steps, "periodic": None, "include": ["forces"]}

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        forces = state.getForces().value_in_unit(unit.kilojoules / unit.mole / unit.nanometer)
        subset = self._atomSubset if self._atomSubset is not None else range(len(forces))
        line = (
            " ".join(f"{forces[i][0]:g} {forces[i][1]:g} {forces[i][2]:g}" for i in subset) + "\n"
        )
        self._out.write(line)
        self._out.flush()


def add_standard_reporters(
    simulation: app.Simulation,
    output_dir: Path,
    report_interval: int,
    output_freq: int,
    num_steps: int,
    logger: logging.Logger | None = None,
) -> None:
    """Attach the standard set of reporters to *simulation* in-place.

    Reporters added:
    - :class:`openmm.app.StateDataReporter` → stdout (energy, temperature,
      progress, speed, remaining time).
    - :class:`openmm.app.DCDReporter` → ``<output_dir>/trajectory.dcd``.
    - :class:`openmm.app.StateDataReporter` → ``<output_dir>/energies.csv``
      (step + configurational potential energy, one row per trajectory frame).
    - :class:`ForceReporter` → ``<output_dir>/forces.txt``.

    :param simulation: The simulation object to attach reporters to.
    :param output_dir: Directory where trajectory and force files are written.
    :param report_interval: Steps between stdout state-data reports.
    :param output_freq: Steps between DCD frame and force writes.
    :param num_steps: Total number of steps (used by the progress reporter).
    :param logger: Optional logger for path-confirmation messages.
    :return: None."""
    _log = logger or logging.getLogger(__name__)

    trajectory_dcd = output_dir / "trajectory.dcd"
    energies_csv = output_dir / "energies.csv"
    forces_txt = output_dir / "forces.txt"

    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            reportInterval=report_interval,
            step=True,
            temperature=True,
            potentialEnergy=True,
            totalEnergy=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=num_steps,
            separator="\t",
        )
    )
    simulation.reporters.append(app.DCDReporter(str(trajectory_dcd), output_freq))
    simulation.reporters.append(
        app.StateDataReporter(
            str(energies_csv),
            reportInterval=output_freq,
            step=True,
            potentialEnergy=True,
            separator=",",
        )
    )
    simulation.reporters.append(ForceReporter(str(forces_txt), output_freq))

    _log.info(f"DCD trajectory → {trajectory_dcd}")
    _log.info(f"Energies       → {energies_csv}")
    _log.info(f"Forces         → {forces_txt}")
