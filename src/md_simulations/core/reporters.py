import sys
import logging
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

from md_simulations.core.forces import FORCES_FILE, ForceWriter


class ForceReporter:
    """Write per-atom (or per-bead) forces to a float32 ``.npy`` file.

    The array has shape (frames, atoms, 3): one frame per report interval, holding
    the x, y, z force components (kJ mol⁻¹ nm⁻¹) of every atom in *atomSubset*
    (or of all atoms when *atomSubset* is ``None``).

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
        self._file = Path(file)
        self._reportInterval = reportInterval
        self._atomSubset = atomSubset
        # The site count is only known once a frame arrives, since atomSubset may be None.
        self._writer: ForceWriter | None = None

    def __del__(self) -> None:
        try:
            if self._writer is not None:
                self._writer.close()
        except Exception:
            pass

    def describeNextReport(self, simulation: app.Simulation) -> dict:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {"steps": steps, "periodic": None, "include": ["forces"]}

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        forces = np.asarray(
            state.getForces().value_in_unit(unit.kilojoules / unit.mole / unit.nanometer)
        )
        if self._atomSubset is not None:
            forces = forces[self._atomSubset]
        if self._writer is None:
            self._writer = ForceWriter(self._file, len(forces))
        self._writer.write(forces)


def add_standard_reporters(
    simulation: app.Simulation,
    output_dir: Path,
    report_interval: int,
    output_freq: int,
    num_steps: int,
    logger: logging.Logger | None = None,
    write_forces: bool = True,
) -> None:
    """Attach the standard set of reporters to *simulation* in-place.

    Reporters added:
    - :class:`openmm.app.StateDataReporter` → stdout (energy, temperature,
      progress, speed, remaining time).
    - :class:`openmm.app.DCDReporter` → ``<output_dir>/trajectory.dcd``.
    - :class:`openmm.app.StateDataReporter` → ``<output_dir>/energies.csv``
      (step + configurational potential energy, one row per trajectory frame).
    - :class:`ForceReporter` → ``<output_dir>/forces.npy``, unless *write_forces* is false.

    :param simulation: The simulation object to attach reporters to.
    :param output_dir: Directory where trajectory and force files are written.
    :param report_interval: Steps between stdout state-data reports.
    :param output_freq: Steps between DCD frame and force writes.
    :param num_steps: Total number of steps (used by the progress reporter).
    :param logger: Optional logger for path-confirmation messages.
    :param write_forces: Whether to write ``forces.npy``.
    :return: None."""
    _log = logger or logging.getLogger(__name__)

    trajectory_dcd = output_dir / "trajectory.dcd"
    energies_csv = output_dir / "energies.csv"
    forces_npy = output_dir / FORCES_FILE

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
    if write_forces:
        simulation.reporters.append(ForceReporter(str(forces_npy), output_freq))

    _log.info(f"DCD trajectory → {trajectory_dcd}")
    _log.info(f"Energies       → {energies_csv}")
    _log.info(
        f"Forces         → {forces_npy}" if write_forces else "Forces         → (not written)"
    )
